from __future__ import annotations

import hashlib
import inspect
import json
import os
import tempfile
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
from numpy.lib.format import open_memmap
from torch.utils.data import DataLoader
from torchvision import transforms
from tqdm import tqdm

from ..data.manifest_io import write_json_atomic
from ..data.torch_dataset import RetrievalTorchDataset
from ..io_integrity import exclusive_file_lock, fsync_directory, fsync_file
from ..provenance import sha256_file
from ..records import MetricRecord


@dataclass(frozen=True)
class RetrievalRanks:
    text_retrieval: np.ndarray
    image_retrieval: np.ndarray

    def save(self, path: str | Path) -> Path:
        output = Path(path)
        output.parent.mkdir(parents=True, exist_ok=True)
        self.validate()
        descriptor, temporary_name = tempfile.mkstemp(
            dir=output.parent,
            prefix=f".{output.name}.",
            suffix=".part",
        )
        temporary = Path(temporary_name)
        try:
            with os.fdopen(descriptor, "wb") as handle:
                np.savez_compressed(
                    handle,
                    text_retrieval=self.text_retrieval,
                    image_retrieval=self.image_retrieval,
                )
                handle.flush()
                os.fsync(handle.fileno())
            os.replace(temporary, output)
            fsync_directory(output.parent)
        except BaseException:
            temporary.unlink(missing_ok=True)
            raise
        return output

    @classmethod
    def load(
        cls,
        path: str | Path,
        *,
        expected_images: int | None = None,
        expected_texts: int | None = None,
    ) -> RetrievalRanks:
        with np.load(path, allow_pickle=False) as payload:
            if set(payload.files) != {"text_retrieval", "image_retrieval"}:
                raise ValueError(f"Unexpected retrieval-rank arrays in {path}: {payload.files}")
            ranks = cls(
                np.array(payload["text_retrieval"], copy=True),
                np.array(payload["image_retrieval"], copy=True),
            )
        ranks.validate(expected_images=expected_images, expected_texts=expected_texts)
        return ranks

    def validate(
        self,
        *,
        expected_images: int | None = None,
        expected_texts: int | None = None,
    ) -> None:
        arrays = {
            "text_retrieval": (self.text_retrieval, expected_images),
            "image_retrieval": (self.image_retrieval, expected_texts),
        }
        for name, (array, expected_length) in arrays.items():
            if not isinstance(array, np.ndarray) or array.ndim != 1:
                raise ValueError(f"{name} must be a one-dimensional NumPy array")
            if not np.issubdtype(array.dtype, np.integer):
                raise ValueError(f"{name} must use an integer dtype, got {array.dtype}")
            if expected_length is not None and len(array) != expected_length:
                raise ValueError(
                    f"{name} length mismatch: {len(array)} != {expected_length}"
                )
            if bool((array < 0).any()):
                raise ValueError(f"{name} contains a negative rank")
        if expected_texts is not None and bool(
            (self.text_retrieval >= expected_texts).any()
        ):
            raise ValueError("text_retrieval contains a rank outside the text corpus")
        if expected_images is not None and bool(
            (self.image_retrieval >= expected_images).any()
        ):
            raise ValueError("image_retrieval contains a rank outside the image corpus")


class FeatureStore:
    def __init__(self, root: Path):
        self.root = root
        self.arrays: dict[str, np.ndarray] = {}

    def create(self, name: str, shape: tuple[int, ...], dtype) -> np.ndarray:
        path = self.root / f"{name}.npy"
        array = open_memmap(path, mode="w+", dtype=dtype, shape=shape)
        self.arrays[name] = array
        return array

    def open(self, name: str) -> np.ndarray:
        if name not in self.arrays:
            self.arrays[name] = np.load(self.root / f"{name}.npy", mmap_mode="r")
        return self.arrays[name]

    def flush(self) -> None:
        for array in self.arrays.values():
            if hasattr(array, "flush"):
                array.flush()


FEATURE_CACHE_SCHEMA = 3


def feature_implementation_sha256(loaded) -> str:
    files = {Path(__file__).resolve(), Path(inspect.getsourcefile(RetrievalTorchDataset)).resolve()}
    model_source = inspect.getsourcefile(type(loaded.model))
    if model_source:
        files.add(Path(model_source).resolve())
    inventory = [
        {"path": path.name, "sha256": sha256_file(path)}
        for path in sorted(files, key=lambda value: value.as_posix())
    ]
    payload = json.dumps(inventory, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def _expected_feature_arrays(model_name: str) -> set[str]:
    names = {"image_features", "text_features"}
    if model_name in {"albef", "tcl", "blip"}:
        names.update({"image_embeds", "text_atts"})
        names.add("text_ids" if model_name == "blip" else "text_embeds")
    return names


def _all_finite(array: np.ndarray) -> bool:
    if not np.issubdtype(array.dtype, np.number):
        return True
    if not np.issubdtype(array.dtype, np.floating):
        return True
    chunk = max(1, min(len(array), 32)) if array.ndim else 1
    if array.ndim == 0:
        return bool(np.isfinite(array))
    for start in range(0, len(array), chunk):
        if not bool(np.isfinite(array[start : start + chunk]).all()):
            return False
    return True


def _array_descriptor(path: Path) -> dict[str, object]:
    if path.is_symlink() or not path.is_file():
        raise ValueError(f"Feature array is not a regular file: {path}")
    array = np.load(path, mmap_mode="r", allow_pickle=False)
    if not _all_finite(array):
        raise ValueError(f"Feature array contains a non-finite value: {path}")
    return {
        "file": path.name,
        "size_bytes": path.stat().st_size,
        "sha256": sha256_file(path),
        "shape": list(array.shape),
        "dtype": str(array.dtype),
    }


def verify_feature_cache(
    cache_dir: Path,
    *,
    model_name: str,
    checkpoint_sha256: str | None,
    expected_images: int,
    expected_texts: int,
    implementation_sha256: str,
    deep: bool = True,
) -> dict[str, object]:
    if cache_dir.is_symlink() or not cache_dir.is_dir():
        raise ValueError(f"Feature cache is not a regular directory: {cache_dir}")
    complete = cache_dir / "complete.json"
    if complete.is_symlink() or not complete.is_file():
        raise ValueError(f"Feature cache has no regular completion manifest: {cache_dir}")
    try:
        metadata = json.loads(complete.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise ValueError(f"Invalid feature completion manifest: {complete}") from error
    if metadata.get("schema_version") != FEATURE_CACHE_SCHEMA:
        raise ValueError(
            f"Unsupported feature-cache schema at {complete}: "
            f"{metadata.get('schema_version')!r}"
        )
    expected_identity = {
        "model": model_name,
        "checkpoint_sha256": checkpoint_sha256,
        "images": expected_images,
        "texts": expected_texts,
        "cache_key": cache_dir.name,
        "implementation_sha256": implementation_sha256,
    }
    for key, expected in expected_identity.items():
        if metadata.get(key) != expected:
            raise ValueError(
                f"Feature-cache identity mismatch for {key}: "
                f"{metadata.get(key)!r} != {expected!r}"
            )

    expected_names = _expected_feature_arrays(model_name)
    allowed_files = {f"{name}.npy" for name in expected_names} | {"complete.json"}
    actual_files = {path.name for path in cache_dir.iterdir() if path.is_file()}
    unexpected = sorted(actual_files - allowed_files)
    missing = sorted(allowed_files - actual_files)
    if unexpected or missing:
        raise ValueError(
            f"Feature-cache file-set mismatch at {cache_dir}: "
            f"missing={missing}, unexpected={unexpected}"
        )
    if any(path.is_dir() or path.is_symlink() for path in cache_dir.iterdir()):
        raise ValueError(f"Feature cache contains a nested directory or symlink: {cache_dir}")

    recorded_arrays = metadata.get("arrays")
    if not isinstance(recorded_arrays, dict) or set(recorded_arrays) != expected_names:
        raise ValueError(f"Feature-cache array inventory mismatch at {complete}")
    descriptors: dict[str, dict[str, object]] = {}
    for name in sorted(expected_names):
        path = cache_dir / f"{name}.npy"
        array = np.load(path, mmap_mode="r", allow_pickle=False)
        descriptor = {
            "file": path.name,
            "size_bytes": path.stat().st_size,
            "shape": list(array.shape),
            "dtype": str(array.dtype),
        }
        if deep:
            if not _all_finite(array):
                raise ValueError(f"Feature array contains a non-finite value: {path}")
            descriptor["sha256"] = sha256_file(path)
        recorded = recorded_arrays[name]
        for key, value in descriptor.items():
            if not isinstance(recorded, dict) or recorded.get(key) != value:
                raise ValueError(
                    f"Feature-cache descriptor mismatch for {name}.{key}: "
                    f"{None if not isinstance(recorded, dict) else recorded.get(key)!r} "
                    f"!= {value!r}"
                )
        descriptors[name] = descriptor

    image_features = np.load(cache_dir / "image_features.npy", mmap_mode="r")
    text_features = np.load(cache_dir / "text_features.npy", mmap_mode="r")
    if image_features.ndim != 3 or image_features.shape[0] != expected_images:
        raise ValueError("image_features has an invalid shape")
    if text_features.ndim != 2 or text_features.shape[0] != expected_texts:
        raise ValueError("text_features has an invalid shape")
    if str(image_features.dtype) != "float32" or str(text_features.dtype) != "float32":
        raise ValueError("Aligned feature arrays must use float32")
    image_views = int(image_features.shape[1])
    if metadata.get("image_views") != image_views:
        raise ValueError("Feature-cache image-view count does not match image_features")
    if model_name in {"albef", "tcl", "blip"}:
        image_embeds = np.load(cache_dir / "image_embeds.npy", mmap_mode="r")
        text_atts = np.load(cache_dir / "text_atts.npy", mmap_mode="r")
        if image_embeds.ndim != 4 or image_embeds.shape[:2] != (
            expected_images,
            image_views,
        ):
            raise ValueError("image_embeds has an invalid shape")
        if text_atts.ndim != 2 or text_atts.shape[0] != expected_texts:
            raise ValueError("text_atts has an invalid shape")
        if str(image_embeds.dtype) != "float16" or str(text_atts.dtype) != "uint8":
            raise ValueError("Fusion image embeddings or text attention use an invalid dtype")
        token_name = "text_ids" if model_name == "blip" else "text_embeds"
        token_array = np.load(cache_dir / f"{token_name}.npy", mmap_mode="r")
        if token_array.ndim not in {2, 3} or token_array.shape[0] != expected_texts:
            raise ValueError(f"{token_name} has an invalid shape")
    return metadata


def _prepare_incomplete_cache(cache_dir: Path, expected_names: set[str]) -> None:
    cache_dir.mkdir(parents=True, exist_ok=True)
    if cache_dir.is_symlink():
        raise ValueError(f"Refusing a symlink feature cache: {cache_dir}")
    removable = {f"{name}.npy" for name in expected_names}
    removable.update({"complete.json"})
    for path in cache_dir.iterdir():
        if path.is_symlink() or path.is_dir():
            raise ValueError(f"Refusing unexpected entry in incomplete cache: {path}")
        if path.name not in removable and not path.name.endswith(".part"):
            raise ValueError(f"Refusing unexpected file in incomplete cache: {path}")
        path.unlink()


def _normalizer():
    return transforms.Normalize(
        (0.48145466, 0.4578275, 0.40821073),
        (0.26862954, 0.26130258, 0.27577711),
    )


@torch.no_grad()
def extract_features(
    loaded, dataset: RetrievalTorchDataset, batch_size: int, cache_dir: Path
) -> FeatureStore:
    cache_dir = Path(cache_dir)
    lock_path = cache_dir.parent / f".{cache_dir.name}.lock"
    checkpoint_sha256 = getattr(loaded, "checkpoint_sha256", None)
    implementation_sha256 = feature_implementation_sha256(loaded)
    expected_names = _expected_feature_arrays(loaded.name)
    with exclusive_file_lock(lock_path):
        complete = cache_dir / "complete.json"
        if complete.is_file():
            verify_feature_cache(
                cache_dir,
                model_name=loaded.name,
                checkpoint_sha256=checkpoint_sha256,
                expected_images=len(dataset),
                expected_texts=len(dataset.texts),
                implementation_sha256=implementation_sha256,
                deep=True,
            )
            return FeatureStore(cache_dir)
        _prepare_incomplete_cache(cache_dir, expected_names)
        store = _extract_feature_arrays(loaded, dataset, batch_size, cache_dir)
        for name in sorted(expected_names):
            fsync_file(cache_dir / f"{name}.npy")
        fsync_directory(cache_dir)
        arrays = {
            name: _array_descriptor(cache_dir / f"{name}.npy")
            for name in sorted(expected_names)
        }
        manifest = {
            "schema_version": FEATURE_CACHE_SCHEMA,
            "model": loaded.name,
            "checkpoint_sha256": checkpoint_sha256,
            "images": len(dataset),
            "texts": len(dataset.texts),
            "image_views": int(store.open("image_features").shape[1]),
            "cache_key": cache_dir.name,
            "implementation_sha256": implementation_sha256,
            "arrays": arrays,
        }
        write_json_atomic(complete, manifest)
        verify_feature_cache(
            cache_dir,
            model_name=loaded.name,
            checkpoint_sha256=checkpoint_sha256,
            expected_images=len(dataset),
            expected_texts=len(dataset.texts),
            implementation_sha256=implementation_sha256,
            deep=True,
        )
        return store


@torch.no_grad()
def _extract_feature_arrays(
    loaded, dataset: RetrievalTorchDataset, batch_size: int, cache_dir: Path
) -> FeatureStore:
    store = FeatureStore(cache_dir)
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=0,
        collate_fn=dataset.collate_fn,
        pin_memory=torch.cuda.is_available(),
    )
    model = loaded.model
    device = next(model.parameters()).device
    normalize = _normalizer()
    initialized = False

    for images, text_groups, image_ids, text_id_groups, _ in tqdm(
        loader, desc=f"features:{loaded.name}"
    ):
        images = images.to(device, non_blocking=True)
        view_count = images.shape[1] if images.ndim == 5 else 1
        if images.ndim == 5:
            images = images.flatten(0, 1)
        images = normalize(images)
        texts = [text for group in text_groups for text in group]
        text_ids = [text_id for group in text_id_groups for text_id in group]
        if loaded.name in {"albef", "tcl", "blip"}:
            text_input = loaded.tokenizer(
                texts,
                padding="max_length",
                truncation=True,
                max_length=30,
                return_tensors="pt",
            ).to(device)
            image_output = model.inference_image(images)
            text_output = model.inference_text(text_input)
            image_features = image_output["image_feat"]
            text_features = text_output["text_feat"]
        else:
            output = model.inference(images, texts)
            image_output = {"image_feat": output["image_feat"]}
            text_output = {"text_feat": output["text_feat"]}
            image_features = output["image_feat"].float()
            text_features = output["text_feat"].float()

        image_features = image_features.reshape(
            -1, view_count, image_features.shape[-1]
        )
        image_output["image_feat"] = image_features
        if "image_embed" in image_output:
            image_embed = image_output["image_embed"]
            image_output["image_embed"] = image_embed.reshape(
                -1, view_count, *image_embed.shape[1:]
            )

        if not initialized:
            store.create(
                "image_features",
                (len(dataset), image_features.shape[1], image_features.shape[-1]),
                np.float32,
            )
            store.create("text_features", (len(dataset.texts), text_features.shape[-1]), np.float32)
            if loaded.name in {"albef", "tcl", "blip"}:
                store.create(
                    "image_embeds",
                    (len(dataset), *image_output["image_embed"].shape[1:]),
                    np.float16,
                )
                store.create(
                    "text_atts", (len(dataset.texts), text_input.attention_mask.shape[-1]), np.uint8
                )
                if loaded.name == "blip":
                    store.create(
                        "text_ids", (len(dataset.texts), text_input.input_ids.shape[-1]), np.int32
                    )
                else:
                    text_embed = text_output["text_embed"]
                    store.create(
                        "text_embeds", (len(dataset.texts), *text_embed.shape[1:]), np.float16
                    )
            initialized = True

        store.open("image_features")[image_ids] = image_output["image_feat"].float().cpu().numpy()
        store.open("text_features")[text_ids] = text_features.float().cpu().numpy()
        if loaded.name in {"albef", "tcl", "blip"}:
            store.open("image_embeds")[image_ids] = image_output["image_embed"].half().cpu().numpy()
            store.open("text_atts")[text_ids] = text_input.attention_mask.byte().cpu().numpy()
            if loaded.name == "blip":
                store.open("text_ids")[text_ids] = text_input.input_ids.int().cpu().numpy()
            else:
                store.open("text_embeds")[text_ids] = text_output["text_embed"].half().cpu().numpy()

    if not initialized:
        raise ValueError("Cannot extract features from an empty dataset")
    store.flush()
    return store


def _tensor(array: np.ndarray, indices=None, *, dtype=torch.float32, device=None):
    selected = array if indices is None else array[indices]
    # Copy from a read-only memmap before torch conversion.
    value = torch.from_numpy(np.array(selected, copy=True)).to(dtype=dtype)
    return value.to(device) if device is not None else value


def _aligned_ranks(store: FeatureStore, dataset: RetrievalTorchDataset, device) -> RetrievalRanks:
    # Averaging per-view dot products is exactly equivalent to using the mean
    # of individually normalized image features.
    image_features = _tensor(store.open("image_features"), device=device).mean(dim=1)
    text_features = _tensor(store.open("text_features"), device=device)
    image_ranks = np.empty(len(dataset), dtype=np.int32)
    text_ranks = np.empty(len(dataset.texts), dtype=np.int32)

    chunk_size = 256
    for start in tqdm(range(0, len(dataset), chunk_size), desc="rank:TR"):
        end = min(start + chunk_size, len(dataset))
        similarities = image_features[start:end] @ text_features.T
        for offset, image_index in enumerate(range(start, end)):
            positives = dataset.image2text[image_index]
            best_positive = similarities[offset, positives].max()
            image_ranks[image_index] = int((similarities[offset] > best_positive).sum().item())
    for start in tqdm(range(0, len(dataset.texts), chunk_size), desc="rank:IR"):
        end = min(start + chunk_size, len(dataset.texts))
        similarities = text_features[start:end] @ image_features.T
        targets = torch.tensor(
            [dataset.text2image[index] for index in range(start, end)], device=device
        )
        target_scores = similarities[torch.arange(end - start, device=device), targets]
        text_ranks[start:end] = (similarities > target_scores.unsqueeze(1)).sum(dim=1).cpu().numpy()
    return RetrievalRanks(image_ranks, text_ranks)


@torch.no_grad()
def _fusion_ranks(
    loaded, store: FeatureStore, dataset: RetrievalTorchDataset, device
) -> RetrievalRanks:
    model = loaded.model
    image_feature_views = _tensor(store.open("image_features"), device=device)
    image_features = image_feature_views.mean(dim=1)
    text_features = _tensor(store.open("text_features"), device=device)
    image_embeds = store.open("image_embeds")
    text_atts = store.open("text_atts")
    text_embeds = store.open("text_embeds") if loaded.name != "blip" else None
    text_ids = store.open("text_ids") if loaded.name == "blip" else None
    k_i2t = min(int(loaded.config.get("k_test", 128)), len(dataset.texts))
    k_t2i = min(int(loaded.config.get("k_test", 128)), len(dataset))
    image_ranks = np.full(len(dataset), k_i2t, dtype=np.int32)
    text_ranks = np.full(len(dataset.texts), k_t2i, dtype=np.int32)

    for image_index in tqdm(range(len(dataset)), desc="rerank:TR"):
        similarity = image_features[image_index] @ text_features.T
        candidate_ids = similarity.topk(k_i2t).indices.cpu().numpy()
        image_views = _tensor(image_embeds, [image_index], device=device).squeeze(0)
        view_count = image_views.shape[0]
        encoder_output = (
            image_views.unsqueeze(1)
            .expand(view_count, k_i2t, *image_views.shape[1:])
            .reshape(view_count * k_i2t, *image_views.shape[1:])
        )
        encoder_attention = torch.ones(encoder_output.shape[:-1], dtype=torch.long, device=device)
        candidate_atts = (
            _tensor(text_atts, candidate_ids, dtype=torch.long, device=device)
            .unsqueeze(0)
            .expand(view_count, -1, -1)
            .reshape(view_count * k_i2t, -1)
        )
        if loaded.name == "blip":
            candidate_text = (
                _tensor(text_ids, candidate_ids, dtype=torch.long, device=device)
                .unsqueeze(0)
                .expand(view_count, -1, -1)
                .reshape(view_count * k_i2t, -1)
            )
            candidate_text[:, 0] = model.tokenizer.enc_token_id
            output = model.text_encoder(
                input_ids=candidate_text,
                attention_mask=candidate_atts,
                encoder_hidden_states=encoder_output,
                encoder_attention_mask=encoder_attention,
                return_dict=True,
            )
        else:
            candidate_base = _tensor(text_embeds, candidate_ids, device=device)
            candidate_text = (
                candidate_base.unsqueeze(0)
                .expand(view_count, -1, -1, -1)
                .reshape(view_count * k_i2t, *candidate_base.shape[1:])
            )
            output = model.text_encoder(
                encoder_embeds=candidate_text,
                attention_mask=candidate_atts,
                encoder_hidden_states=encoder_output,
                encoder_attention_mask=encoder_attention,
                return_dict=True,
                mode="fusion",
            )
        scores = (
            model.itm_head(output.last_hidden_state[:, 0, :])[:, 1]
            .reshape(view_count, k_i2t)
            .mean(dim=0)
        )
        positive_positions = [
            position
            for position, text_id in enumerate(candidate_ids)
            if int(text_id) in dataset.image2text[image_index]
        ]
        if positive_positions:
            best_positive = scores[positive_positions].max()
            image_ranks[image_index] = int((scores > best_positive).sum().item())

    for text_index in tqdm(range(len(dataset.texts)), desc="rerank:IR"):
        similarity = text_features[text_index] @ image_features.T
        candidate_ids = similarity.topk(k_t2i).indices.cpu().numpy()
        candidate_image_views = _tensor(image_embeds, candidate_ids, device=device)
        view_count = candidate_image_views.shape[1]
        encoder_output = candidate_image_views.reshape(
            k_t2i * view_count, *candidate_image_views.shape[2:]
        )
        encoder_attention = torch.ones(encoder_output.shape[:-1], dtype=torch.long, device=device)
        candidate_atts = _tensor(text_atts, [text_index], dtype=torch.long, device=device).expand(
            k_t2i * view_count, -1
        )
        if loaded.name == "blip":
            candidate_text = (
                _tensor(text_ids, [text_index], dtype=torch.long, device=device)
                .expand(k_t2i * view_count, -1)
                .clone()
            )
            candidate_text[:, 0] = model.tokenizer.enc_token_id
            output = model.text_encoder(
                input_ids=candidate_text,
                attention_mask=candidate_atts,
                encoder_hidden_states=encoder_output,
                encoder_attention_mask=encoder_attention,
                return_dict=True,
            )
        else:
            candidate_text = _tensor(text_embeds, [text_index], device=device).expand(
                k_t2i * view_count, -1, -1
            )
            output = model.text_encoder(
                encoder_embeds=candidate_text,
                attention_mask=candidate_atts,
                encoder_hidden_states=encoder_output,
                encoder_attention_mask=encoder_attention,
                return_dict=True,
                mode="fusion",
            )
        scores = (
            model.itm_head(output.last_hidden_state[:, 0, :])[:, 1]
            .reshape(k_t2i, view_count)
            .mean(dim=1)
        )
        target = dataset.text2image[text_index]
        positions = np.where(candidate_ids == target)[0]
        if len(positions):
            target_score = scores[int(positions[0])]
            text_ranks[text_index] = int((scores > target_score).sum().item())
    return RetrievalRanks(image_ranks, text_ranks)


def evaluate_retrieval(
    loaded,
    dataset: RetrievalTorchDataset,
    *,
    batch_size: int,
    cache_dir: str | Path,
) -> RetrievalRanks:
    device = next(loaded.model.parameters()).device
    store = extract_features(loaded, dataset, batch_size, Path(cache_dir))
    if loaded.name in {"clip_vit_b16", "clip_rn101"}:
        return _aligned_ranks(store, dataset, device)
    return _fusion_ranks(loaded, store, dataset, device)


def records_from_ranks(
    *,
    run_id: str,
    dataset: str,
    attack: str,
    source_model: str,
    target_model: str,
    clean: RetrievalRanks,
    adversarial: RetrievalRanks,
    seed: int,
    ks: tuple[int, ...] = (1, 5, 10),
    experiment: str = "",
    variant: str = "",
    defense: str = "none",
) -> list[MetricRecord]:
    rows: list[MetricRecord] = []
    directions = {
        "tr": (clean.text_retrieval, adversarial.text_retrieval),
        "ir": (clean.image_retrieval, adversarial.image_retrieval),
    }
    for direction, (clean_ranks, adversarial_ranks) in directions.items():
        for k in ks:
            originally_correct = clean_ranks < k
            denominator = int(originally_correct.sum())
            asr = (
                100.0
                * float(np.logical_and(originally_correct, adversarial_ranks >= k).sum())
                / denominator
                if denominator
                else float("nan")
            )
            recall = 100.0 * float(originally_correct.mean())
            rows.extend(
                [
                    MetricRecord(
                        run_id,
                        dataset,
                        attack,
                        source_model,
                        target_model,
                        direction,
                        "asr",
                        k,
                        asr,
                        seed,
                        experiment,
                        variant,
                        defense,
                    ),
                    MetricRecord(
                        run_id,
                        dataset,
                        "clean",
                        source_model,
                        target_model,
                        direction,
                        "recall",
                        k,
                        recall,
                        seed,
                        experiment,
                        variant,
                        defense,
                    ),
                ]
            )
    return rows
