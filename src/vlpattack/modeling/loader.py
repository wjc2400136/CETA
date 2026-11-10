from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

from ..catalog import canonical_name
from ..config import named_config
from ..paths import resolve_project_path
from ..provenance import sha256_file


@dataclass
class LoadedModel:
    name: str
    model: Any
    tokenizer: Any
    ref_model: Any
    config: dict[str, Any]
    checkpoint: Path | None
    checkpoint_sha256: str | None


def _legacy_config(name: str, dataset: str) -> tuple[dict[str, Any], Path | None]:
    source = named_config("models", name)
    config = {
        "image_res": int(source["image_size"]),
        "embed_dim": int(source["embed_dim"]),
        "vision_width": int(source["vision_width"]),
        "bert_config": str(resolve_project_path(source["bert_config"])),
        "batch_size_eval": 8,
        "k_test": 128,
    }
    checkpoint = resolve_project_path(source["checkpoints"][dataset])
    return config, checkpoint


def _torch_load(path: Path):
    import torch

    try:
        return torch.load(path, map_location="cpu", weights_only=True)
    except TypeError:
        return torch.load(path, map_location="cpu")


def _validate_load_result(name: str, missing: list[str], unexpected: list[str]) -> None:
    allowed_missing_fragments = (
        "visual_encoder_m",
        "text_encoder_m",
        "vision_proj_m",
        "text_proj_m",
        "queue",
        "queue_ptr",
        "temp",
        "momentum",
        "alpha",
    )
    critical_missing = [
        key for key in missing if not any(fragment in key for fragment in allowed_missing_fragments)
    ]
    allowed_unexpected_fragments = allowed_missing_fragments + (
        "bert.",
        "cls.",
        "mlm_head",
        "itm_head_m",
    )
    critical_unexpected = [
        key
        for key in unexpected
        if not any(fragment in key for fragment in allowed_unexpected_fragments)
    ]
    if critical_missing:
        preview = ", ".join(critical_missing[:8])
        raise RuntimeError(f"{name} checkpoint is missing critical keys: {preview}")
    if len(critical_unexpected) > 20:
        preview = ", ".join(critical_unexpected[:8])
        raise RuntimeError(f"{name} checkpoint has too many unexplained keys: {preview}")


def load_model(
    name: str,
    dataset: str,
    device: str,
    *,
    with_mlm: bool = True,
    local_files_only: bool = False,
) -> LoadedModel:
    import torch
    from transformers import BertForMaskedLM

    from vlpattack.legacy.models.tokenization_bert import BertTokenizer

    canonical = canonical_name(name)
    dataset = canonical_name(dataset)
    if canonical not in {"albef", "tcl", "clip_vit_b16", "clip_rn101"}:
        raise ValueError(f"Unsupported model: {name}")
    torch_device = torch.device(device)
    if torch_device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but torch.cuda.is_available() is false")

    local_bert = resolve_project_path("data/resources/huggingface/bert-base-uncased")
    required_bert_files = ("config.json", "pytorch_model.bin", "vocab.txt")
    has_local_bert = all((local_bert / name).is_file() for name in required_bert_files)
    bert_source = str(local_bert) if has_local_bert else "bert-base-uncased"
    tokenizer = BertTokenizer.from_pretrained(
        bert_source, local_files_only=local_files_only or has_local_bert
    )
    ref_model = None
    if with_mlm:
        ref_model = BertForMaskedLM.from_pretrained(
            bert_source, local_files_only=local_files_only or has_local_bert
        ).to(torch_device)
        ref_model.eval()
        ref_model.requires_grad_(False)

    if canonical in {"albef", "tcl"}:
        from vlpattack.legacy.models.albef_retrieval import ALBEF
        from vlpattack.legacy.models.vit import interpolate_pos_embed

        config, checkpoint_path = _legacy_config(canonical, dataset)
        if not checkpoint_path.is_file():
            raise FileNotFoundError(f"Missing {canonical} checkpoint: {checkpoint_path}")
        model = ALBEF(config=config, text_encoder=bert_source, tokenizer=tokenizer)
        payload = _torch_load(checkpoint_path)
        state = payload.get("model", payload) if isinstance(payload, dict) else payload
        state = dict(state)
        if canonical == "tcl" and "visual_encoder.pos_embed" in state:
            state["visual_encoder.pos_embed"] = interpolate_pos_embed(
                state["visual_encoder.pos_embed"], model.visual_encoder
            )
        for key in list(state):
            if "bert" in key:
                replacement = key.replace("bert", "")
                state[replacement] = state.pop(key)
        result = model.load_state_dict(state, strict=False)
        _validate_load_result(canonical, list(result.missing_keys), list(result.unexpected_keys))
    else:
        from vlpattack.legacy.models.clip_model import clip

        config = named_config("models", canonical)
        checkpoint_path = resolve_project_path(config["checkpoint"])
        if not checkpoint_path.is_file():
            raise FileNotFoundError(f"Missing {canonical} checkpoint: {checkpoint_path}")
        model, _ = clip.load(str(checkpoint_path), device=torch_device)
        model.set_tokenizer(tokenizer)

    model = model.float().to(torch_device).eval()
    model.requires_grad_(False)
    checkpoint_hash = sha256_file(checkpoint_path) if checkpoint_path else None
    return LoadedModel(
        name=canonical,
        model=model,
        tokenizer=tokenizer,
        ref_model=ref_model,
        config=config,
        checkpoint=checkpoint_path,
        checkpoint_sha256=checkpoint_hash,
    )
