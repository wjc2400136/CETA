from __future__ import annotations

import json
from pathlib import Path

import torch
from PIL import Image
from torch.utils.data import Dataset

from ..catalog import canonical_name
from ..config import load_yaml, named_config
from ..paths import resolve_project_path
from .schema import clean_caption, load_retrieval_dataset


def image_transform(
    model_name: str,
    *,
    defense: str = "none",
    defense_value: float | int | None = None,
    seed: int = 42,
):
    model_name = canonical_name(model_name)
    model_config = named_config("models", model_name)
    image_size = int(model_config["image_size"])
    from torchvision import transforms
    if defense != 'none':
        raise ValueError('This release implements undefended retrieval')
    geometry = ([transforms.Resize((image_size, image_size), interpolation=transforms.InterpolationMode.BICUBIC)]
                if model_config['family'] == 'fusion' else
                [transforms.Resize(image_size, interpolation=transforms.InterpolationMode.BICUBIC),
                 transforms.CenterCrop(image_size)])
    return transforms.Compose([*geometry, transforms.ToTensor()])


class RetrievalTorchDataset(Dataset):
    def __init__(
        self,
        annotation: str | Path,
        image_root: str | Path,
        model_name: str,
        *,
        limit: int | None = None,
        max_words: int = 30,
        defense: str = "none",
        defense_value: float | int | None = None,
        clean_annotation: str | Path | None = None,
        defense_seed: int = 42,
    ):
        source = load_retrieval_dataset(annotation)
        self.samples = source.samples[:limit] if limit is not None else source.samples
        self.annotation_path = source.annotation_path
        self.image_root = resolve_project_path(image_root)
        self.transform = image_transform(
            model_name,
            defense=defense,
            defense_value=defense_value,
            seed=defense_seed,
        )
        self.max_words = max_words
        self.images = [sample.image for sample in self.samples]
        self.texts: list[str] = []
        self.text2image: dict[int, int] = {}
        self.image2text: dict[int, list[int]] = {}
        clean_samples = (
            load_retrieval_dataset(clean_annotation).samples if clean_annotation else source.samples
        )
        clean_samples = clean_samples[:limit] if limit is not None else clean_samples
        if len(clean_samples) != len(self.samples):
            raise ValueError(
                "Text defenses require clean/adversarial annotations with the same image count"
            )
        for image_index, sample in enumerate(self.samples):
            clean_sample = clean_samples[image_index]
            if len(clean_sample.captions) != len(sample.captions):
                raise ValueError(
                    "Text defenses require clean/adversarial caption groups with equal lengths"
                )
            self.image2text[image_index] = []
            for caption, clean_value in zip(
                sample.captions, clean_sample.captions, strict=True
            ):
                text_index = len(self.texts)
                self.texts.append(clean_caption(caption, max_words))
                self.text2image[text_index] = image_index
                self.image2text[image_index].append(text_index)

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, index):
        sample = self.samples[index]
        path = self.image_root / sample.image
        try:
            with Image.open(path) as source:
                rgb = source.convert("RGB")
                image = self.transform(rgb)
        except FileNotFoundError as error:
            raise FileNotFoundError(
                f"Missing image {path}; check the dataset layout in the README"
            ) from error
        text_ids = self.image2text[index]
        texts = [self.texts[text_index] for text_index in text_ids]
        return image, texts, index, text_ids, sample.image

    @staticmethod
    def collate_fn(batch):
        images, text_groups, image_ids, text_id_groups, image_names = zip(*batch, strict=False)
        return (
            torch.stack(images, dim=0),
            list(text_groups),
            list(image_ids),
            list(text_id_groups),
            list(image_names),
        )

    def write_annotation(self, path: str | Path, captions: list[list[str]] | None = None) -> Path:
        output = Path(path)
        output.parent.mkdir(parents=True, exist_ok=True)
        rows = []
        for index, sample in enumerate(self.samples):
            rows.append(
                {
                    "image": sample.image,
                    "caption": captions[index] if captions is not None else list(sample.captions),
                }
            )
        output.write_text(json.dumps(rows, ensure_ascii=False, indent=2), encoding="utf-8")
        return output
