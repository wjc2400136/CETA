from __future__ import annotations

import json
import re
from dataclasses import dataclass
from pathlib import Path

from ..paths import resolve_project_path


@dataclass(frozen=True)
class RetrievalSample:
    image: str
    captions: tuple[str, ...]


@dataclass(frozen=True)
class RetrievalDataset:
    annotation_path: Path
    samples: tuple[RetrievalSample, ...]

    @property
    def image_count(self) -> int:
        return len(self.samples)

    @property
    def caption_count(self) -> int:
        return sum(len(sample.captions) for sample in self.samples)


def clean_caption(caption: str, max_words: int = 30) -> str:
    """Normalize a caption and actually apply the requested word limit."""
    value = re.sub(r"([,.'!?\"()*#:;~])", "", caption.lower())
    value = value.replace("-", " ").replace("/", " ").replace("<person>", "person")
    value = re.sub(r"\s{2,}", " ", value).strip()
    return " ".join(value.split()[:max_words])


def load_retrieval_dataset(path: str | Path, *, max_words: int | None = None) -> RetrievalDataset:
    annotation_path = resolve_project_path(path)
    try:
        raw = json.loads(annotation_path.read_text(encoding="utf-8"))
    except FileNotFoundError as error:
        raise FileNotFoundError(f"Missing dataset annotation: {annotation_path}") from error
    if not isinstance(raw, list):
        raise ValueError(f"Retrieval annotation must be a JSON list: {annotation_path}")

    samples: list[RetrievalSample] = []
    for index, item in enumerate(raw):
        if not isinstance(item, dict) or not isinstance(item.get("image"), str):
            raise ValueError(f"Invalid image record at index {index} in {annotation_path}")
        captions = item.get("caption")
        if (
            not isinstance(captions, list)
            or not captions
            or not all(isinstance(x, str) for x in captions)
        ):
            raise ValueError(f"Invalid caption list at index {index} in {annotation_path}")
        values = tuple(clean_caption(x, max_words) if max_words else x for x in captions)
        samples.append(RetrievalSample(image=item["image"].replace("\\", "/"), captions=values))
    return RetrievalDataset(annotation_path=annotation_path, samples=tuple(samples))
