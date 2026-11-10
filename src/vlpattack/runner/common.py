from __future__ import annotations
import random
from pathlib import Path
import numpy as np
import torch
from PIL import Image
from tqdm import tqdm
from ..data.torch_dataset import RetrievalTorchDataset


def _seed_everything(seed: int, deterministic: bool = True) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = not deterministic
    torch.backends.cudnn.deterministic = deterministic


@torch.no_grad()
def _build_projection_matrix(loaded, dataset: RetrievalTorchDataset, run):
    rng = random.Random(run.seed)
    sample_count = max(1, int(0.4 * len(dataset.texts)))
    texts = rng.sample(dataset.texts, sample_count)
    features = []
    device = next(loaded.model.parameters()).device
    for start in tqdm(range(0, len(texts), 256), desc="semantic projection"):
        batch = texts[start : start + 256]
        inputs = loaded.tokenizer(
            batch, padding="max_length", truncation=True, max_length=30, return_tensors="pt"
        ).to(device)
        features.append(loaded.model.inference_text(inputs)["text_feat"].float().cpu())
    corpus = torch.cat(features, dim=0).T
    left, _, _ = torch.linalg.svd(corpus, full_matrices=True)
    projection = left[:, 1:] @ left[:, 1:].T
    return projection


def _save_image(tensor: torch.Tensor, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    array = torch.clamp(tensor, 0.0, 1.0).mul(255).round().byte().permute(1, 2, 0).cpu().numpy()
    temporary = path.with_suffix(path.suffix + ".part")
    Image.fromarray(array).save(temporary, format="PNG")
    temporary.replace(path)
