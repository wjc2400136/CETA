from __future__ import annotations


import json


import numpy as np


import torch


from torch.utils.data import DataLoader


from torchvision import transforms


from ..data.manifest_io import write_json_atomic


from ..data.torch_dataset import RetrievalTorchDataset


from ..provenance import sha256_file


@torch.no_grad()
def _full_clean_pool(loaded, dataset, run, run_dir):
    # A one-image technical smoke still uses the complete clean source pool.
    if run.limit is not None:
        dataset = RetrievalTorchDataset(dataset.annotation_path, dataset.image_root, run.source_model)
    loader = DataLoader(dataset, batch_size=1, shuffle=False, num_workers=0,
                        collate_fn=dataset.collate_fn, pin_memory=torch.cuda.is_available())
    normalize = transforms.Normalize((0.48145466, 0.4578275, 0.40821073),
                                     (0.26862954, 0.26130258, 0.27577711))
    images_out, texts_out = [], []
    for index, (images, groups, image_ids, text_ids, _names) in enumerate(loader):
        images = images.to(run.device)
        texts = [text for group in groups for text in group]
        inputs = loaded.tokenizer(texts, padding="max_length", truncation=True,
                                  max_length=30, return_tensors="pt").to(run.device)
        # Match eval_SADCA.py's text-then-image prepass and preserve the output
        # features exactly; no extra normalization of selected negative features.
        text_features = loaded.model.inference_text(inputs)["text_feat"]
        image_features = loaded.model.inference_image(normalize(images))["image_feat"]
        images_out.append(image_features.detach())
        texts_out.append(text_features.detach())
        if index == 0 or (index + 1) % 100 == 0 or index + 1 == len(dataset):
            print(json.dumps({"stage": "sadca_full_source_pool", "images": index+1,
                              "total": len(dataset), "source": loaded.name}), flush=True)
    image_features = torch.cat(images_out, dim=0)
    text_features = torch.cat(texts_out, dim=0)
    if (image_features.dtype != torch.float32 or text_features.dtype != torch.float32
            or not torch.isfinite(image_features).all() or not torch.isfinite(text_features).all()):
        raise ValueError("SADCA source prepass returned invalid features")
    pool_dir = run_dir / "sadca-source-pool"
    pool_dir.mkdir(exist_ok=False)
    np.save(pool_dir / "image_features.npy", image_features.cpu().numpy(), allow_pickle=False)
    np.save(pool_dir / "text_features.npy", text_features.cpu().numpy(), allow_pickle=False)
    order = {"image_names": dataset.images, "clean_texts": dataset.texts,
             "caption_to_image": dataset.text2image, "image_to_captions": dataset.image2text}
    write_json_atomic(pool_dir / "ordered-inputs.json", order)
    # Read the saved arrays independently before publishing their completion.
    for name, expected in (("image_features", image_features), ("text_features", text_features)):
        saved = np.load(pool_dir / (name + ".npy"), mmap_mode="r", allow_pickle=False)
        if (saved.dtype != np.float32 or saved.shape != tuple(expected.shape)
                or not bool(np.isfinite(saved).all())
                or not np.array_equal(saved, expected.cpu().numpy())):
            raise ValueError("SADCA saved source-pool array failed full readback")
    saved_order = json.loads((pool_dir / "ordered-inputs.json").read_text(encoding="utf-8"))
    if (saved_order["image_names"] != dataset.images or saved_order["clean_texts"] != dataset.texts
            or {int(k):v for k,v in saved_order["caption_to_image"].items()} != dataset.text2image
            or {int(k):v for k,v in saved_order["image_to_captions"].items()} != dataset.image2text):
        raise ValueError("SADCA saved full source-pool order changed")
    identity = {
        "schema": "SADCA_COLD_FULL_SOURCE_NEGATIVE_POOL_V1", "model": loaded.name,
        "checkpoint_sha256": loaded.checkpoint_sha256,
        "annotation_sha256": sha256_file(dataset.annotation_path),
        "images": len(dataset), "captions": len(dataset.texts),
        "max_length": 30, "source_transform": "unchanged project native model transform",
        "build_batch_size": 1, "build_num_workers": 0, "cache_reused": False,
        "model_mode": "eval_float32_no_grad", "target_access": False,
        "arrays": {name: {"size_bytes": (pool_dir/name).stat().st_size,
                           "sha256": sha256_file(pool_dir/name)}
                   for name in ("image_features.npy", "text_features.npy", "ordered-inputs.json")},
    }
    write_json_atomic(pool_dir / "complete.json", identity)
    return {"image_features": image_features, "text_features": text_features,
            "image_ids": range(len(dataset)), "caption_ids": range(len(dataset.texts)),
            "caption_to_image": dataset.text2image, "image_to_captions": dataset.image2text,
            "identity": identity}
