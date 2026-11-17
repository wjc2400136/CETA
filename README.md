# CETA

**Beyond Word Importance: Candidate Enumeration for Transferable Vision-Language Attacks** · Image–text retrieval

[中文复现指南](README.zh-CN.md) · [English reproduction guide](README.en.md) · [Reference ASR@1](reference/main_retrieval_r1.csv)

This repository contains the ten methods in the main retrieval comparison on Flickr30K and MSCOCO. CETA enumerates complete one-word candidate sentences, selects one per caption, and uses the selected captions to guide the shared image perturbation.

The public release covers the main retrieval experiment only. Its two language guides take a new user through the reference results, Linux environment, fixed data and model assets, a small execution check, the full matrix and interrupted-run recovery. The paper's cross-task, multimodal-judgment, defense, ablation, seed and cost studies belong to the separate institutional project.

```bash
python retrieval.py plan
python retrieval.py reference
```

Both commands use the Python standard library. The second verifies all 640 rank-one reference values. The language guides cover the Linux environment, pinned inputs, generation, target evaluation and reuse.
