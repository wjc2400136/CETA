# Reproducing the main retrieval comparison

[Home](README.md) · [中文](README.zh-CN.md)

Each attack generates one image and its group of final captions per source image. The same delivery is evaluated on four target models. Text retrieval (TR) queries the caption gallery with an image. Image retrieval (IR) queries the image gallery with a caption.

## Read the reference results

With Git and Python 3.10 or 3.11 installed, open a terminal and run

```bash
git clone https://github.com/wjc2400136/CETA.git
cd CETA
python retrieval.py plan
python retrieval.py reference
```

The reader checks the source/configuration digests, the reference CSV digest, the 640 unique cell identities and their rounded display values. In the 64 CETA cells it also recomputes ASR from the saved integer counts. For the other 576 cells the reference export contains full-precision percentages. [Reference CSV](reference/main_retrieval_r1.csv)

## Numerical environment and inputs

Use Linux x86-64 with Conda and an NVIDIA driver compatible with CUDA 12.1. Python 3.10.14, PyTorch 2.1.2, torchvision 0.16.2, NumPy 1.26.4, Transformers 4.28.1 and the pinned MKL packages match the retrieval stack. Run the following from the repository root.

```bash
conda env create -f environment-linux.yml
conda activate vlpa
python -m pip install -r requirements.txt
python -m pip check
python prepare_assets.py splits
python prepare_assets.py weights
```

For one dataset, add `--dataset flickr30k` or `--dataset mscoco` to both preparation commands. The two public Karpathy test annotations come from a fixed SimVLA commit. Flickr30K line endings are restored to the byte representation used in the original run. The weight manifest lists immutable checkpoint digests, byte counts, fetch locations, the BERT tokenizer and its masked language model. [Asset manifest](configs/assets.json)

Obtain the original image archives under the applicable dataset terms. Substitute the local paths below. The installer reads the named test images, compares existing files with the matching archive members and verifies the ordered tree digest. On a conflict it reports the image path. Move that file aside and repeat the command. [Flickr30K archive](https://huggingface.co/datasets/gondimjoaom/flickr30k-trainready/resolve/c3fcb3bd59659d98d3fa17c5eaff538e2f17060d/flickr30k-images.tar.gz) · [COCO val2014](https://s3.amazonaws.com/images.cocodataset.org/zips/val2014.zip)

```bash
python prepare_assets.py images --dataset flickr30k --archive /path/to/flickr30k-images.tar.gz
python prepare_assets.py images --dataset mscoco --archive /path/to/val2014.zip
python retrieval.py check-inputs --dataset flickr30k
python retrieval.py check-inputs --dataset mscoco
```

Existing files can also be placed in the paths named by the manifests. The test trees are `data/flickr30k/flickr30k-images/` and `data/mscoco/val2014/`. Input checks cover split files, the ordered image tree and model/MLM files. Flickr30K has 1,000 images with 5,000 captions. MSCOCO has 5,000 with 25,010. Preserve each complete caption group and its original order.

## Generate and evaluate

Start with eight complete groups to exercise generation, PNG serialization and target evaluation. A direction with zero clean hits has a null ASR in a smoke run.

```bash
python retrieval.py generate --dataset flickr30k --method ceta --source albef --limit 8 --output runs/smoke
python retrieval.py evaluate --dataset flickr30k --method ceta --source albef --target clip_vit_b16 --limit 8 --output runs/smoke
```

For a full source case, omit `--limit`. Use the same case directory and identity for evaluation.

```bash
python retrieval.py generate --dataset flickr30k --method ceta --source albef --output runs/flickr-ceta-albef
python retrieval.py evaluate --dataset flickr30k --method ceta --source albef --target all --output runs/flickr-ceta-albef
python retrieval.py matrix --output runs/full --device cuda:0
```

The last command runs the entire matrix serially. It performs 80 generations and 320 source-target evaluations for two splits, ten attacks and four source and target models. A full matrix run through the public entry point has no measured runtime yet. Allow ample scratch space for fusion-model feature extraction, especially for MSCOCO. Successful target evaluation releases its temporary features. [Experiment protocol](configs/main.json)

`plan` and `matrix` accept the same dataset, method, source and target filters. Omit a filter to include every value along that dimension. For example, these commands select one generation, one target evaluation and two ASR@1 cells. The matrix can reuse a completed generation in the same output tree when you later select another target.

```bash
python retrieval.py plan --dataset flickr30k --method ceta --source albef --target tcl
python retrieval.py matrix --dataset flickr30k --method ceta --source albef --target tcl --output runs/full
```

Method keys are `pgd`, `bert_attack`, `sep_attack`, `co_attack`, `sga`, `dra`, `sa_aet`, `simvla`, `sadca` and `ceta`. Model keys are `albef`, `tcl`, `clip_vit_b16` and `clip_rn101`. The clean and adversarial inputs use the target's native preprocessing. Fusion targets rerank their top 128 candidates. CLIP ranks using embedding similarity.

Generation uses seed 42, batches of four complete image groups, a raw-RGB bound of 2/255, a base image step of 0.5/255 and at most one original word position per caption. Captions are cleaned to 30 whitespace-delimited words. Native token limits are 30 for ALBEF/TCL and 77 for CLIP, including special tokens. The native image grid is 384 pixels square for ALBEF/TCL and 224 for CLIP.

CETA's main recipe uses a 9×9 box filter, scale range 0.50, noise 8/255, ten three-view updates and unit momentum. Four image groups are optimized together, with single-image physical forwards at the image endpoint. Main-table SADCA builds its source-only negative pool on the full clean split and uses five alignment steps and two rounds of five image updates. SA-AET constructs its semantic projection from the clean captions.

SimVLA's authors identified a one-position shift required when mapping word spans to MLM output including the leading CLS token for SGA, DRA and SA-AET. This benchmark also corrects Sep-Attack and Co-Attack where the same coordinate convention applies. SGA, DRA and SA-AET use text KL for position importance and negative dot product for candidate scoring. Co-Attack uses its corrected final text in the image objective. [Upstream explanation](https://github.com/RYC-98/SimVLA#bug-fix) · [Sources](SOURCES.md)

## Read outputs and recover

Matrix cases live under `runs/full/<dataset>/<method>/<source>/`. Each has lossless PNGs under `gallery/`, the final caption groups in `gallery.json` and a versioned digest receipt in `generation.json`. For each target, `evaluation/<target>/clean.npz` and `adversarial.npz` contain zero-based ranks in both directions. In `metrics.json`, `clean_hits` counts original rank-zero queries and `lost_clean_hits` counts how many moved below rank one. ASR@1 is 100 times their ratio.

- If an input fails verification, compare its dataset, directory layout and digest with the manifests. Move an incorrect file aside before downloading it again. A BERT snapshot should contain exactly the four listed files.
- After interruption, evaluate the saved gallery if `generation.json` exists. For each completed target the runner checks the generation and target identity, the hashes of both rank files and the metrics file, and both TR/IR rows against the ranks. It also accepts the previous release's gallery when the generator and protocol pins match. If a target lacks `complete.json`, rename its partial directory and evaluate that target again. Use a fresh case directory for an interrupted generation.
- After a CUDA OOM, release other GPU workloads or use a larger GPU with the same protocol. For an MKL `iJIT_NotifyEvent` load failure, recreate the environment using the pinned Linux YAML.
- The public packaging checks cover code, saved reference data, metadata commands and module imports. The reference measurements came from the original experiments. A fresh GPU matrix through this packaging has not been executed.
