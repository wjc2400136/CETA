# Method and model sources

The release exposes the main image–text retrieval experiment. The nine baseline attack modules, the retrieval-rank evaluator and the native model definitions were copied from the frozen execution source. CETA retains the Full-recipe candidate supplier, selector, projection, momentum and synchronous four-owner image kernel. The public entry point replaces machine-bound paths, request records and output bookkeeping. The source manifest identifies the files used by that entry point. [Source digests](reference/source-sha256.json)

| Method | Primary implementation or derivation | Main protocol detail |
|---|---|---|
| PGD | [Madry et al.](https://arxiv.org/abs/1706.06083) | Image-only embedding KL adaptation for retrieval |
| BERT-Attack | [BERT-Attack, 8a93610](https://github.com/LinyangLee/BERT-Attack/tree/8a936107985b4ee4b08bbfdf2c0508c64cdd5c9d) | Text-only word substitution, with the leading CLS position accounted for |
| Sep-Attack and Co-Attack | [Co-Attack, 96f2c0e](https://github.com/adversarial-for-goodness/Co-Attack/tree/96f2c0ebf743f600860ed7aa78c14b38c8883521) | Independent modalities for Sep-Attack, text-guided image attack for Co-Attack |
| SGA | [SGA, f49e692](https://github.com/Zoky-2020/SGA/tree/f49e6926cd13148e49bdbc2c3e537660e096230c) | Grouped captions, corrected MLM span, KL word importance |
| DRA | [VLPTransferAttack, 91f7832](https://github.com/SensenGao/VLPTransferAttack/tree/91f78325e7cf0c497c9bbb5be2215be652d8cd9d) | Corrected SGA text branch with DRA look-ahead |
| SA-AET | [SA-AET, e9e664a](https://github.com/jiaxiaojunQAQ/SA-AET/tree/e9e664addb2f0e4181f4851b9624dd86f6f91a4b) | Corrected text branch, original SVD projection and projected look-ahead |
| SimVLA | [SimVLA, 29a9ddc](https://github.com/RYC-98/SimVLA/tree/29a9ddc5d124ba12e2b5109f504587e103f71908) | Two-stage attack and semantic abstraction under the shared text budget |
| SADCA | [SADCA, a3072c6](https://github.com/LiYuanBoJNU/SADCA/tree/a3072c64f6355753742609d637d43f2837d7b6a1) | Four-owner 2×5 update schedule, full clean negative pool, native class-identity checks |
| CETA | [Full recipe](src/vlpattack/attacks/csga.py), [native inputs](src/vlpattack/attacks/csga_native_inputs.py), [joint update](src/vlpattack/attacks/ceta_owner_batch.py) | Fixed 9×9 reference view, complete one-position sentence enumeration and three-view updates |

Model definitions and checkpoint-loading conventions trace to [ALBEF](https://github.com/salesforce/ALBEF), [TCL](https://github.com/uta-smile/TCL), [OpenAI CLIP](https://github.com/openai/CLIP) and [Transformers](https://github.com/huggingface/transformers). Each upstream component retains its own terms. The experiment inputs remain subject to the [Flickr30K](https://shannon.cs.illinois.edu/DenotationGraph/) and [MSCOCO](https://cocodataset.org/#termsofuse) dataset terms. Source files carry upstream notices where supplied. [Third-party notices](notices/)
