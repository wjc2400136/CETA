# 主检索实验复现指南

[首页](README.md) · [English](README.en.md)

公开版覆盖 Flickr30K、MSCOCO 上的主检索表。每种方法在四个源模型上分别生成攻击样本，每份样本都交给四个目标模型。TR 是用图检索文本，IR 是用文本检索图像。两种方向分别按干净检索命中数计算 ASR@1。

## 先查看结果

安装 Git 和 Python 3.10 或 3.11。在终端依次执行以下命令。此步只读取仓库中的文件。

```bash
git clone https://github.com/wjc2400136/CETA.git
cd CETA
python retrieval.py plan
python retrieval.py reference
```

最后一条命令校验源码和协议清单、参考 CSV 的 SHA-256、全表键及数值。640 行对应 2 个数据集 × 10 种方法 × 4 个源 × 4 个目标 × 2 个方向。`asr_percent` 为原精度，`asr_display` 为论文的两位小数。CETA 的 64 行另附整数分子、分母，可逐行复算。其余 576 行从原主表导出的百分比保留到原精度，该表没有给出这些行的整数。 [参考表](reference/main_retrieval_r1.csv)

## 创建环境

数值实验使用 Linux x86-64、Miniconda 或 Anaconda，以及支持 CUDA 12.1 的 NVIDIA 驱动。在仓库根目录执行。Python 3.10.14、PyTorch 2.1.2、torchvision 0.16.2、NumPy 1.26.4 和 Transformers 4.28.1 对应原检索环境。环境文件固定了兼容旧版 PyTorch 的 MKL 运行库。

```bash
conda env create -f environment-linux.yml
conda activate vlpa
python -m pip install -r requirements.txt
python -m pip check
```

## 准备输入

1. 下载精确的两份检索标注和模型权重。脚本按冻结版本取回分词器、BERT MLM、ALBEF/TCL 的数据集专用权重以及 CLIP ViT-B/16 和 RN101 的共享权重。对单个数据集准备时，给两条命令都加上 `--dataset flickr30k` 或 `--dataset mscoco`。重复执行会核对已存在文件。权重下载位置、字节数和 SHA-256 可在 [资产清单](configs/assets.json) 中查看。

```bash
python prepare_assets.py splits
python prepare_assets.py weights
```

2. 取得获准使用的原始图像归档，将下列命令中的路径替换为自己的文件位置。脚本按测试标注选图，将已有图片与归档中的对应文件逐一比对，再按原实验顺序核验文件树哈希。发现冲突时会列出图片路径，可将该文件移到其他位置后重新执行。[Flickr30K 固定版本图像归档](https://huggingface.co/datasets/gondimjoaom/flickr30k-trainready/resolve/c3fcb3bd59659d98d3fa17c5eaff538e2f17060d/flickr30k-images.tar.gz) · [COCO 官方 val2014.zip](https://s3.amazonaws.com/images.cocodataset.org/zips/val2014.zip)

```bash
python prepare_assets.py images --dataset flickr30k --archive /path/to/flickr30k-images.tar.gz
python prepare_assets.py images --dataset mscoco --archive /path/to/val2014.zip
python retrieval.py check-inputs --dataset flickr30k
python retrieval.py check-inputs --dataset mscoco
```

已有相同文件也可直接放入 `data/flickr30k/flickr30k-images/`、`data/mscoco/val2014/`、`data/annotations/` 以及资产清单列出的权重路径。最后两条命令会逐文件核验测试集、分词器和权重。Flickr30K 为 1,000 图/5,000 句，MSCOCO 为 5,000 图/25,010 句；保留标注原序、每图完整句组及重复题注。

## 从最小检查到全表

先用前 8 个完整图组核对一次生成、PNG 回读和单目标检索。样本较少时部分方向的干净命中数可能为零，输出的 ASR 为 `null`。

```bash
python retrieval.py generate --dataset flickr30k --method ceta --source albef --limit 8 --output runs/smoke
python retrieval.py evaluate --dataset flickr30k --method ceta --source albef --target clip_vit_b16 --limit 8 --output runs/smoke
```

完整运行一个源，生成一次，再评估四个目标。两步使用同一个输出目录，评估时可用 `--target tcl` 等只选一个目标。

```bash
python retrieval.py generate --dataset flickr30k --method ceta --source albef --output runs/flickr-ceta-albef
python retrieval.py evaluate --dataset flickr30k --method ceta --source albef --target all --output runs/flickr-ceta-albef
```

完整主表包括 80 次生成与 320 次源–目标评估，以下命令串行执行。目标评估使用独立临时特征目录，完成后保留整数排名、两行 ASR 和校验记录。大规模融合模型特征需要充足的磁盘空间。完整公开入口尚未计时。

```bash
python retrieval.py matrix --output runs/full --device cuda:0
```

`plan` 与 `matrix` 使用相同的数据集、方法、源和目标筛选条件。省略某一项时，该项取全部值。以下两条命令预览并运行同一组实验，共一次生成、一次目标评估、两个方向的 ASR@1。在同一矩阵目录下改用其他目标时，程序可复用已完成的生成。

```bash
python retrieval.py plan --dataset flickr30k --method ceta --source albef --target tcl
python retrieval.py matrix --dataset flickr30k --method ceta --source albef --target tcl --output runs/full
```

矩阵中每个案例位于 `runs/full/<数据集>/<方法>/<源>/`，单案例则位于传给 `--output` 的目录。`gallery/` 保存无损 PNG，`gallery.json` 保留逐图最终文本，`generation.json` 绑定版本及交付物哈希。每个 `evaluation/<目标>/` 下，`clean.npz` 与 `adversarial.npz` 保存从零计数的 TR/IR 排名。`metrics.json` 的 `clean_hits` 统计原排名 0，`lost_clean_hits` 统计攻击后离开排名 0 的原命中，ASR 等于 100 乘以后者与前者的比值。

## 方法与口径

方法键为 `pgd`、`bert_attack`、`sep_attack`、`co_attack`、`sga`、`dra`、`sa_aet`、`simvla`、`sadca`、`ceta`。源和目标键均可选 `albef`、`tcl`、`clip_vit_b16`、`clip_rn101`。共同主表协议为 seed 42、每批 4 个图组、原始 RGB 上的扰动上限 2/255、基础步长 0.5/255、每句最多一处原词位置。融合模型原生图为 384×384，CLIP 为 224×224。题注先截到 30 个空白分词，融合模型与 CLIP 的原生 tokenizer 上限分别为 30 和 77 个 token（含特殊符）。ALBEF/TCL 执行前 128 候选的融合重排，CLIP 使用特征相似度排序。 [协议](configs/main.json)

SimVLA 发布库指出 SGA、DRA 和 SA-AET 应在完整 MLM 输出中将 WordPiece 跨度向后移一格，越过开头的 `[CLS]`。本基准也对采用同一路径的 Sep-Attack、Co-Attack 修正了跨度。SGA、DRA、SA-AET 的文本位置排序用文本 KL，候选替换保留负点积评分；Co-Attack 使用修正后的最终文本参与图像目标。源文件保留了各方法自身的初始化、搜索和更新。 [上游说明](https://github.com/RYC-98/SimVLA#bug-fix) · [方法来源](SOURCES.md)

CETA 使用固定的 9×9 箱式滤波、尺度范围 0.50、噪声 8/255、10 步三视图更新和动量 1。四个图组同步优化，图像端点按原方法逐图前向。SADCA 主表采用 5 步对齐加 2 轮 × 5 次图像更新，预先从完整干净检索集构建负样本池。SA-AET 从干净题注计算语义投影矩阵。程序会从源图组生成满足本方法口径的交付物，再从磁盘按目标模型的原生预处理读取图文对。

## 恢复与核查

- 文件缺失或哈希不符时，先检查标注相对路径、权重所属数据集、Flickr30K 目录层级及 COCO 年份。下载脚本不会覆盖已存在的错误文件。将该文件移至其他位置再重新下载。
- 中断后先查案例目录中的 `generation.json`。存在时可用原评估命令继续。程序核对生成回执与目标身份、两份排名及指标文件的哈希，再从排名重算 TR/IR 两行指标。上一版已完成的图库也按原生成程序和协议的固定哈希核验。某个目标缺少 `complete.json` 时，将该目标目录改名保留，再重跑它的评估命令。生成中断且没有 `generation.json` 时，另选空目录开始该案例。
- CUDA OOM 时先释放其他 GPU 作业，或选择显存更大的设备；保留已写出的生成和已完成目标。主表批量、句组、迭代和检索范围应保持协议值。出现 `iJIT_NotifyEvent` 时按本仓库的 Linux YAML 重建环境。
- 发布核查包括源码/结果清单、元数据命令和攻击模块导入。参考结果来自原实验。公开目录的独立包装入口尚未在新机器上完成 GPU 全矩阵回放。
