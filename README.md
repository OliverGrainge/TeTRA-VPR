# TeTRA‑VPR: Ternary Transformer for Efficient Visual Place Recognition

Welcome to the official codebase that accompanies the paper **“TeTRA‑VPR: A Ternary Transformer Approach for Compact Visual Place Recognition”** (ICRA 2025, pre‑print arXiv:2503.02511). This repository provides everything you need to reproduce the two‑stage training pipeline—**progressive distillation pre‑training** and **supervised fine‑tuning**—as well as scripts for evaluation and inference on common VPR benchmarks.

---

## Table of Contents

1. [Key Features](#key-features)
2. [Requirements & Installation](#requirements--installation)
3. [Repository Structure](#repository-structure)
4. [Datasets](#datasets)
5. [Configuration System](#configuration-system)
6. [Stage 1 – Progressive Distillation Pre‑Training](#stage-1--progressive-distillation-pre-training)
7. [Stage 2 – Supervised Fine‑Tuning](#stage-2--supervised-fine-tuning)
8. [Evaluation & Inference](#evaluation--inference)
9. [Pre‑Trained Checkpoints](#pre-trained-checkpoints)
10. [Reproducing the Paper](#reproducing-the-paper)
11. [Citation](#citation)
12. [License](#license)
13. [Contact](#contact)

---

## Key Features

* **Ultra‑low‑bit ViT backbone** – weights quantised to **2‑bit ternary** precision; final embeddings **1‑bit binary**.
* **Progressive Quantisation‑Aware Training** – smooth sigmoid schedule for stable convergence (Eq. 9 / 10 in the paper).
* **Multi‑level Distillation Loss** – aligns classification tokens, patch tokens and attention maps to a DinoV2‑BoQ teacher.
* **Plug‑and‑Play Aggregation** – choose between **BoQ**, **SALAD**, **MixVPR** or **GeM** via `--agg_arch`.
* **Turn‑key scripts** – `pretrain.py` and `finetune.py` reproduce all experiments with a single command.

---

## Requirements & Installation

```bash
# Create a fresh environment (tested with Python 3.10)
conda create -n tetra_vpr python=3.10
conda activate tetra_vpr

# Core dependencies
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
pip install pytorch-lightning==2.2.2 timm==0.9.16 faiss-gpu==1.7.4
pip install opencv-python scikit-learn pandas tqdm yacs

# (Optional) Mixed‑precision & BF16 support
pip install xformers==0.0.26
```

> **GPU:** All results were obtained on NVIDIA H100 (80 GB, SM90). Training also fits on ≥24 GB GPUs (A6000, 4090) with `--accumulate_grad_batches` set appropriately.

---

## Repository Structure

```text
├── configs.py              # Dataclass definitions: ModelConfig, DistillConfig, TeTRAConfig
├── pretrain.py             # Stage 1 – progressive ternary distillation
├── finetune.py             # Stage 2 – supervised binary fine‑tuning
├── data/                   # Data loaders & augmentation pipelines
├── models/                 # ViT backbone, quantisation ops, aggregation heads
├── losses/                 # Distillation + Multi‑Similarity losses
├── scripts/                # Utility scripts (evaluation, export, plotting)
└── README.md               # You are here 🚀
```

---

## Datasets

### 1. **Unlabelled distillation data**

* **San Francisco XL panoramas** and **GSV‑Cities/Images**. Place the raw JPEGs under a single root folder, e.g.:

  ```text
  /data/vpr_datasets/gsv-cities/Images/Bangkok/...
  ```

### 2. **Supervised fine‑tuning & validation**

* **GSV‑Cities** (training)   – `--train_dataset_dir`
* **MSLS**, **Pitts30k**, **Tokyo247**, **SVOX‑{Night,Rain,Snow,Sun}** (validation) – group them under `--val_dataset_dir`:

  ```text
  /data/vpr_datasets/MSLS/...
  /data/vpr_datasets/Pitts30k/...
  ...
  ```

> Dataset download links and scripts are provided in `scripts/download_datasets.sh`.

---

## Configuration System

All hyper‑parameters are defined in three `@dataclass` objects inside **`configs.py`**:

* **`ModelConfig`** – backbone & aggregation architecture, descriptor normalisation.
* **`DistillConfig`** – learning rate, batch size, augmentation level, etc. for **pre‑training**.
* **`TeTRAConfig`** – hyper‑parameters for **fine‑tuning**, freezing policy, quant schedule.

Every field can be overridden from the CLI, e.g. `--batch_size 256`. Run `python pretrain.py --help` for a complete list.

---

## Stage 1 – Progressive Distillation Pre‑Training

Train the ternary ViT backbone from scratch using unlabeled images.

```bash
python pretrain.py \
  --train_dataset_dir /data/vpr_datasets/gsv-cities/Images \
  --backbone_arch ternaryvitbase \
  --agg_arch boq \
  --lr 4e-4 \
  --batch_size 128 \
  --accumulate_grad_batches 2 \
  --max_epochs 30 \
  --weight_decay 0.01 \
  --use_attn_loss True \
  --precision bf16-mixed
```

Outputs:

* `./checkpoints/pretrain_epoch=29.ckpt` – lightning checkpoint (≈ 47 MB).
* `./logs/` – TensorBoard event files with loss curves & quant λ(t).

⚠️ **Tips**

* Use `--noramlize False` if you plan to integrate with aggregation heads that already normalise outputs.
* Mixed precision speeds up training by \~30 % on Ampere and newer GPUs.

---

## Stage 2 – Supervised Fine‑Tuning

Fine‑tune aggregation head + final ViT block using place labels.

```bash
python finetune.py \
  --pretrain_checkpoint ./checkpoints/pretrain_epoch=29.ckpt \
  --train_dataset_dir /data/vpr_datasets/gsv-cities \
  --val_dataset_dir   /data/vpr_datasets \
  --cities Bangkok London Rome "LosAngeles" \
  --agg_arch boq \
  --quant_schedule logistic \
  --freeze_backbone True \
  --freeze_all_except_last_n 1 \
  --lr 1e-4 \
  --batch_size 200 \
  --max_epochs 40 \
  --precision bf16-mixed
```

Outputs:

* `./checkpoints/finetune_boQ_epoch=39.ckpt` – final binary‑embedding model (≈ 49 MB).
* `./embeddings/*.bin` – optional cached descriptors for validation sets.

---

## Evaluation & Inference

Generate binary descriptors & compute Recall\@1 on **Tokyo247**:

```bash
python scripts/eval.py \
  --checkpoint ./checkpoints/finetune_boQ_epoch=39.ckpt \
  --dataset Tokyo247 --topk 100
```

*Uses FAISS Hamming index; expects query/reference split under the dataset folder.*

For **real‑time localisation** on a video stream:

```bash
python scripts/stream_localisation.py --checkpoint <ckpt> --source 0
```

---

## Pre‑Trained Checkpoints

| Model        | Aggregation | Descriptor Dim | R\@1 (Tokyo247) | Size  | Link                                              |
| ------------ | ----------- | -------------- | --------------- | ----- | ------------------------------------------------- |
| TeTRA‑BoQ    | BoQ         | 1024 (binary)  | **88.6 %**      | 49 MB | [download](https://example.com/tetra_boQ.ckpt)    |
| TeTRA‑SALAD  | SALAD       | 256 (binary)   | 85.4 %          | 45 MB | [download](https://example.com/tetra_salad.ckpt)  |
| TeTRA‑MixVPR | MixVPR      | 512 (binary)   | 82.5 %          | 46 MB | [download](https://example.com/tetra_mixvpr.ckpt) |

---

## Reproducing the Paper

To replicate Tables I & III from the manuscript:

```bash
bash scripts/reproduce_icra25.sh  # runs all benchmarks & logs metrics to CSV
python scripts/plot_tradeoffs.py   # recreates Fig. 3 & 4
```

> **Expected hardware:** 1× A100 80GB or 4× RTX 4090 (gradient accumulation=3).

---

## Citation

If you use this codebase or the pretrained models, please cite:

```bibtex
@article{Grainge2025TeTRA,
  title   = {TeTRA--VPR: A Ternary Transformer Approach for Compact Visual Place Recognition},
  author  = {Oliver Grainge and Michael Milford and Indu Bodala and Sarvapali~D.~Ramchurn and Shoaib Ehsan},
  journal = {IEEE Transactions on Robotics},
  year    = {2025},
  note    = {arXiv:2503.02511}
}
```

---

## License

This project is released under the **MIT License**. See [`LICENSE`](LICENSE) for details.

---

## Contact

* **Ollie Grainge** – [oeg1n18@soton.ac.uk](mailto:oeg1n18@soton.ac.uk)
* Pull requests are welcome! Open an issue for feature requests or bug reports.

> *“A place is worth a bag of ternary queries.”*  – TeTRA motto 🛰️🚀
