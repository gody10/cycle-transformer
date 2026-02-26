# CyTran Baseline — CTPA Image Translation

> **One-line summary**: Train a ConvTransformer-based CycleGAN (CyTran) to translate between
> ARTERIAL and NATIVE lung-CT contrast phases, evaluate on held-out 3-D volumes
> with PSNR / SSIM / MAE / RMSE / FID, and (optionally) compare against a latent-
> diffusion baseline — all using the **same** preprocessing contract as `ctpa_cycleGAN`.

---

## Quick Start

All commands assume you are **inside the `cycle-transformer/` directory** and that
the shared Coltea data lives at `../data/Coltea-Lung-CT-100W/`.

### Option A — Basic training (no validation)

```bash
python train.py \
  --dataroot ../data/Coltea-Lung-CT-100W \
  --model cytran \
  --name coltea_cytran_baseline \
  --batch_size 2 \
  --n_epochs 50 \
  --n_epochs_decay 40 \
  --no_html \
  --display_id -1
```

### Option B — Training with validation & early stopping

```bash
python train_with_validation_checkpoints.py \
  --dataroot ../data/Coltea-Lung-CT-100W \
  --model cytran \
  --name coltea_cytran_baseline \
  --batch_size 2 \
  --n_epochs 50 \
  --n_epochs_decay 40 \
  --use_validation \
  --patience 50 \
  --no_html \
  --display_id -1
```

### Option C — 2-D slice testing (per-patient metrics)

```bash
python test.py \
  --dataroot ../data/Coltea-Lung-CT-100W \
  --model cytran \
  --name coltea_cytran_baseline \
  --epoch best
```

Add `--subtraction_eval` to also compute subtraction metrics (Sub PSNR/SSIM/MAE/RMSE)
and generate 3-view subtraction visualizations. Add `--save_subtractions` to save
subtraction NIfTI volumes denormalized to HU.

### Option D — Honest 3-D stitching (for model comparison)

```bash
python inference_and_stitch.py \
  --dataroot ../data/Coltea-Lung-CT-100W \
  --model cytran \
  --name coltea_cytran_baseline \
  --epoch best
```

### Option E — Cross-model comparison (CyTran vs Diffusion)

```bash
python compare_models.py \
  --diffusion_dir ../ctpa_medvae_latent_diffusion/predictions_medvae_ldm/epoch_100 \
  --cyclegan_dir results/coltea_cytran_baseline/stitched_3d_epoch_best \
  --ground_truth_dir ../data/Coltea-Lung-CT-100W/ARTERIAL \
  --output_dir results/cytran_vs_diffusion_comparison
```

### Option F — Plot training curves

```bash
python plot_losses.py \
  --log_file checkpoints/coltea_cytran_baseline/train_log.txt \
  --output_dir checkpoints/coltea_cytran_baseline/loss_plots
```

### CSV overrides

The default CSVs are `../data/Coltea-Lung-CT-100W/{train,eval,test}_data.csv`.
Override with `--train_csv`, `--val_csv`, `--test_csv`, and `--csv_column` flags.

### Running on Kubernetes (EIDF)

```bash
kubectl apply -f job.yml          # edit the args: section first
kubectl logs -f job/cytran        # stream logs
kubectl delete job cytran          # clean up
```

---

## Table of Contents

1. [Project Overview](#project-overview)
2. [Data Parity with ctpa_cycleGAN](#data-parity-with-ctpa_cyclegan)
3. [Architecture](#architecture)
4. [Directory Structure](#directory-structure)
5. [Dataset Classes](#dataset-classes)
6. [Training](#training)
7. [Inference & 3-D Stitching](#inference--3-d-stitching)
8. [Model Comparison](#model-comparison)
9. [Kubernetes Deployment](#kubernetes-deployment)
10. [Key File Reference](#key-file-reference)
11. [Troubleshooting](#troubleshooting)
12. [Completed Training Runs](#completed-training-runs)

---

## 1. Project Overview

This repository wraps the **CyTran** (ConvTransformer CycleGAN) architecture from
[Ristea et al., 2023](https://arxiv.org/abs/2110.06400) in an identical benchmarking
harness to `ctpa_cycleGAN`. The goal is **apple-to-apple** metrics between CyTran
and other generative approaches (ResNet CycleGAN, SynDiff, latent diffusion) on the
Coltea Lung-CT 100W dataset.

### What is CyTran?

CyTran replaces the standard ResNet-9-blocks generator in CycleGAN with a
**ConvTransformer**: a hybrid architecture that uses convolutional downsampling/
upsampling with a Vision-Transformer bottleneck. This allows the generator to
capture both local texture (convolutions) and global structure (self-attention).

Key paper: *CyTran: A Cycle-Consistent Transformer with Multi-Level Consistency
for Non-Contrast to Contrast CT Translation* (Ristea et al., 2023).

---

## 2. Data Parity with ctpa_cycleGAN

Both repositories consume the **same** input data using the **same** preprocessing
contract, producing outputs in an **identical** value range and spatial layout:

| Property | Value |
|---|---|
| HU window | `[-1000, 1000]` (width 2000, level 0) |
| After windowing | `[0, 1]` via `ScaleIntensityRanged` |
| Model input range | `[-1, 1]` (Tanh generator output) |
| XY resize | 256 × 256 |
| Orientation | RAS |
| Minimum depth | 64 (zero-padded) |
| Divisible pad | k = 16 |
| Transforms library | MONAI |

### Data directories

The Coltea dataset should follow this structure:

```
../data/Coltea-Lung-CT-100W/
├── ARTERIAL/            # Domain B — contrast-enhanced NIfTI volumes
│   ├── patient_001.nii.gz
│   └── ...
├── NATIVE/              # Domain A — non-contrast NIfTI volumes
│   ├── patient_001.nii.gz
│   └── ...
├── train_data.csv       # Training split
├── eval_data.csv        # Validation split
└── test_data.csv        # Test split
```

Each CSV has a `patient_id` column listing stem names (without `.nii.gz`).

---

## 3. Architecture

### Generator — ConvTransformer

Unlike the ResNet-9-blocks generator used in `ctpa_cycleGAN`, CyTran uses a
**ConvTransformer** generator with the following structure:

```
Input (1 × 256 × 256)
  │
  ▼
Encoder (Conv downsampling)
  ├─ ReflectionPad2d(3) + Conv2d(1→ngf, k=7) + Norm + ReLU
  ├─ Conv2d(ngf→2×ngf, k=3, s=2) + Norm + ReLU         # downsample 1
  ├─ Conv2d(2×ngf→4×ngf, k=3, s=2) + Norm + ReLU       # downsample 2
  └─ Conv2d(4×ngf→8×ngf, k=3, s=2) + Norm + ReLU       # downsample 3
  │
  ▼
Transformer bottleneck (dim = 8×ngf = 128)
  ├─ depth × [PreNorm → DepthWiseConv Attention → Residual]
  └─ depth × [PreNorm → Conv FeedForward → Residual]
  │
  ▼
Decoder (ConvTranspose upsampling)
  ├─ ConvTranspose2d(8×ngf→4×ngf, k=3, s=2) + Norm + ReLU
  ├─ ConvTranspose2d(4×ngf→2×ngf, k=3, s=2) + Norm + ReLU
  ├─ ConvTranspose2d(2×ngf→ngf, k=3, s=2) + Norm + ReLU
  └─ ReflectionPad2d(3) + Conv2d(ngf→1, k=7)
  │
  ▼
Output (1 × 256 × 256)   # Tanh not explicitly applied — raw output
```

Default hyperparameters:

| Parameter | Default | Description |
|---|---|---|
| `--ngf_cytran` | 16 | Base number of generator filters |
| `--n_downsampling` | 3 | Number of down/up-sampling stages |
| `--depth` | 3 | Number of transformer layers in bottleneck |
| `--heads` | 6 | Number of attention heads |
| `--dropout` | 0.05 | Dropout rate in transformer |

Bottleneck feature dimension: `2^n_downsampling × ngf_cytran = 2^3 × 16 = 128`.

### Discriminator — PatchGAN

Same as standard CycleGAN: a 70×70 PatchGAN (`--netD basic`, `--n_layers_D 3`).

### Loss functions

| Loss | Weight | Formula |
|---|---|---|
| GAN (LSGAN) | `--lambda_A`, `--lambda_B` (default 10.0) | MSE between D output and target label |
| Cycle consistency | `--lambda_A`, `--lambda_B` (default 10.0) | L1(x, G_B(G_A(x))) + L1(y, G_A(G_B(y))) |
| Identity | `--lambda_identity` × lambda (default 0.5) | L1(y, G_A(y)) + L1(x, G_B(x)) |

### Optimizer

| Parameter | Default |
|---|---|
| Optimizer | Adam |
| Learning rate | 0.0001 |
| Beta1 | 0.5 |
| LR policy | Linear decay |
| `--n_epochs` | 50 (constant LR) |
| `--n_epochs_decay` | 40 (linear decay to 0) |

---

## 4. Directory Structure

```
cycle-transformer/
├── CYTRAN_BASELINE.md               # ← this document
├── README.md                        # Original CyTran paper README
├── Dockerfile                       # Docker build for Kubernetes
├── job.yml                          # Kubernetes job template (EIDF H100)
├── requirements.txt                 # Python dependencies
│
├── dataset.py                       # MONAI-based data loading (ported from ctpa_cycleGAN)
├── train.py                         # Basic training loop
├── train_with_validation_checkpoints.py  # Training with val tracking + early stopping
├── test.py                          # Full 3-D volume test with per-patient metrics
├── inference_and_stitch.py          # Honest 3-D stitching for model comparison
├── compare_models.py                # Cross-model comparison (CyTran vs Diffusion)
├── plot_losses.py                   # Loss curve visualization
│
├── models/
│   ├── __init__.py                  # Model registry
│   ├── base_model.py                # Abstract model base class
│   ├── cytran_model.py              # CyTran model (CycleGAN with ConvTransformer generators)
│   ├── conv_transformer.py          # ConvTransformer architecture (Encoder → Transformer → Decoder)
│   ├── cycle_gan_model.py           # Standard CycleGAN model (not used for CyTran)
│   └── networks.py                  # Discriminator definitions, init weights, schedulers
│
├── options/
│   ├── base_options.py              # Shared CLI options (data paths, model params, CSV splits)
│   ├── train_options.py             # Training-specific options (LR, epochs, CyTran params)
│   └── test_options.py              # Test-specific options (results_dir, epoch)
│
├── util/
│   ├── __init__.py
│   ├── util.py                      # Tensor ↔ image utilities, mkdir, etc.
│   ├── visualizer.py                # Loss logging, HTML visualization
│   └── html.py                      # HTML report generation
│
├── data/                            # Original CyTran dataset loaders (not used by new pipeline)
│   ├── __init__.py
│   ├── ct_dataset.py
│   └── ...
│
├── datasets/                        # External datasets / symlinks
│
├── checkpoints/                     # Training checkpoints (auto-created)
│   └── coltea_cytran_baseline/
│       ├── latest_net_G_A.pth
│       ├── latest_net_G_B.pth
│       ├── best_net_G_A.pth
│       ├── best_net_G_B.pth
│       ├── train_log.txt
│       └── opt.txt
│
├── results/                         # Test & inference outputs (auto-created)
│   └── coltea_cytran_baseline/
│       ├── test_best/               # 2-D test output
│       │   ├── metrics.csv
│       │   ├── test_results.json
│       │   ├── slices/
│       │   └── volumes/
│       └── stitched_3d_epoch_best/  # 3-D stitched output
│           ├── {pid}_cytran_pred.nii.gz
│           ├── {pid}_ground_truth.nii.gz
│           ├── {pid}_source.nii.gz
│           └── inference_manifest.csv
│
└── scripts/                         # Utility scripts
```

---

## 5. Dataset Classes

Both classes are in `dataset.py` and mirror the identically-named classes in
`ctpa_cycleGAN/dataset.py`.

### `ColteaSliceDataset` — Training & validation

- **Purpose**: Supplies 2-D axial slices for CycleGAN training.
- **Returns**: `{"A": tensor, "B": tensor, "A_paths": str, "B_paths": str}`
  where tensors are `[1, 256, 256]` in `[-1, 1]`.
- **Preprocessing**: MONAI transforms → [0,1] per-slice → linearly map to [-1,1].
- **Pairing**: Slices are paired by index across A/B domains; filtered to the
  intersection of valid patients from the CSV.

### `ColteaPairedDataset3D` — Testing & inference

- **Purpose**: Loads full 3-D NIfTI volumes for per-patient testing.
- **Returns**: `{"source": tensor, "target": tensor, "patient_id": str}`
  where tensors are `[1, D, H, W]` in `[0, 1]` (not shifted to [-1,1] — the caller
  handles normalization per-slice).
- **Preprocessing**: Same MONAI transforms, applied volume-level.

### Helper functions

| Function | Description |
|---|---|
| `_build_load_transforms()` | Constructs the MONAI transform chain |
| `_normalize_to_neg1_pos1(t)` | `t * 2 - 1` (maps [0,1]→[-1,1]) |
| `_denormalize_to_01(t)` | `(t + 1) / 2` (maps [-1,1]→[0,1]) |
| `_collect_valid_samples(csv, dataroot, ...)` | Reads CSV, filters to existing NIfTI pairs |

---

## 6. Training

### Standard training

```bash
python train.py \
  --dataroot ../data/Coltea-Lung-CT-100W \
  --model cytran \
  --name coltea_cytran_baseline \
  --batch_size 2 \
  --n_epochs 50 \
  --n_epochs_decay 40 \
  --no_html --display_id -1
```

### Training with validation

```bash
python train_with_validation_checkpoints.py \
  --dataroot ../data/Coltea-Lung-CT-100W \
  --model cytran \
  --name coltea_cytran_baseline \
  --batch_size 2 \
  --n_epochs 50 \
  --n_epochs_decay 40 \
  --use_validation --patience 50 \
  --no_html --display_id -1
```

When `--use_validation` is set, the script:
1. Runs a validation pass every epoch on the `--val_csv` split.
2. Computes validation L1 loss and PSNR.
3. Saves `best_net_G_A.pth` / `best_net_G_B.pth` when PSNR improves.
4. Stops early if PSNR has not improved for `--patience` epochs.

### Key training options

| Option | Default | Description |
|---|---|---|
| `--model` | `cytran` | Model class to use |
| `--name` | `cytran` | Experiment name (checkpoint subdirectory) |
| `--batch_size` | 2 | Batch size |
| `--n_epochs` | 50 | Epochs at constant learning rate |
| `--n_epochs_decay` | 40 | Epochs with linear LR decay |
| `--lr` | 0.0001 | Initial learning rate |
| `--beta1` | 0.5 | Adam momentum |
| `--gan_mode` | `lsgan` | GAN loss type |
| `--ngf_cytran` | 16 | Generator base filter count |
| `--n_downsampling` | 3 | Conv encoder downsampling stages |
| `--depth` | 3 | Transformer bottleneck layers |
| `--heads` | 6 | Attention heads |
| `--dropout` | 0.05 | Transformer dropout |
| `--use_validation` | False | Enable validation loop |
| `--patience` | 50 | Early stopping patience |
| `--save_epoch_freq` | 2 | Save checkpoint every N epochs |
| `--no_html` | False | Disable HTML report generation |
| `--display_id` | 1 | Visdom display ID (-1 to disable) |

### Checkpoints

Saved to `checkpoints/<name>/`:

| File | Description |
|---|---|
| `latest_net_G_A.pth` | Latest G_A weights |
| `latest_net_G_B.pth` | Latest G_B weights |
| `latest_net_D_A.pth` | Latest D_A weights |
| `latest_net_D_B.pth` | Latest D_B weights |
| `best_net_G_A.pth` | Best G_A (by val PSNR, if `--use_validation`) |
| `best_net_G_B.pth` | Best G_B (by val PSNR, if `--use_validation`) |
| `<N>_net_G_A.pth` | Epoch N snapshot |
| `train_log.txt` | Per-iteration training log |
| `opt.txt` | Saved options |

---

## 7. Inference & 3-D Stitching

### Option C — `test.py` (per-patient 3-D metrics)

```bash
python test.py \
  --dataroot ../data/Coltea-Lung-CT-100W \
  --model cytran \
  --name coltea_cytran_baseline \
  --epoch best
```

**With subtraction evaluation:**

```bash
python test.py \
  --dataroot ../data/Coltea-Lung-CT-100W \
  --model cytran \
  --name coltea_cytran_baseline \
  --epoch best \
  --subtraction_eval \
  --save_subtractions
```

**What it does:**
1. Loads each test-set patient as a full 3-D volume via `ColteaPairedDataset3D`.
2. Runs G_A slice-by-slice (domain A→B translation).
3. Computes per-patient PSNR, SSIM, MAE, RMSE.
4. Saves 4-panel comparison PNGs (source / prediction / ground truth / difference).
5. Outputs `metrics.csv`, `test_results.json`, and NIfTI volumes.
6. (If `--subtraction_eval`) Computes subtraction metrics: Sub PSNR, Sub SSIM, Sub MAE,
   Sub RMSE by comparing `(source − generated)` vs `(source − ground_truth)`.
   Generates 3-view (sagittal, coronal, axial) subtraction comparison visualizations.
7. (If `--save_subtractions`) Saves subtraction NIfTI volumes denormalized to HU.

| Flag | Description |
|------|-------------|
| `--subtraction_eval` | Enable subtraction evaluation and visualization |
| `--save_subtractions` | Save subtraction NIfTI volumes (source − generated, source − ground_truth) in HU |

**Output structure:**

```
results/<name>/test_<epoch>/
├── metrics.csv              # Per-patient metrics table (includes sub_* if --subtraction_eval)
├── test_results.json        # Summary statistics (mean ± std)
├── visualizations/          # Comparison PNGs per patient
│   ├── <pid>_slices.png
│   ├── <pid>_3view_{sagittal,coronal,axial}.png
│   ├── <pid>_subtractions_{sagittal,coronal,axial}.png  # if --subtraction_eval
│   └── metrics_summary.png
├── volumes/
│   ├── <pid>_pred.nii.gz
│   └── <pid>_ground_truth.nii.gz
└── subtractions/            # if --save_subtractions
    ├── <pid>_subtraction_synthesized.nii.gz
    └── <pid>_subtraction_gt.nii.gz
```

### Option D — `inference_and_stitch.py` (honest 3-D stitching)

```bash
python inference_and_stitch.py \
  --dataroot ../data/Coltea-Lung-CT-100W \
  --model cytran \
  --name coltea_cytran_baseline \
  --epoch best
```

**Key design principle:** No post-processing. Raw model output in `[0, 1]` is saved
directly as NIfTI, preserving full HU-window information for fair comparison with
other models.

**Output structure:**

```
results/<name>/stitched_3d_epoch_<epoch>/
├── <pid>_cytran_pred.nii.gz       # Model prediction
├── <pid>_ground_truth.nii.gz      # Target ARTERIAL volume
├── <pid>_source.nii.gz            # Source NATIVE volume
└── inference_manifest.csv          # Per-patient manifest
```

**Note:** The output suffix is `_cytran_pred` (vs `_cyclegan_pred` in ctpa_cycleGAN)
to avoid filename collisions when comparing models.

### Metrics

All metrics are computed on `[0, 1]` range volumes:

| Metric | Implementation | Notes |
|---|---|---|
| PSNR | `10 * log10(1.0 / MSE)` | Peak = 1.0 since data is in [0,1] |
| SSIM | `scipy.ndimage.uniform_filter` based | 3-D structural similarity with 11×11×11 window |
| MAE | `mean(abs(pred - gt))` | Mean absolute error |
| RMSE | `sqrt(mean((pred - gt)^2))` | Root mean squared error |
| FID | Inception-v3 features | Used only in `compare_models.py` |

**Subtraction metrics** (with `--subtraction_eval`): compares `(source − generated)` vs
`(source − ground_truth)`, computing Sub PSNR, Sub SSIM, Sub MAE, Sub RMSE. These
measure how well the model preserves the contrast difference between source and target.

---

## 8. Model Comparison

```bash
python compare_models.py \
  --diffusion_dir ../ctpa_medvae_latent_diffusion/predictions_medvae_ldm/epoch_100 \
  --cyclegan_dir results/coltea_cytran_baseline/stitched_3d_epoch_best \
  --ground_truth_dir ../data/Coltea-Lung-CT-100W/ARTERIAL \
  --output_dir results/cytran_vs_diffusion_comparison \
  --cyclegan_suffix _cytran_pred
```

This script:
1. Finds matching patients across both model output directories and ground truth.
2. Computes PSNR, SSIM, and FID for each model.
3. Generates comparison tables and box plots.
4. Saves results to `comparison_results.json`.

**Note:** The `--cyclegan_suffix` defaults to `_cytran_pred` (matching the suffix
in `inference_and_stitch.py`). When comparing against ctpa_cycleGAN outputs, use
`--cyclegan_suffix _cyclegan_pred`.

---

## 9. Kubernetes Deployment

### Docker image

```
gody10/cycle_trans:latest
```

Built from the `Dockerfile` in this directory. Contains all dependencies
(PyTorch, MONAI, nibabel, scipy, etc.).

### Job template (`job.yml`)

The job specification is pre-configured for the EIDF cluster with:

| Resource | Value |
|---|---|
| GPU | 1 × NVIDIA H100 80GB |
| CPU | 10 cores |
| Memory | 64 Gi |
| Storage | PVC `eidf212-cephfs` at `/cephfs/eidf212/shared/odiamant/` |
| runAsUser | 28574 |

### Editing the job

Open `job.yml` and update the `args:` section with the desired command. Examples:

**Training:**
```yaml
args:
  - >-
    cd /cephfs/eidf212/shared/odiamant/cycle-transformer &&
    python train_with_validation_checkpoints.py
    --dataroot ../data/Coltea-Lung-CT-100W
    --model cytran
    --name coltea_cytran_baseline
    --batch_size 2
    --n_epochs 50
    --n_epochs_decay 40
    --use_validation
    --patience 50
    --no_html
    --display_id -1
```

**Testing:**
```yaml
args:
  - >-
    cd /cephfs/eidf212/shared/odiamant/cycle-transformer &&
    python test.py
    --dataroot ../data/Coltea-Lung-CT-100W
    --model cytran
    --name coltea_cytran_baseline
    --epoch best
    --subtraction_eval
    --save_subtractions
```

**Inference:**
```yaml
args:
  - >-
    cd /cephfs/eidf212/shared/odiamant/cycle-transformer &&
    python inference_and_stitch.py
    --dataroot ../data/Coltea-Lung-CT-100W
    --model cytran
    --name coltea_cytran_baseline
    --epoch best
```

### Deploy

```bash
kubectl apply -f job.yml
kubectl logs -f job/cytran
kubectl delete job cytran
```

---

## 10. Key File Reference

| File | Purpose | Ported from ctpa_cycleGAN? |
|---|---|---|
| `dataset.py` | MONAI-based data loading (2-D slices + 3-D volumes) | Yes — direct port |
| `train.py` | Basic training loop | No — original CyTran |
| `train_with_validation_checkpoints.py` | Training with val tracking + early stopping | Yes — adapted for CyTran Visualizer API |
| `test.py` | Full 3-D volume testing with metrics | Yes — direct port |
| `inference_and_stitch.py` | Honest 3-D stitching (no post-processing) | Yes — uses `_cytran_pred` suffix |
| `compare_models.py` | Cross-model comparison (PSNR, SSIM, FID) | Yes — labels adapted to "CyTran" |
| `plot_losses.py` | Training loss curve visualization | Yes — direct port |
| `job.yml` | Kubernetes job template | Yes — adapted image + naming |
| `models/cytran_model.py` | CyTran model definition | No — original (untouched) |
| `models/conv_transformer.py` | ConvTransformer architecture | No — original (untouched) |
| `models/networks.py` | Discriminator + utility definitions | No — original (untouched) |
| `options/base_options.py` | Shared CLI options | Modified — added CSV split args |
| `options/train_options.py` | Training CLI options | Modified — added validation args |
| `options/test_options.py` | Test CLI options | Modified — uncommented results_dir |

---

## 11. Troubleshooting

### "No matching patients found" during test or inference

- Check that `--test_csv` points to a valid CSV with a `patient_id` column.
- Verify that the patient stems in the CSV match filenames in
  `<dataroot>/ARTERIAL/` and `<dataroot>/NATIVE/` (without `.nii.gz`).
- Ensure `--Aclass` and `--Bclass` match the subdirectory names (default:
  `NATIVE` and `ARTERIAL`).

### "RuntimeError: Error(s) in loading state_dict"

- Make sure `--model cytran` is set (not `cycle_gan`). The checkpoint keys differ.
- Check that `--ngf_cytran`, `--n_downsampling`, `--depth`, `--heads` match the
  values used during training (defaults: 16, 3, 3, 6).
- Verify you're loading the correct epoch: `--epoch best` vs `--epoch latest` vs
  `--epoch 50`.

### GPU out of memory

- Reduce `--batch_size` (default 2).
- Consider reducing `--ngf_cytran` or `--depth`.
- Inference processes slices one-at-a-time; OOM during inference is unlikely.

### Visdom errors when running headless (Kubernetes)

- Always pass `--display_id -1 --no_html` when running without a display.

### Metrics look off compared to ctpa_cycleGAN

- Confirm both repos use the same test CSV and data directory.
- Verify the windowing: both should use `[-1000, 1000]` HU clamping.
- Check that volumes are in the same orientation (RAS) and spatial size (256×256).

### Comparison script cannot find CyTran volumes

- The default suffix is `_cytran_pred`. Make sure `inference_and_stitch.py` was
  run first and that `--cyclegan_suffix` matches the output naming convention.

---

## 12. Completed Training Runs

| Run name | Epochs | Val PSNR (best) | Notes |
|---|---|---|---|
| *(pending)* | — | — | First run not yet completed |

Update this table after each completed training run for reproducibility tracking.

---

## Citation

If you use this code, please cite the original CyTran paper:

```bibtex
@InProceedings{Ristea-CyTran-2023,
  title     = {CyTran: A Cycle-Consistent Transformer with Multi-Level
               Consistency for Non-Contrast to Contrast CT Translation},
  author    = {Ristea, Nicolae-Catalin and Miron, Andreea-Iuliana and
               Savencu, Olivian and Georgescu, Mariana-Iuliana and
               Verga, Nicolae and Khan, Fahad Shahbaz and Ionescu, Radu Tudor},
  journal   = {Neurocomputing},
  year      = {2023}
}
```
