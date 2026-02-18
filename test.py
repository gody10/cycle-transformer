"""Test script for CyTran baseline with data-parity normalization.

Processes full 3D volumes slice-by-slice and computes metrics.
Uses the same fixed HU windowing [-1000, 1000] as the diffusion model.

Example:
    python test.py --dataroot ../data/Coltea_Processed_Nifti_Registered \
        --name coltea_cytran_baseline --model cytran \
        --input_nc 1 --output_nc 1 --epoch best
"""

import os
import json
import torch
import numpy as np
import nibabel as nib
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from tqdm import tqdm
from torch.utils.data import DataLoader
from scipy.ndimage import uniform_filter

from options.test_options import TestOptions
from models import create_model

from dataset import ColteaPairedDataset3D, _normalize_to_neg1_pos1, _denormalize_to_01


# Defaults are now defined in options/base_options.py (--test_csv, --csv_column)


def psnr(target, pred, data_range=1.0):
    """PSNR on [0, 1] volumes."""
    mse = np.mean((target - pred) ** 2)
    if mse < 1e-10:
        return float("inf")
    return 10.0 * np.log10(data_range ** 2 / mse)


def ssim_3d(target, pred, data_range=1.0, win_size=7):
    """Mean SSIM across axial slices."""
    C1 = (0.01 * data_range) ** 2
    C2 = (0.03 * data_range) ** 2
    scores = []
    for d in range(target.shape[-1]):
        p = pred[:, :, d].astype(np.float64)
        t = target[:, :, d].astype(np.float64)
        mu_p = uniform_filter(p, size=win_size)
        mu_t = uniform_filter(t, size=win_size)
        sig_p = uniform_filter(p ** 2, size=win_size) - mu_p ** 2
        sig_t = uniform_filter(t ** 2, size=win_size) - mu_t ** 2
        sig_pt = uniform_filter(p * t, size=win_size) - mu_p * mu_t
        num = (2 * mu_p * mu_t + C1) * (2 * sig_pt + C2)
        den = (mu_p ** 2 + mu_t ** 2 + C1) * (sig_p + sig_t + C2)
        scores.append(float(np.mean(num / den)))
    return float(np.mean(scores))


def ssim_2d(target, pred, data_range=1.0, win_size=7):
    """SSIM for a single 2D slice."""
    C1 = (0.01 * data_range) ** 2
    C2 = (0.03 * data_range) ** 2
    p = pred.astype(np.float64)
    t = target.astype(np.float64)
    mu_p = uniform_filter(p, size=win_size)
    mu_t = uniform_filter(t, size=win_size)
    sig_p = uniform_filter(p ** 2, size=win_size) - mu_p ** 2
    sig_t = uniform_filter(t ** 2, size=win_size) - mu_t ** 2
    sig_pt = uniform_filter(p * t, size=win_size) - mu_p * mu_t
    num = (2 * mu_p * mu_t + C1) * (2 * sig_pt + C2)
    den = (mu_p ** 2 + mu_t ** 2 + C1) * (sig_p + sig_t + C2)
    return float(np.mean(num / den))


def mae(target, pred):
    """Mean Absolute Error on [0, 1] volumes."""
    return float(np.mean(np.abs(target - pred)))


def rmse(target, pred):
    """Root Mean Squared Error on [0, 1] volumes."""
    return float(np.sqrt(np.mean((target - pred) ** 2)))


def tensor_to_numpy(tensor):
    """Convert tensor from [-1,1] to numpy array in [0,1] range."""
    img = tensor.squeeze().cpu().numpy()
    img = (img + 1) / 2.0  # [-1,1] -> [0,1]
    img = np.clip(img, 0, 1)
    return img


def save_comparison_grid(art_slice, pred_slice, gt_slice, patient_id, slice_idx,
                         output_dir, vol_psnr=None, vol_ssim=None):
    """
    Saves a 4-panel PNG: Input | Predicted (with MAE/SSIM) | Ground Truth | Absolute Error.

    Args:
        art_slice: 2D numpy array [H, W] (Input/Arterial) in [0, 1]
        pred_slice: 2D numpy array [H, W] (Generated Native) in [0, 1]
        gt_slice: 2D numpy array [H, W] (Ground Truth Native) in [0, 1]
        patient_id: str
        slice_idx: int, the z-index of this slice
        output_dir: str
        vol_psnr: float, volume-level PSNR for suptitle
        vol_ssim: float, volume-level SSIM for suptitle
    """
    # Compute per-slice metrics
    slice_mae = mae(gt_slice, pred_slice)
    slice_ssim = ssim_2d(gt_slice, pred_slice)
    slice_psnr = psnr(gt_slice, pred_slice)
    abs_error = np.abs(gt_slice - pred_slice)

    fig, axes = plt.subplots(1, 4, figsize=(20, 5))

    # Common display settings
    kwargs = {'cmap': 'gray', 'vmin': 0, 'vmax': 1}

    # 1. Source (Contrast)
    axes[0].imshow(art_slice, **kwargs)
    axes[0].set_title("Source (Contrast)")
    axes[0].axis("off")

    # 2. Predicted with per-slice metrics
    axes[1].imshow(pred_slice, **kwargs)
    axes[1].set_title(f"Predicted\nMAE={slice_mae:.4f}  SSIM={slice_ssim:.4f}")
    axes[1].axis("off")

    # 3. Ground Truth
    axes[2].imshow(gt_slice, **kwargs)
    axes[2].set_title("Ground Truth")
    axes[2].axis("off")

    # 4. Absolute Error heatmap
    im = axes[3].imshow(abs_error, cmap='inferno', vmin=0, vmax=0.3)
    axes[3].set_title("Absolute Error")
    axes[3].axis("off")
    fig.colorbar(im, ax=axes[3], fraction=0.046, pad=0.04, label="Absolute Error")

    # Suptitle with patient info and volume-level metrics
    suptitle = f"Patient: {patient_id} — Axial Slice {slice_idx}"
    if vol_psnr is not None and vol_ssim is not None:
        suptitle += f"\nVol PSNR: {vol_psnr:.2f} dB | SSIM: {vol_ssim:.4f}"
    plt.suptitle(suptitle, fontsize=14)
    plt.tight_layout()

    # Create 'slices' subdirectory to keep things organized
    slices_dir = os.path.join(output_dir, "slices")
    os.makedirs(slices_dir, exist_ok=True)

    save_path = os.path.join(slices_dir, f"{patient_id}_slice_{slice_idx:03d}.png")
    plt.savefig(save_path, dpi=150)
    plt.close(fig)


def test_full_volume(model, source_vol, target_vol, opt, patient_id, output_dir):
    """
    Process entire 3D volume slice by slice, compute metrics, and save visualizations.

    Visualization grids are saved after the full volume is assembled so that
    volume-level metrics can be displayed in the suptitle.

    Args:
        model: CyTran model in eval mode.
        source_vol: (1, H, W, D) tensor in [0, 1].
        target_vol: (1, H, W, D) tensor in [0, 1].
        opt: Options object.
        patient_id: str.
        output_dir: str.

    Returns:
        generated_volume: (H, W, D) in [0, 1].
        gt_volume: (H, W, D) in [0, 1].
        input_volume: (H, W, D) in [0, 1].
        metrics_dict: dict with psnr, ssim, mae, rmse.
    """
    C, H, W, D = source_vol.shape
    generated_slices = []

    save_indices = {int(D * 0.25), int(D * 0.50), int(D * 0.75)}
    # Store slices for deferred visualization
    saved_slices = {}

    for d in range(D):
        src_slice = source_vol[:, :, :, d].unsqueeze(0)  # (1, 1, H, W)
        tgt_slice = target_vol[:, :, :, d].unsqueeze(0)

        # [0, 1] -> [-1, 1]
        src_slice = _normalize_to_neg1_pos1(src_slice)
        tgt_slice = _normalize_to_neg1_pos1(tgt_slice)

        slice_data = {
            "A": src_slice,
            "B": tgt_slice,
            "A_paths": "arterial",
            "B_paths": "native",
        }

        model.set_input(slice_data)
        model.test()

        visuals = model.get_current_visuals()
        gen_slice = visuals.get("fake_B", visuals.get("fake"))

        gen_np = tensor_to_numpy(gen_slice)  # (H, W) in [0, 1]
        generated_slices.append(gen_np)

        if d in save_indices:
            input_np = tensor_to_numpy(slice_data["A"])
            gt_np = tensor_to_numpy(slice_data["B"])
            saved_slices[d] = (input_np, gen_np, gt_np)

    generated_volume = np.stack(generated_slices, axis=-1)  # (H, W, D)
    gt_volume = target_vol.squeeze(0).cpu().numpy()  # (H, W, D) in [0, 1]
    input_volume = source_vol.squeeze(0).cpu().numpy()

    # Compute volume-level metrics
    vol_psnr = psnr(gt_volume, generated_volume, data_range=1.0)
    vol_ssim = ssim_3d(gt_volume, generated_volume, data_range=1.0)
    vol_mae = mae(gt_volume, generated_volume)
    vol_rmse = rmse(gt_volume, generated_volume)

    metrics_dict = {
        "psnr": vol_psnr,
        "ssim": vol_ssim,
        "mae": vol_mae,
        "rmse": vol_rmse,
    }

    # Now save visualization grids with volume-level metrics in suptitle
    for d, (input_np, gen_np, gt_np) in saved_slices.items():
        save_comparison_grid(
            input_np, gen_np, gt_np, patient_id, d, output_dir,
            vol_psnr=vol_psnr, vol_ssim=vol_ssim,
        )

    return generated_volume, gt_volume, input_volume, metrics_dict


if __name__ == "__main__":
    opt = TestOptions().parse()
    opt.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    # Hard-code some parameters for test
    opt.num_threads = 0
    opt.batch_size = 1
    opt.serial_batches = True
    opt.no_flip = True

    # Get config from options or use defaults
    test_csv = opt.test_csv
    test_col = opt.csv_column
    data_root = opt.dataroot

    # Create output directory
    output_dir = os.path.join(opt.results_dir, opt.name, f"test_{opt.epoch}")
    os.makedirs(output_dir, exist_ok=True)

    print(f"=" * 60)
    print(f"CyTran Testing - {opt.name}")
    print(f"=" * 60)
    print(f"Loading model from epoch: {opt.epoch}")
    print(f"Output directory: {output_dir}")
    print(f"Device: {opt.device}")

    # Create 3D volume dataset with diffusion-matched normalization
    test_ds = ColteaPairedDataset3D(test_csv, test_col, data_root)
    test_loader = DataLoader(
        test_ds,
        batch_size=1,
        shuffle=False,
        num_workers=0,
        pin_memory=True
    )

    print(f"Number of test patients: {len(test_ds)}")

    # Create and load model
    model = create_model(opt)
    model.setup(opt)

    if opt.eval:
        model.eval()

    # Results storage
    results = []

    print(f"\nStarting inference on {len(test_ds)} patients...")
    print("-" * 40)

    with torch.no_grad():
        for i, batch in enumerate(tqdm(test_loader, desc="Testing")):
            patient_id = batch["patient_id"][0]
            source_vol = batch["source"].squeeze(0)  # (1, H, W, D) in [0, 1]
            target_vol = batch["target"].squeeze(0)

            try:
                # Process full 3D volume slice-by-slice
                generated_vol, gt_vol, input_vol, metrics_dict = test_full_volume(
                    model, source_vol, target_vol, opt, patient_id, output_dir
                )

                results.append({
                    "patient_id": patient_id,
                    **metrics_dict,
                })

                print(f"  {patient_id}: PSNR={metrics_dict['psnr']:.2f} dB | "
                      f"SSIM={metrics_dict['ssim']:.4f} | "
                      f"MAE={metrics_dict['mae']:.4f} | "
                      f"RMSE={metrics_dict['rmse']:.4f}")

                # Save NIfTI volumes
                volumes_dir = os.path.join(output_dir, "volumes")
                os.makedirs(volumes_dir, exist_ok=True)
                nib.save(
                    nib.Nifti1Image(generated_vol.astype(np.float32), affine=np.eye(4)),
                    os.path.join(volumes_dir, f"{patient_id}_pred.nii.gz")
                )
                nib.save(
                    nib.Nifti1Image(gt_vol.astype(np.float32), affine=np.eye(4)),
                    os.path.join(volumes_dir, f"{patient_id}_ground_truth.nii.gz")
                )

            except Exception as e:
                print(f"\nError processing {patient_id}: {e}")
                results.append({
                    "patient_id": patient_id,
                    "psnr": float('nan'),
                    "ssim": float('nan'),
                    "mae": float('nan'),
                    "rmse": float('nan'),
                })
                continue

    # Save metrics to CSV
    df = pd.DataFrame(results)
    df.to_csv(os.path.join(output_dir, "metrics.csv"), index=False)

    # Build and save JSON results (matching diffusion model format)
    metric_names = ["psnr", "ssim", "mae", "rmse"]
    aggregate_metrics = {}
    for m in metric_names:
        vals = df[m].dropna()
        aggregate_metrics[f"{m}_mean"] = float(vals.mean())
        aggregate_metrics[f"{m}_std"] = float(vals.std())
        aggregate_metrics[f"{m}_min"] = float(vals.min())
        aggregate_metrics[f"{m}_max"] = float(vals.max())

    json_results = {
        "checkpoint": f"{opt.name}/epoch_{opt.epoch}",
        "test_csv": test_csv,
        "num_samples": len(test_ds),
        "aggregate_metrics": aggregate_metrics,
        "per_patient_results": [
            {k: float(v) if isinstance(v, (np.floating, float)) else v for k, v in r.items()}
            for r in results
        ],
    }

    json_path = os.path.join(output_dir, "test_results.json")
    with open(json_path, "w") as f:
        json.dump(json_results, f, indent=2)

    # Print summary
    print("\n" + "=" * 60)
    print(f"TEST RESULTS ({len(test_ds)} Patients)")
    print("=" * 60)
    print(f"Average PSNR: {aggregate_metrics['psnr_mean']:.4f} +/- {aggregate_metrics['psnr_std']:.4f}")
    print(f"Average SSIM: {aggregate_metrics['ssim_mean']:.4f} +/- {aggregate_metrics['ssim_std']:.4f}")
    print(f"Average MAE:  {aggregate_metrics['mae_mean']:.4f} +/- {aggregate_metrics['mae_std']:.4f}")
    print(f"Average RMSE: {aggregate_metrics['rmse_mean']:.4f} +/- {aggregate_metrics['rmse_std']:.4f}")
    print(f"\nResults saved to: {output_dir}")
    print(f"  > Metrics CSV:  metrics.csv")
    print(f"  > Metrics JSON: test_results.json")
    print(f"  > Images:       slices/ directory")
    print(f"  > Volumes:      volumes/ directory")
    print("=" * 60)
