#!/usr/bin/env python3
"""
Common Evaluation Script — Fair comparison of 3D Latent Diffusion vs CyTran.

Takes two folders of 3D NIfTI volumes (one from each model) and a folder of
ground truth volumes, then computes PSNR, SSIM, and FID on a common scale.

Scale Contract:
    All volumes are expected in [0, 1] range (un-normalised from each model's
    native output).  If the diffusion model outputs [-1, 1], this script
    converts it before computing metrics.

Metrics:
    - PSNR (Peak Signal-to-Noise Ratio) — computed per-volume, reported as mean +/- std
    - SSIM (Structural Similarity)      — computed per-volume, reported as mean +/- std
    - FID  (Frechet Inception Distance) — computed across all slices from all volumes

Usage:
    python compare_models.py \
        --diffusion_dir  ./predictions_medvae_ldm/ \
        --cyclegan_dir   ./results/cytran_3d_volumes/ \
        --gt_dir         ./ground_truth_volumes/ \
        --output_dir     ./comparison_results/

    If ground truth volumes are embedded inside the prediction folders
    (with _ground_truth.nii.gz suffix from inference_and_stitch.py):
        python compare_models.py \
            --diffusion_dir ./predictions_medvae_ldm/ \
            --cyclegan_dir  ./results/cytran_3d_volumes/ \
            --gt_from_cyclegan \
            --output_dir ./comparison_results/
"""

import os
import sys
import argparse
import json
import logging
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import nibabel as nib
from tqdm import tqdm
from scipy.ndimage import uniform_filter

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)


# ============================================================================
# Metric implementations (no external dependency on skimage or torchmetrics)
# ============================================================================


def compute_psnr(pred: np.ndarray, target: np.ndarray, data_range: float = 1.0) -> float:
    """Peak Signal-to-Noise Ratio on 3D volumes in [0, 1]."""
    mse = np.mean((pred - target) ** 2)
    if mse < 1e-10:
        return float("inf")
    return 10.0 * np.log10(data_range ** 2 / mse)


def compute_ssim_3d(
    pred: np.ndarray,
    target: np.ndarray,
    data_range: float = 1.0,
    win_size: int = 7,
) -> float:
    """
    Structural Similarity Index on 3D volumes, computed slice-by-slice
    and averaged (consistent with standard medical imaging practice).

    Uses uniform-filter local statistics (no external dependency).
    """
    C1 = (0.01 * data_range) ** 2
    C2 = (0.03 * data_range) ** 2

    ssim_scores = []
    depth = pred.shape[-1] if pred.ndim == 3 else pred.shape[0]

    for d in range(depth):
        if pred.ndim == 3:
            p = pred[:, :, d].astype(np.float64)
            t = target[:, :, d].astype(np.float64)
        else:
            p = pred[d].astype(np.float64)
            t = target[d].astype(np.float64)

        mu_p = uniform_filter(p, size=win_size)
        mu_t = uniform_filter(t, size=win_size)

        sigma_p_sq = uniform_filter(p ** 2, size=win_size) - mu_p ** 2
        sigma_t_sq = uniform_filter(t ** 2, size=win_size) - mu_t ** 2
        sigma_pt = uniform_filter(p * t, size=win_size) - mu_p * mu_t

        num = (2 * mu_p * mu_t + C1) * (2 * sigma_pt + C2)
        den = (mu_p ** 2 + mu_t ** 2 + C1) * (sigma_p_sq + sigma_t_sq + C2)

        ssim_scores.append(float(np.mean(num / den)))

    return float(np.mean(ssim_scores))


# ============================================================================
# FID — Frechet Inception Distance on 2D slices
# ============================================================================

def _get_inception_features(slices: np.ndarray, batch_size: int = 32) -> np.ndarray:
    """
    Extract InceptionV3 features from a collection of 2D slices.

    Args:
        slices: (N, H, W) array in [0, 1] (grayscale).
        batch_size: Inference batch size.

    Returns:
        (N, 2048) feature matrix.
    """
    try:
        import torch
        import torch.nn as nn
        from torchvision.models import inception_v3, Inception_V3_Weights
    except ImportError:
        logger.warning("torch/torchvision not available — skipping FID.")
        return None

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Load Inception with pretrained weights, remove classification head
    model = inception_v3(weights=Inception_V3_Weights.DEFAULT)
    # Replace final FC with identity to get pool5 features
    model.fc = nn.Identity()
    model.eval()
    model.to(device)

    features_list = []
    n = len(slices)

    for start in range(0, n, batch_size):
        end = min(start + batch_size, n)
        batch = slices[start:end]  # (B, H, W) in [0, 1]

        # Inception expects (B, 3, 299, 299)
        import torch.nn.functional as F

        tensor = torch.from_numpy(batch).float().unsqueeze(1)  # (B, 1, H, W)
        tensor = tensor.repeat(1, 3, 1, 1)  # (B, 3, H, W)
        tensor = F.interpolate(tensor, size=(299, 299), mode="bilinear", align_corners=False)
        tensor = tensor.to(device)

        with torch.no_grad():
            feat = model(tensor)  # (B, 2048)

        features_list.append(feat.cpu().numpy())

    all_features = np.concatenate(features_list, axis=0)
    return all_features


def compute_fid(features_pred: np.ndarray, features_target: np.ndarray) -> float:
    """
    Compute FID from two feature matrices.

    FID = ||mu_1 - mu_2||^2 + Tr(C_1 + C_2 - 2*sqrt(C_1 * C_2))
    """
    from scipy.linalg import sqrtm

    mu1 = np.mean(features_pred, axis=0)
    mu2 = np.mean(features_target, axis=0)
    sigma1 = np.cov(features_pred, rowvar=False)
    sigma2 = np.cov(features_target, rowvar=False)

    diff = mu1 - mu2
    covmean_sq = sqrtm(sigma1 @ sigma2)

    # Numerical stability
    if np.iscomplexobj(covmean_sq):
        covmean_sq = covmean_sq.real

    fid = float(diff @ diff + np.trace(sigma1 + sigma2 - 2 * covmean_sq))
    return max(fid, 0.0)  # FID should be non-negative


def extract_all_slices(volumes: List[np.ndarray]) -> np.ndarray:
    """Flatten a list of 3D volumes (H, W, D) into a stack of 2D slices (N, H, W)."""
    all_slices = []
    for vol in volumes:
        if vol.ndim == 3:
            for d in range(vol.shape[-1]):
                all_slices.append(vol[:, :, d])
        elif vol.ndim == 4:
            vol = vol.squeeze(0)
            for d in range(vol.shape[-1]):
                all_slices.append(vol[:, :, d])
    return np.stack(all_slices, axis=0)


# ============================================================================
# Volume loading utilities
# ============================================================================

def load_volume(path: str) -> np.ndarray:
    """Load a NIfTI volume and return as numpy array."""
    img = nib.load(path)
    data = img.get_fdata().astype(np.float32)
    return data


def ensure_01_range(vol: np.ndarray) -> np.ndarray:
    """
    If volume appears to be in [-1, 1] range, convert to [0, 1].
    Otherwise, clip to [0, 1].
    """
    vmin, vmax = vol.min(), vol.max()
    if vmin < -0.5:
        # Likely in [-1, 1] range
        vol = (vol + 1.0) / 2.0
    return np.clip(vol, 0.0, 1.0)


def find_matching_volumes(
    pred_dir: str,
    gt_dir: Optional[str],
    pred_suffix: str = "",
    gt_suffix: str = "",
) -> List[Tuple[str, str, str]]:
    """
    Match predicted volumes to ground truth volumes by patient ID.

    Returns list of (patient_id, pred_path, gt_path) tuples.
    """
    matches = []
    pred_files = sorted(Path(pred_dir).glob("*.nii.gz"))

    for pf in pred_files:
        name = pf.stem.replace(".nii", "")  # remove .nii from .nii.gz
        # Try to extract patient ID
        if pred_suffix and name.endswith(pred_suffix):
            pid = name[: -len(pred_suffix)]
        else:
            pid = name

        if gt_dir:
            # Look for matching GT file
            gt_candidates = [
                Path(gt_dir) / f"{pid}{gt_suffix}.nii.gz",
                Path(gt_dir) / f"{pid}_ground_truth.nii.gz",
                Path(gt_dir) / f"{pid}.nii.gz",
                Path(gt_dir) / pid / "native.nii.gz",
            ]
            gt_path = None
            for gc in gt_candidates:
                if gc.exists():
                    gt_path = str(gc)
                    break
            if gt_path:
                matches.append((pid, str(pf), gt_path))
        else:
            matches.append((pid, str(pf), None))

    return matches


# ============================================================================
# Main evaluation
# ============================================================================

def evaluate_model(
    pred_dir: str,
    gt_dir: Optional[str],
    model_name: str,
    pred_suffix: str = "",
    gt_suffix: str = "",
) -> Tuple[pd.DataFrame, List[np.ndarray], List[np.ndarray]]:
    """
    Evaluate a single model's predictions against ground truth.

    Returns:
        per_patient_df: DataFrame with per-patient PSNR and SSIM
        pred_volumes: list of prediction volumes in [0, 1]
        gt_volumes: list of ground truth volumes in [0, 1]
    """
    matches = find_matching_volumes(pred_dir, gt_dir, pred_suffix, gt_suffix)
    if not matches:
        logger.error(f"No matching volumes found for {model_name}")
        logger.error(f"  pred_dir: {pred_dir}")
        logger.error(f"  gt_dir:   {gt_dir}")
        return pd.DataFrame(), [], []

    logger.info(f"Evaluating {model_name}: {len(matches)} volumes")

    results = []
    pred_volumes = []
    gt_volumes = []

    for pid, pred_path, gt_path in tqdm(matches, desc=f"{model_name}"):
        try:
            pred_vol = ensure_01_range(load_volume(pred_path))
            gt_vol = ensure_01_range(load_volume(gt_path))

            # Handle shape mismatch by cropping to common size
            min_shape = tuple(min(p, g) for p, g in zip(pred_vol.shape, gt_vol.shape))
            pred_vol = pred_vol[: min_shape[0], : min_shape[1], : min_shape[2]]
            gt_vol = gt_vol[: min_shape[0], : min_shape[1], : min_shape[2]]

            psnr_val = compute_psnr(pred_vol, gt_vol, data_range=1.0)
            ssim_val = compute_ssim_3d(pred_vol, gt_vol, data_range=1.0)

            results.append({
                "patient_id": pid,
                "model": model_name,
                "PSNR": psnr_val,
                "SSIM": ssim_val,
                "shape": str(pred_vol.shape),
            })

            pred_volumes.append(pred_vol)
            gt_volumes.append(gt_vol)

        except Exception as e:
            logger.error(f"  {pid}: FAILED — {e}")
            results.append({
                "patient_id": pid,
                "model": model_name,
                "PSNR": float("nan"),
                "SSIM": float("nan"),
                "shape": "error",
            })

    return pd.DataFrame(results), pred_volumes, gt_volumes


def parse_args():
    parser = argparse.ArgumentParser(
        description="Compare 3D Latent Diffusion vs CyTran volumes",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    parser.add_argument(
        "--diffusion_dir", type=str, required=True,
        help="Folder of 3D .nii.gz volumes from the Diffusion model"
    )
    parser.add_argument(
        "--cyclegan_dir", type=str, required=True,
        help="Folder of 3D .nii.gz volumes from CyTran (inference_and_stitch.py output)"
    )
    parser.add_argument(
        "--gt_dir", type=str, default=None,
        help="Folder of ground truth 3D .nii.gz volumes (if separate from predictions)"
    )
    parser.add_argument(
        "--gt_from_cyclegan", action="store_true",
        help="Use *_ground_truth.nii.gz from cyclegan_dir as GT for both models"
    )
    parser.add_argument(
        "--diffusion_suffix", type=str, default="_pred",
        help="Suffix to strip from diffusion filenames to get patient ID"
    )
    parser.add_argument(
        "--cyclegan_suffix", type=str, default="_cytran_pred",
        help="Suffix to strip from CyTran filenames to get patient ID"
    )
    parser.add_argument(
        "--gt_suffix", type=str, default="_ground_truth",
        help="Suffix to strip from GT filenames to get patient ID"
    )
    parser.add_argument(
        "--output_dir", type=str, default="./comparison_results",
        help="Directory for output CSV, plots and FID"
    )
    parser.add_argument(
        "--compute_fid", action="store_true",
        help="Compute FID (requires torch + torchvision; can be slow)"
    )
    parser.add_argument(
        "--fid_batch_size", type=int, default=32,
        help="Batch size for FID feature extraction"
    )

    return parser.parse_args()


def main():
    args = parse_args()
    os.makedirs(args.output_dir, exist_ok=True)

    logger.info("=" * 70)
    logger.info("MODEL COMPARISON — 3D Latent Diffusion vs CyTran")
    logger.info("=" * 70)
    logger.info(f"Diffusion dir:  {args.diffusion_dir}")
    logger.info(f"CyTran dir:     {args.cyclegan_dir}")
    logger.info(f"GT dir:         {args.gt_dir}")
    logger.info(f"Compute FID:    {args.compute_fid}")
    logger.info(f"Output:         {args.output_dir}")

    # Determine GT directory
    if args.gt_from_cyclegan:
        gt_dir_diffusion = args.cyclegan_dir
        gt_dir_cyclegan = args.cyclegan_dir
        logger.info("Using GT volumes from CyTran output directory for both models.")
    elif args.gt_dir:
        gt_dir_diffusion = args.gt_dir
        gt_dir_cyclegan = args.gt_dir
    else:
        gt_dir_diffusion = args.diffusion_dir
        gt_dir_cyclegan = args.cyclegan_dir
        logger.info("No explicit GT dir — assuming GT files colocated with predictions.")

    # -- Evaluate Diffusion Model --
    logger.info("")
    diff_df, diff_vols, diff_gt_vols = evaluate_model(
        pred_dir=args.diffusion_dir,
        gt_dir=gt_dir_diffusion,
        model_name="Diffusion",
        pred_suffix=args.diffusion_suffix,
        gt_suffix=args.gt_suffix,
    )

    # -- Evaluate CyTran --
    logger.info("")
    cyc_df, cyc_vols, cyc_gt_vols = evaluate_model(
        pred_dir=args.cyclegan_dir,
        gt_dir=gt_dir_cyclegan,
        model_name="CyTran",
        pred_suffix=args.cyclegan_suffix,
        gt_suffix=args.gt_suffix,
    )

    # -- Combine results --
    combined_df = pd.concat([diff_df, cyc_df], ignore_index=True)
    combined_path = os.path.join(args.output_dir, "per_patient_metrics.csv")
    combined_df.to_csv(combined_path, index=False)
    logger.info(f"\nPer-patient metrics saved to: {combined_path}")

    # -- Summary statistics --
    summary_rows = []
    for model_name, df in [("Diffusion", diff_df), ("CyTran", cyc_df)]:
        if df.empty:
            continue
        row = {
            "Model": model_name,
            "N": len(df),
            "PSNR_mean": df["PSNR"].mean(),
            "PSNR_std": df["PSNR"].std(),
            "SSIM_mean": df["SSIM"].mean(),
            "SSIM_std": df["SSIM"].std(),
        }
        summary_rows.append(row)

    summary_df = pd.DataFrame(summary_rows)

    # -- FID (optional) --
    if args.compute_fid and diff_vols and cyc_vols:
        logger.info("\nComputing FID (extracting Inception features) ...")

        # Extract 2D slices for FID
        gt_slices = extract_all_slices(diff_gt_vols if diff_gt_vols else cyc_gt_vols)
        diff_slices = extract_all_slices(diff_vols) if diff_vols else None
        cyc_slices = extract_all_slices(cyc_vols) if cyc_vols else None

        gt_feats = _get_inception_features(gt_slices, batch_size=args.fid_batch_size)

        if gt_feats is not None:
            if diff_slices is not None:
                diff_feats = _get_inception_features(diff_slices, batch_size=args.fid_batch_size)
                fid_diff = compute_fid(diff_feats, gt_feats) if diff_feats is not None else float("nan")
            else:
                fid_diff = float("nan")

            if cyc_slices is not None:
                cyc_feats = _get_inception_features(cyc_slices, batch_size=args.fid_batch_size)
                fid_cyc = compute_fid(cyc_feats, gt_feats) if cyc_feats is not None else float("nan")
            else:
                fid_cyc = float("nan")

            # Add FID to summary
            for row in summary_rows:
                if row["Model"] == "Diffusion":
                    row["FID"] = fid_diff
                elif row["Model"] == "CyTran":
                    row["FID"] = fid_cyc

            summary_df = pd.DataFrame(summary_rows)
            logger.info(f"FID — Diffusion: {fid_diff:.2f},  CyTran: {fid_cyc:.2f}")
        else:
            logger.warning("Inception features unavailable — FID not computed.")

    # -- Save summary --
    summary_path = os.path.join(args.output_dir, "summary.csv")
    summary_df.to_csv(summary_path, index=False)

    # Pretty print
    logger.info("")
    logger.info("=" * 70)
    logger.info("SUMMARY")
    logger.info("=" * 70)
    for _, row in summary_df.iterrows():
        logger.info(f"  {row['Model']:12s}  |  "
                     f"PSNR: {row['PSNR_mean']:.2f} +/- {row['PSNR_std']:.2f}  |  "
                     f"SSIM: {row['SSIM_mean']:.4f} +/- {row['SSIM_std']:.4f}"
                     + (f"  |  FID: {row['FID']:.2f}" if "FID" in row and not np.isnan(row.get("FID", float("nan"))) else ""))
    logger.info("=" * 70)

    # -- Save as JSON for easy programmatic access --
    summary_json = summary_df.to_dict(orient="records")
    json_path = os.path.join(args.output_dir, "summary.json")
    with open(json_path, "w") as f:
        json.dump(summary_json, f, indent=2, default=str)
    logger.info(f"Results saved to: {args.output_dir}")


if __name__ == "__main__":
    main()
