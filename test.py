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
import matplotlib.gridspec as gridspec
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


# ============================================================================
# ROI vessel metrics (identical to ctpa_medvae_latent_diffusion/evaluate.py)
# ============================================================================

def compute_cnr(subtraction: np.ndarray, vessel_mask: np.ndarray) -> float:
    """
    Compute Contrast-to-Noise Ratio.
    CNR = (mean_vessel - mean_background) / std_background
    """
    if vessel_mask.sum() == 0 or (~vessel_mask).sum() == 0:
        return 0.0
    mean_vessel = float(subtraction[vessel_mask].mean())
    bg = subtraction[~vessel_mask]
    std_bg = float(bg.std())
    if std_bg < 1e-8:
        return 0.0
    return float((mean_vessel - float(bg.mean())) / std_bg)


def compute_vessel_dice(
    pred_sub: np.ndarray,
    gt_sub: np.ndarray,
    threshold_hu: float = 50.0,
    hu_range: float = 1000.0,
) -> float:
    """Dice coefficient on thresholded vessel regions (subtraction in [-1,1])."""
    threshold_norm = threshold_hu / hu_range
    pred_mask = pred_sub > threshold_norm
    gt_mask = gt_sub > threshold_norm
    intersection = (pred_mask & gt_mask).sum()
    total = pred_mask.sum() + gt_mask.sum()
    if total == 0:
        return 1.0
    return float(2 * intersection / total)


def create_vessel_mask(
    subtraction: np.ndarray,
    threshold_hu: float = 50.0,
    hu_range: float = 1000.0,
) -> np.ndarray:
    """Binary vessel mask from a subtraction image in [-1,1]."""
    return subtraction > (threshold_hu / hu_range)


def compute_roi_metrics(
    pred_sub: np.ndarray,
    gt_sub: np.ndarray,
    vessel_mask: np.ndarray,
    data_range: float = 2.0,
) -> dict:
    """
    Compute metrics restricted to vessel ROI region.

    Args:
        pred_sub: Predicted subtraction (D, H, W) in [-1, 1]
        gt_sub:   Ground-truth subtraction (D, H, W) in [-1, 1]
        vessel_mask: Boolean mask of vessel regions (from GT subtraction)
        data_range: Data range for PSNR (2.0 for [-1, 1])
    """
    _HU_RANGE = 1000.0
    metrics = {
        "roi_dice": compute_vessel_dice(pred_sub, gt_sub),
        "roi_cnr_gt": compute_cnr(gt_sub, vessel_mask),
        "roi_cnr_pred": compute_cnr(pred_sub, vessel_mask),
    }
    if vessel_mask.sum() > 0:
        metrics["roi_mae"] = float(np.mean(np.abs(pred_sub[vessel_mask] - gt_sub[vessel_mask])))
        metrics["roi_mae_hu"] = metrics["roi_mae"] * _HU_RANGE
        mse = float(np.mean((pred_sub[vessel_mask] - gt_sub[vessel_mask]) ** 2))
        metrics["roi_psnr"] = float(10 * np.log10((data_range ** 2) / mse)) if mse > 0 else float('inf')
    else:
        metrics["roi_mae"] = 0.0
        metrics["roi_mae_hu"] = 0.0
        metrics["roi_psnr"] = float('inf')
    return metrics


# ============================================================================
# FID metric (identical to ctpa_medvae_latent_diffusion/evaluate.py)
# ============================================================================

def extract_centre_axial_slice_uint8(volume: np.ndarray) -> np.ndarray:
    """Extract centre axial slice from (D,H,W) volume in [-1,1] as uint8 (H,W,3)."""
    sl = volume[volume.shape[0] // 2]
    sl_uint8 = np.clip((sl + 1.0) / 2.0 * 255, 0, 255).astype(np.uint8)
    return np.stack([sl_uint8] * 3, axis=-1)


def compute_fid_2d(real_slices: list, fake_slices: list, device) -> float:
    """
    Compute FID between two sets of (H,W,3) uint8 arrays using InceptionV3.
    Requires torchmetrics.
    """
    import os
    # Redirect torch/hub cache to a writable location (avoids PermissionError on /.cache in containers)
    os.environ.setdefault("TORCH_HOME", "/cephfs/eidf212/shared/odiamant/.cache/torch")
    os.makedirs(os.environ["TORCH_HOME"], exist_ok=True)
    from torchmetrics.image.fid import FrechetInceptionDistance
    fid_metric = FrechetInceptionDistance(feature=2048, normalize=False).to(device)
    fid_metric.reset()
    for sl in real_slices:
        t = torch.from_numpy(sl).permute(2, 0, 1).unsqueeze(0).to(device)
        fid_metric.update(t, real=True)
    for sl in fake_slices:
        t = torch.from_numpy(sl).permute(2, 0, 1).unsqueeze(0).to(device)
        fid_metric.update(t, real=False)
    return float(fid_metric.compute())


# ============================================================================
# FRD metric (Fréchet Radiomic Distance) — copied from evaluate_2d.py
# Konz et al., 2026
# ============================================================================

_FRD_GLCM_FEATURES = [
    "Autocorrelation", "JointAverage", "ClusterProminence", "ClusterShade",
    "ClusterTendency", "Contrast", "Correlation", "DifferenceAverage",
    "DifferenceEntropy", "DifferenceVariance", "JointEnergy", "JointEntropy",
    "Imc1", "Imc2", "Idm", "Idmn", "Id", "Idn", "InverseVariance",
    "MaximumProbability", "SumEntropy", "SumSquares",
]


def extract_slices_for_radiomics(volume: np.ndarray, num_slices: int = 5) -> list:
    """(H, W, D) float32 in [0, 1] → list of num_slices (H, W) float32 arrays."""
    D = volume.shape[2]
    indices = np.linspace(D // 8, 7 * D // 8, num_slices).astype(int)
    return [volume[:, :, int(i)].astype(np.float32) for i in indices]


def _build_radiomic_extractor(use_wavelet: bool = True):
    """Create a configured PyRadiomics feature extractor (lazy, cached via caller)."""
    try:
        from radiomics import featureextractor
        import logging
    except ImportError as e:
        raise ImportError(
            "PyRadiomics is required for FRD. Install with: pip install pyradiomics"
        ) from e

    import logging as _logging
    _logging.getLogger("radiomics").setLevel(_logging.ERROR)

    settings = {
        "binWidth": 5,
        "normalize": True,
        "normalizeScale": 100,
        "voxelArrayShift": 300,
        "force2D": True,
        "force2Ddimension": 0,
        "label": 1,
        "verbose": False,
        "geometryTolerance": 1e-6,
    }
    extractor = featureextractor.RadiomicsFeatureExtractor(**settings)
    extractor.disableAllFeatures()
    extractor.enableFeatureClassByName("firstorder")
    for feat_name in _FRD_GLCM_FEATURES:
        extractor.enableFeaturesByName(glcm=[feat_name])
    extractor.enableFeatureClassByName("glrlm")
    extractor.enableFeatureClassByName("glszm")
    extractor.enableFeatureClassByName("ngtdm")
    extractor.disableAllImageTypes()
    extractor.enableImageTypeByName("Original")
    if use_wavelet:
        extractor.enableImageTypeByName("LoG", customArgs={"sigma": [2.0, 3.0, 4.0, 5.0]})
        extractor.enableImageTypeByName("Wavelet")
    return extractor


def check_frd_runtime_compatibility() -> tuple:
    """Validate FRD dependencies early to avoid failing only at the end of evaluation."""
    np_major = int(np.__version__.split(".", 1)[0])
    if np_major >= 2:
        return (
            False,
            f"NumPy {np.__version__} detected. FRD with current PyRadiomics build requires NumPy < 2.",
        )
    try:
        import radiomics  # noqa: F401
    except Exception as exc:
        return False, f"PyRadiomics import failed: {exc}"
    return True, f"FRD dependencies OK (NumPy {np.__version__})"


def extract_radiomic_features(images: list, extractor) -> np.ndarray:
    """Extract radiomic features from a list of (H, W) float32 [0,1] images.

    Returns (N, ~464) float32 array; failed extractions → zeros.
    """
    import SimpleITK as sitk

    all_features = []
    n_failed = 0
    fallback = None
    for image in tqdm(images, desc="Radiomics", leave=False):
        try:
            img_uint8 = np.clip(image * 255, 0, 255).astype(np.uint8)
            img_3d = img_uint8[np.newaxis, :, :].astype(np.float32)  # (1, H, W)
            sitk_image = sitk.GetImageFromArray(img_3d)
            sitk_image.SetSpacing((1.0, 1.0, 1.0))
            mask_3d = np.ones((1, img_uint8.shape[0], img_uint8.shape[1]), dtype=np.uint8)
            mask_3d[0][0][0] = 0  # corner=0 workaround for PyRadiomics bug #765
            sitk_mask = sitk.GetImageFromArray(mask_3d)
            sitk_mask.SetSpacing((1.0, 1.0, 1.0))
            result = extractor.execute(sitk_image, sitk_mask)
            feats = np.array(
                [float(v) for k, v in sorted(result.items())
                 if not k.startswith("diagnostics_")],
                dtype=np.float32,
            )
            if fallback is None:
                fallback = np.zeros_like(feats)
            all_features.append(feats)
        except Exception:
            n_failed += 1
            all_features.append(None)

    if fallback is None:
        fallback = np.zeros(1, dtype=np.float32)
    all_features = [f if f is not None else np.zeros_like(fallback) for f in all_features]
    if n_failed > 0:
        print(f"  [FRD] Skipped {n_failed}/{len(images)} images due to extraction errors.")
    features = np.stack(all_features, axis=0)
    return np.nan_to_num(features, nan=0.0, posinf=0.0, neginf=0.0)


def _zscore_and_pca(features_ref: np.ndarray, features_test: np.ndarray):
    """Z-score normalise (using ref stats) then PCA-reduce to full-rank size."""
    mu = np.mean(features_ref, axis=0, keepdims=True)
    std = np.std(features_ref, axis=0, keepdims=True)
    std = np.where(std < 1e-10, 1.0, std)
    ref_norm = (features_ref - mu) / std
    test_norm = (features_test - mu) / std
    n, d = ref_norm.shape
    n_components = max(1, min(n - 1, d))
    ref_centered = ref_norm - ref_norm.mean(axis=0)
    _, _, Vt = np.linalg.svd(ref_centered, full_matrices=False)
    components = Vt[:n_components]
    ref_pca = ref_centered @ components.T
    test_pca = (test_norm - ref_norm.mean(axis=0)) @ components.T
    return ref_pca, test_pca


def _frechet_distance(mu1: np.ndarray, sigma1: np.ndarray,
                      mu2: np.ndarray, sigma2: np.ndarray,
                      eps: float = 1e-6) -> float:
    """Fréchet distance between two multivariate Gaussians."""
    from scipy import linalg
    diff = mu1 - mu2
    covmean, _ = linalg.sqrtm(sigma1 @ sigma2, disp=False)
    if np.iscomplexobj(covmean):
        if not np.allclose(np.imag(covmean), 0, atol=1e-2):
            offset = np.eye(sigma1.shape[0]) * eps
            covmean, _ = linalg.sqrtm((sigma1 + offset) @ (sigma2 + offset), disp=False)
        covmean = np.real(covmean)
    if not np.isfinite(covmean).all():
        offset = np.eye(sigma1.shape[0]) * eps
        covmean, _ = linalg.sqrtm((sigma1 + offset) @ (sigma2 + offset), disp=False)
        covmean = np.real(covmean)
    return float(diff @ diff + np.trace(sigma1 + sigma2 - 2 * covmean))


def compute_frd(real_images: list, fake_images: list) -> float:
    """Compute Fréchet Radiomic Distance between two sets of (H, W) float32 [0,1] images."""
    try:
        extractor = _build_radiomic_extractor(use_wavelet=True)
    except ImportError as e:
        print(f"  [FRD] Skipping: {e}")
        return float("nan")
    print("  Extracting radiomic features for real images ...")
    feats_real = extract_radiomic_features(real_images, extractor)
    print("  Extracting radiomic features for fake images ...")
    feats_fake = extract_radiomic_features(fake_images, extractor)
    ref_pca, test_pca = _zscore_and_pca(feats_real, feats_fake)
    mu1 = np.mean(ref_pca, axis=0)
    sigma1 = np.cov(ref_pca, rowvar=False)
    mu2 = np.mean(test_pca, axis=0)
    sigma2 = np.cov(test_pca, rowvar=False)
    return _frechet_distance(mu1, sigma1, mu2, sigma2)


# ============================================================================
# Publication-quality visualization (matching diffusion model evaluate.py)
# ============================================================================

def apply_ct_window(
    data: np.ndarray,
    level: float = 0.0,
    width: float = 2000.0,
    hu_range: float = 1000.0,
) -> np.ndarray:
    """
    Apply CT windowing to data in [-1, 1] range for display.

    Converts from the model's [-1, 1] normalized space (mapping to
    [-hu_range, hu_range] HU) into a [0, 1] display range using the
    specified window level and width.
    """
    min_hu = level - width / 2.0
    max_hu = level + width / 2.0
    min_val = min_hu / hu_range
    max_val = max_hu / hu_range
    windowed = (data - min_val) / (max_val - min_val)
    return np.clip(windowed, 0.0, 1.0)


def compute_slice_mae(pred_slice: np.ndarray, target_slice: np.ndarray) -> float:
    """Compute Mean Absolute Error for a single 2D slice."""
    return float(np.mean(np.abs(pred_slice - target_slice)))


def compute_slice_ssim_2d(
    pred_slice: np.ndarray,
    target_slice: np.ndarray,
    data_range: float = 2.0,
    win_size: int = 7,
) -> float:
    """
    Compute 2D SSIM for a single slice using uniform-filter local statistics.
    Operates on [-1, 1] data (data_range=2.0).
    """
    C1 = (0.01 * data_range) ** 2
    C2 = (0.03 * data_range) ** 2

    pred_f = pred_slice.astype(np.float64)
    tgt_f = target_slice.astype(np.float64)

    mu_p = uniform_filter(pred_f, size=win_size)
    mu_t = uniform_filter(tgt_f, size=win_size)

    sigma_p_sq = uniform_filter(pred_f ** 2, size=win_size) - mu_p ** 2
    sigma_t_sq = uniform_filter(tgt_f ** 2, size=win_size) - mu_t ** 2
    sigma_pt = uniform_filter(pred_f * tgt_f, size=win_size) - mu_p * mu_t

    num = (2 * mu_p * mu_t + C1) * (2 * sigma_pt + C2)
    den = (mu_p ** 2 + mu_t ** 2 + C1) * (sigma_p_sq + sigma_t_sq + C2)

    return float(np.mean(num / den))


def visualize_slices(
    source: np.ndarray,
    pred: np.ndarray,
    target: np.ndarray,
    patient_id: str,
    output_path: str,
    slice_indices=None,
    num_slices: int = 1,
    metrics=None,
    ct_window_level: float = 0.0,
    ct_window_width: float = 2000.0,
    error_percentile: float = 99.0,
) -> None:
    """
    Create publication-quality visualization comparing source, predicted, and ground truth slices.
    Identical to ctpa_medvae_latent_diffusion/evaluate.py visualize_slices().

    Args:
        source, pred, target: Volumes in (D, H, W) or (C, D, H, W), values in [-1, 1].
    """
    if source.ndim == 4:
        source = source.squeeze(0)
    if pred.ndim == 4:
        pred = pred.squeeze(0)
    if target.ndim == 4:
        target = target.squeeze(0)

    depth = source.shape[0]
    D, H, W = source.shape

    slice_aspect = H / W if H != W else 1.0

    if slice_indices is None:
        central_slice = depth // 2
        if num_slices == 1:
            indices = [central_slice]
        elif num_slices == 2:
            indices = [central_slice, depth // 4]
        else:
            indices = list(np.linspace(depth // 8, 7 * depth // 8, num_slices).astype(int))
    else:
        indices = slice_indices

    num_slices_actual = len(indices)

    source_disp = apply_ct_window(source, level=ct_window_level, width=ct_window_width)
    pred_disp   = apply_ct_window(pred,   level=ct_window_level, width=ct_window_width)
    target_disp = apply_ct_window(target, level=ct_window_level, width=ct_window_width)

    diff = np.abs(pred - target)

    err_vmax = float(np.percentile(diff, error_percentile))
    if err_vmax < 1e-6:
        err_vmax = 1.0

    sample_slice = source_disp[indices[0]].T
    panel_rows, panel_cols = sample_slice.shape
    effective_h = panel_rows * (slice_aspect if isinstance(slice_aspect, (int, float)) and slice_aspect != 1.0 else 1.0)
    effective_w = panel_cols
    panel_ratio = effective_h / effective_w

    panel_width = 5.5
    panel_height = panel_width * panel_ratio
    fig_width  = 4 * panel_width + 0.6
    fig_height = num_slices_actual * panel_height + 1.5
    fig = plt.figure(figsize=(fig_width, fig_height), dpi=300)
    gs = gridspec.GridSpec(
        num_slices_actual + 1, 5,
        height_ratios=[1] * num_slices_actual + [0.05],
        width_ratios=[1, 1, 1, 1, 0.03],
        wspace=0.05, hspace=0.15,
    )

    base_titles = ['Source (Contrast)', 'Predicted', 'Ground Truth', 'Absolute Error']
    im_err = None

    for row, slice_idx in enumerate(indices):
        sl_mae  = compute_slice_mae(pred[slice_idx], target[slice_idx])
        sl_ssim = compute_slice_ssim_2d(pred[slice_idx], target[slice_idx])

        volumes = [
            source_disp[slice_idx],
            pred_disp[slice_idx],
            target_disp[slice_idx],
            diff[slice_idx],
        ]
        cmaps = ['gray', 'gray', 'gray', 'inferno']

        for col, (vol, cmap) in enumerate(zip(volumes, cmaps)):
            ax = fig.add_subplot(gs[row, col])

            if col < 3:
                ax.imshow(vol.T, cmap=cmap, origin='lower', vmin=0, vmax=1, aspect=slice_aspect)
            else:
                im_err = ax.imshow(
                    vol.T, cmap=cmap, origin='lower',
                    vmin=0, vmax=err_vmax, aspect=slice_aspect,
                )

            ax.set_xticks([])
            ax.set_yticks([])

            if row == 0:
                ax.set_title(base_titles[col], fontsize=16, fontweight='bold', pad=12)

            if col == 1:
                ax.set_xlabel(
                    f'MAE={sl_mae:.4f}  SSIM={sl_ssim:.4f}',
                    fontsize=11, labelpad=4,
                )

            if col == 0:
                ax.set_ylabel(f'Slice {slice_idx}', fontsize=14, fontweight='bold', labelpad=10)

            for spine in ax.spines.values():
                spine.set_visible(False)

    if im_err is not None:
        cbar_ax = fig.add_subplot(gs[:num_slices_actual, 4])
        cbar = fig.colorbar(im_err, cax=cbar_ax, orientation='vertical')
        cbar.set_label('Absolute Error (data units)', fontsize=14, fontweight='bold', labelpad=10)
        cbar.ax.tick_params(labelsize=12)

    title_text = f'Patient: {patient_id}  |  Window L={ct_window_level} W={ct_window_width}'
    if metrics:
        metrics_text = (
            f"Volume PSNR: {metrics.get('psnr', 0):.2f} dB | "
            f"SSIM: {metrics.get('ssim', 0):.4f} | "
            f"MAE: {metrics.get('mae', 0):.4f}"
        )
        fig.text(
            0.5, 0.01, metrics_text, ha='center', fontsize=14, fontweight='bold',
            bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.9,
                      edgecolor='gray', linewidth=1.5),
        )

    fig.suptitle(title_text, fontsize=18, fontweight='bold', y=0.99)

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    plt.savefig(output_path, dpi=300, bbox_inches='tight', facecolor='white',
                pad_inches=0.15, transparent=False)
    plt.close()

    print(f"  Saved visualization: {output_path}")


def visualize_3view(
    source: np.ndarray,
    pred: np.ndarray,
    target: np.ndarray,
    patient_id: str,
    output_path: str,
    metrics=None,
    ct_window_level: float = 0.0,
    ct_window_width: float = 2000.0,
    error_percentile: float = 99.0,
) -> None:
    """
    Create 3-view visualization (sagittal, coronal, axial) for a single patient.
    Identical to ctpa_medvae_latent_diffusion/evaluate.py visualize_3view().

    Args:
        source, pred, target: Volumes in (D, H, W) or (C, D, H, W), values in [-1, 1].
    """
    if source.ndim == 4:
        source = source.squeeze(0)
    if pred.ndim == 4:
        pred = pred.squeeze(0)
    if target.ndim == 4:
        target = target.squeeze(0)

    D, H, W = source.shape

    d_idx = D // 2
    h_idx = H // 2
    w_idx = W // 2

    source_disp = apply_ct_window(source, level=ct_window_level, width=ct_window_width)
    pred_disp   = apply_ct_window(pred,   level=ct_window_level, width=ct_window_width)
    target_disp = apply_ct_window(target, level=ct_window_level, width=ct_window_width)

    diff = np.abs(pred - target)

    views = [
        ('axial', 'Axial View',
         source_disp[d_idx], pred_disp[d_idx], target_disp[d_idx], diff[d_idx],
         source[d_idx], pred[d_idx], target[d_idx],
         True, H / W if H != W else 1.0),
        ('coronal', 'Coronal View',
         source_disp[:, h_idx, :], pred_disp[:, h_idx, :], target_disp[:, h_idx, :], diff[:, h_idx, :],
         source[:, h_idx, :], pred[:, h_idx, :], target[:, h_idx, :],
         False, W / D if W != D else 1.0),
        ('sagittal', 'Sagittal View',
         source_disp[:, :, w_idx], pred_disp[:, :, w_idx], target_disp[:, :, w_idx], diff[:, :, w_idx],
         source[:, :, w_idx], pred[:, :, w_idx], target[:, :, w_idx],
         True, D / H if D != H else 1.0),
    ]

    col_titles = ['Source (Contrast)', 'Predicted', 'Ground Truth', 'Absolute Error']

    output_base = str(output_path)
    if output_base.endswith('.png'):
        output_base = output_base[:-4]

    for (view_fn, view_title, src_d, prd_d, tgt_d, dif_d,
         src_raw, prd_raw, tgt_raw, do_transpose, aspect_val) in views:

        sl_mae  = compute_slice_mae(prd_raw, tgt_raw)
        sl_ssim = compute_slice_ssim_2d(prd_raw, tgt_raw)

        view_err_vmax = float(np.percentile(dif_d, error_percentile))
        if view_err_vmax < 1e-6:
            view_err_vmax = 1.0

        def orient(arr):
            return arr.T if do_transpose else arr

        slices = [orient(src_d), orient(prd_d), orient(tgt_d), orient(dif_d)]
        cmaps  = ['gray', 'gray', 'gray', 'inferno']

        img_h, img_w = slices[0].shape
        if isinstance(aspect_val, (int, float)) and aspect_val != 1:
            display_h = img_h * aspect_val
        else:
            display_h = img_h
        display_w = img_w
        panel_aspect = display_h / display_w

        base_width = 5
        panel_height = base_width * panel_aspect
        fig_width  = 4 * base_width + 0.8
        fig_height = panel_height + 1.6

        fig = plt.figure(figsize=(fig_width, fig_height), dpi=300)
        gs = gridspec.GridSpec(1, 5, width_ratios=[1, 1, 1, 1, 0.04], wspace=0.03)

        im_err = None
        for col, (slc, cmap) in enumerate(zip(slices, cmaps)):
            ax = fig.add_subplot(gs[0, col])
            if col < 3:
                ax.imshow(slc, cmap=cmap, origin='lower', vmin=0, vmax=1, aspect=aspect_val)
            else:
                im_err = ax.imshow(
                    slc, cmap=cmap, origin='lower',
                    vmin=0, vmax=view_err_vmax, aspect=aspect_val,
                )
            ax.set_xticks([])
            ax.set_yticks([])

            if col == 1:
                title = f'{col_titles[col]}\nMAE={sl_mae:.4f}  SSIM={sl_ssim:.4f}'
            else:
                title = col_titles[col]
            ax.set_title(title, fontsize=13, fontweight='bold', pad=10)

            for spine in ax.spines.values():
                spine.set_visible(False)

        if im_err is not None:
            cbar_ax = fig.add_subplot(gs[0, 4])
            cbar = fig.colorbar(im_err, cax=cbar_ax, orientation='vertical')
            cbar.set_label('Absolute Error', fontsize=12, fontweight='bold')
            cbar.ax.tick_params(labelsize=10)

        title_text = f'Patient: {patient_id} \u2014 {view_title}  (Window L={ct_window_level} W={ct_window_width})'
        if metrics:
            metrics_text = (
                f"Vol PSNR: {metrics.get('psnr', 0):.2f} dB | "
                f"SSIM: {metrics.get('ssim', 0):.4f}"
            )
            title_text += f'\n{metrics_text}'
        fig.suptitle(title_text, fontsize=15, fontweight='bold', y=0.99)

        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        view_output_path = f"{output_base}_{view_fn}.png"
        plt.savefig(view_output_path, dpi=300, bbox_inches='tight', facecolor='white',
                    pad_inches=0.1, transparent=False)
        plt.close()

        print(f"  Saved {view_title}: {view_output_path}")


def visualize_subtractions(
    source: np.ndarray,
    pred: np.ndarray,
    target: np.ndarray,
    patient_id: str,
    output_path: str,
    num_slices: int = 5,
    slice_indices=None,
    metrics=None,
    error_percentile: float = 99.0,
) -> None:
    """
    Create 3-view subtraction comparison plots (sagittal, coronal, axial).

    Each view produces a separate figure with 4 columns:
    Source (Contrast) | GT Subtraction | Synth Subtraction | Absolute Difference

    Args:
        source: Source (contrast) volume (D, H, W) or (C, D, H, W), values in [-1, 1].
        pred: Predicted (generated non-contrast) volume
        target: Target (ground truth non-contrast) volume
        patient_id: Patient identifier for title
        output_path: Path to save the figure (base; view suffix appended)
        num_slices: Unused (kept for API compatibility)
        slice_indices: Unused (kept for API compatibility)
        metrics: Optional dict with subtraction metrics (sub_psnr, sub_ssim, etc.)
        error_percentile: Percentile to clamp the difference colour scale (default 99)
    """
    if source.ndim == 4:
        source = source.squeeze(0)
    if pred.ndim == 4:
        pred = pred.squeeze(0)
    if target.ndim == 4:
        target = target.squeeze(0)

    D, H, W = source.shape

    # Compute subtraction volumes in [-1, 1] data space
    gt_sub = source - target       # GT subtraction
    synth_sub = source - pred      # Synthesized subtraction
    diff = np.abs(gt_sub - synth_sub)

    # CT-windowed source for display
    source_disp = apply_ct_window(source, level=0.0, width=2000.0)

    # Central slices for each view
    d_idx = D // 2
    h_idx = H // 2
    w_idx = W // 2

    views = [
        ('axial', 'Axial View',
         lambda v: v[d_idx], True, H / W if H != W else 1.0),
        ('coronal', 'Coronal View',
         lambda v: v[:, h_idx, :], False, W / D if W != D else 1.0),
        ('sagittal', 'Sagittal View',
         lambda v: v[:, :, w_idx], True, D / H if D != H else 1.0),
    ]

    col_titles = [
        'Source (Contrast)',
        'GT Subtraction\n(source \u2212 ground truth)',
        'Synth Subtraction\n(source \u2212 predicted)',
        'Absolute Difference',
    ]

    output_base = str(output_path)
    if output_base.endswith('.png'):
        output_base = output_base[:-4]

    for view_key, view_label, slicer, do_transpose, aspect_val in views:
        src_sl = slicer(source_disp)
        gt_sl = slicer(gt_sub)
        syn_sl = slicer(synth_sub)
        diff_sl = slicer(diff)

        def orient(arr):
            return arr.T if do_transpose else arr

        src_d = orient(src_sl)
        gt_d = orient(gt_sl)
        syn_d = orient(syn_sl)
        dif_d = orient(diff_sl)

        # Per-view metrics
        sl_mae = compute_slice_mae(slicer(synth_sub), slicer(gt_sub))
        sl_ssim = compute_slice_ssim_2d(slicer(synth_sub), slicer(gt_sub))

        # Colour ranges for this view
        sub_vmax = float(np.percentile(
            np.abs(np.concatenate([gt_d.ravel(), syn_d.ravel()])),
            error_percentile,
        ))
        if sub_vmax < 1e-6:
            sub_vmax = 1.0

        diff_vmax = float(np.percentile(dif_d, error_percentile))
        if diff_vmax < 1e-6:
            diff_vmax = 1.0

        # Dynamic figure sizing
        img_h, img_w = src_d.shape
        display_h = img_h * aspect_val if aspect_val != 1 else img_h
        display_w = img_w
        panel_aspect = display_h / display_w

        base_width = 5
        panel_height = base_width * panel_aspect
        fig_width = 4 * base_width + 0.8
        fig_height = panel_height + 1.6

        fig = plt.figure(figsize=(fig_width, fig_height), dpi=300)
        gs = gridspec.GridSpec(1, 5, width_ratios=[1, 1, 1, 1, 0.04], wspace=0.03)

        slices = [src_d, gt_d, syn_d, dif_d]
        cmaps = ['gray', 'gray', 'gray', 'inferno']
        im_diff = None

        for col, (slc, cmap) in enumerate(zip(slices, cmaps)):
            ax = fig.add_subplot(gs[0, col])

            if col == 0:
                ax.imshow(slc, cmap=cmap, origin='lower', vmin=0, vmax=1, aspect=aspect_val)
            elif col < 3:
                ax.imshow(slc, cmap=cmap, origin='lower', vmin=-sub_vmax, vmax=sub_vmax, aspect=aspect_val)
            else:
                im_diff = ax.imshow(
                    slc, cmap=cmap, origin='lower',
                    vmin=0, vmax=diff_vmax, aspect=aspect_val,
                )

            ax.set_xticks([])
            ax.set_yticks([])

            if col == 2:
                title = f'{col_titles[col]}\nMAE={sl_mae:.4f}  SSIM={sl_ssim:.4f}'
            else:
                title = col_titles[col]
            ax.set_title(title, fontsize=13, fontweight='bold', pad=10)

            for spine in ax.spines.values():
                spine.set_visible(False)

        if im_diff is not None:
            cbar_ax = fig.add_subplot(gs[0, 4])
            cbar = fig.colorbar(im_diff, cax=cbar_ax, orientation='vertical')
            cbar.set_label('Absolute Difference', fontsize=12, fontweight='bold')
            cbar.ax.tick_params(labelsize=10)

        title_text = f'Patient: {patient_id} \u2014 {view_label}  (Subtraction)'
        if metrics:
            metrics_text = (
                f"Sub PSNR: {metrics.get('sub_psnr', 0):.2f} dB | "
                f"Sub SSIM: {metrics.get('sub_ssim', 0):.4f}"
            )
            title_text += f'\n{metrics_text}'
        fig.suptitle(title_text, fontsize=15, fontweight='bold', y=0.99)

        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        view_output_path = f"{output_base}_{view_key}.png"
        plt.savefig(view_output_path, dpi=300, bbox_inches='tight', facecolor='white',
                    pad_inches=0.1, transparent=False)
        plt.close()

        print(f"  Saved subtraction {view_label}: {view_output_path}")


def create_summary_visualization(
    results,
    output_path: str,
) -> None:
    """Create summary bar chart of metrics across all patients."""
    df = pd.DataFrame(results)

    fig, axes = plt.subplots(1, 4, figsize=(16, 4))

    metric_names = ['psnr', 'ssim', 'mae', 'rmse']
    titles = ['PSNR (dB) \u2191', 'SSIM \u2191', 'MAE \u2193', 'RMSE \u2193']
    colors = ['steelblue', 'seagreen', 'coral', 'mediumpurple']

    for ax, metric, title, color in zip(axes, metric_names, titles, colors):
        values = df[metric].values
        x = range(len(values))

        ax.bar(x, values, color=color, alpha=0.7, edgecolor='black')
        ax.axhline(y=values.mean(), color='red', linestyle='--', linewidth=2, label=f'Mean: {values.mean():.3f}')
        ax.set_xlabel('Patient Index')
        ax.set_ylabel(metric.upper())
        ax.set_title(title, fontweight='bold')
        ax.legend()
        ax.grid(axis='y', alpha=0.3)

    plt.tight_layout()
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    plt.savefig(output_path, dpi=150, bbox_inches='tight', facecolor='white')
    plt.close()

    print(f"Saved summary visualization: {output_path}")


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

    # --- Publication-quality visualizations (matching diffusion model) ---
    # Convert [0, 1] → [-1, 1] and transpose (H, W, D) → (D, H, W)
    src_dhw = np.transpose(input_volume * 2.0 - 1.0, (2, 0, 1))   # (D, H, W) in [-1, 1]
    gen_dhw = np.transpose(generated_volume * 2.0 - 1.0, (2, 0, 1))
    gt_dhw  = np.transpose(gt_volume * 2.0 - 1.0, (2, 0, 1))

    viz_dir = os.path.join(output_dir, "visualizations")

    # Multi-slice comparison (3 slices)
    visualize_slices(
        src_dhw, gen_dhw, gt_dhw, patient_id,
        os.path.join(viz_dir, f"{patient_id}_slices.png"),
        num_slices=3, metrics=metrics_dict,
        ct_window_level=0.0, ct_window_width=2000.0,
    )

    # 3-view anatomical visualization (sagittal, coronal, axial)
    visualize_3view(
        src_dhw, gen_dhw, gt_dhw, patient_id,
        os.path.join(viz_dir, f"{patient_id}_3view.png"),
        metrics=metrics_dict,
        ct_window_level=0.0, ct_window_width=2000.0,
    )

    return generated_volume, gt_volume, input_volume, metrics_dict, src_dhw, gen_dhw, gt_dhw


if __name__ == "__main__":
    opt = TestOptions().parse()
    opt.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    subtraction_eval = opt.subtraction_eval
    save_subtractions = opt.save_subtractions
    roi_eval = opt.roi_eval
    vessel_threshold_hu = opt.vessel_threshold_hu
    fid_eval = opt.fid_eval
    frd_eval = opt.frd_eval
    frd_num_slices = opt.frd_num_slices
    save_generated_dir = opt.save_generated_dir
    load_generated_dir = opt.load_generated_dir

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
    if subtraction_eval:
        print("Subtraction evaluation: ENABLED")
    if save_subtractions:
        print("Save subtraction NIfTIs: ENABLED")
    if roi_eval:
        print(f"ROI vessel evaluation: ENABLED (threshold={vessel_threshold_hu} HU)")
    if fid_eval:
        print("FID evaluation: ENABLED")
    if frd_eval:
        print(f"FRD evaluation: ENABLED ({frd_num_slices} slices/volume)")
        frd_ok, frd_msg = check_frd_runtime_compatibility()
        if not frd_ok:
            print(f"  FRD preflight FAILED: {frd_msg}")
            print("  FRD evaluation DISABLED for this run.")
            frd_eval = False
        else:
            print(f"  {frd_msg}")
    if save_generated_dir:
        print(f"Save generated NIfTIs: {save_generated_dir}")
    if load_generated_dir:
        print(f"Load generated NIfTIs: {load_generated_dir}")

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
    fid_real_slices = []
    fid_fake_slices = []
    frd_real_images = []
    frd_fake_images = []

    print(f"\nStarting inference on {len(test_ds)} patients...")
    print("-" * 40)

    with torch.no_grad():
        for i, batch in enumerate(tqdm(test_loader, desc="Testing")):
            patient_id = batch["patient_id"][0]
            source_vol = batch["source"].squeeze(0)  # (1, H, W, D) in [0, 1]
            target_vol = batch["target"].squeeze(0)

            try:
                # Process full 3D volume slice-by-slice (or load from pre-generated dir)
                if load_generated_dir:
                    pred_path = os.path.join(load_generated_dir, f"{patient_id}_pred.nii.gz")
                    generated_vol = nib.load(pred_path).get_fdata().astype(np.float32)
                    gt_vol = target_vol.squeeze(0).cpu().numpy()
                    input_vol = source_vol.squeeze(0).cpu().numpy()
                    vol_psnr = psnr(gt_vol, generated_vol, data_range=1.0)
                    vol_ssim = ssim_3d(gt_vol, generated_vol, data_range=1.0)
                    vol_mae = mae(gt_vol, generated_vol)
                    vol_rmse = rmse(gt_vol, generated_vol)
                    metrics_dict = {"psnr": vol_psnr, "ssim": vol_ssim, "mae": vol_mae, "rmse": vol_rmse}
                    src_dhw = np.transpose(input_vol * 2.0 - 1.0, (2, 0, 1))
                    gen_dhw = np.transpose(generated_vol * 2.0 - 1.0, (2, 0, 1))
                    gt_dhw  = np.transpose(gt_vol * 2.0 - 1.0, (2, 0, 1))
                else:
                    generated_vol, gt_vol, input_vol, metrics_dict, src_dhw, gen_dhw, gt_dhw = test_full_volume(
                        model, source_vol, target_vol, opt, patient_id, output_dir
                    )
                    if save_generated_dir:
                        os.makedirs(save_generated_dir, exist_ok=True)
                        nib.save(
                            nib.Nifti1Image(generated_vol.astype(np.float32), np.eye(4)),
                            os.path.join(save_generated_dir, f"{patient_id}_pred.nii.gz"),
                        )

                # ---- Subtraction evaluation: (source - generated) vs (source - ground_truth) ----
                sub_metrics = {}
                if subtraction_eval:
                    # Compute subtractions in [0, 1] space (H, W, D)
                    synth_sub = input_vol - generated_vol
                    gt_sub = input_vol - gt_vol
                    sub_metrics = {
                        "sub_psnr": psnr(gt_sub, synth_sub, data_range=1.0),
                        "sub_ssim": ssim_3d(gt_sub, synth_sub, data_range=1.0),
                        "sub_mae": mae(gt_sub, synth_sub),
                        "sub_rmse": rmse(gt_sub, synth_sub),
                    }

                    # Subtraction visualization (uses [-1, 1] (D, H, W) volumes)
                    viz_dir = os.path.join(output_dir, "visualizations")
                    sub_vis_path = os.path.join(viz_dir, f"{patient_id}_subtractions.png")
                    visualize_subtractions(
                        source=src_dhw,
                        pred=gen_dhw,
                        target=gt_dhw,
                        patient_id=patient_id,
                        output_path=sub_vis_path,
                        metrics=sub_metrics,
                    )

                # ---- Save subtraction NIfTIs (denormalized to HU) ----
                if save_subtractions:
                    sub_dir = os.path.join(output_dir, "subtractions")
                    os.makedirs(sub_dir, exist_ok=True)
                    # Denormalize: [0, 1] → [-1000, 1000] HU via (val * 2 - 1) * 1000, then subtract
                    hu_scale = 1000.0
                    synth_sub_hu = (input_vol - generated_vol) * 2.0 * hu_scale
                    gt_sub_hu = (input_vol - gt_vol) * 2.0 * hu_scale
                    nib.save(
                        nib.Nifti1Image(synth_sub_hu.astype(np.float32), np.eye(4)),
                        os.path.join(sub_dir, f"{patient_id}_subtraction_synthesized.nii.gz"),
                    )
                    nib.save(
                        nib.Nifti1Image(gt_sub_hu.astype(np.float32), np.eye(4)),
                        os.path.join(sub_dir, f"{patient_id}_subtraction_gt.nii.gz"),
                    )

                # ---- ROI vessel metrics (operates in [-1, 1] (D, H, W) space) ----
                roi_metrics = {}
                if roi_eval:
                    # src_dhw, gen_dhw, gt_dhw are all (D, H, W) in [-1, 1]
                    pred_sub_roi = src_dhw - gen_dhw
                    gt_sub_roi = src_dhw - gt_dhw
                    vessel_mask = create_vessel_mask(gt_sub_roi, threshold_hu=vessel_threshold_hu)
                    roi_metrics = compute_roi_metrics(pred_sub_roi, gt_sub_roi, vessel_mask)

                # ---- FID: collect centre axial slices in [-1, 1] (D, H, W) ----
                if fid_eval:
                    fid_real_slices.append(extract_centre_axial_slice_uint8(gt_dhw))
                    fid_fake_slices.append(extract_centre_axial_slice_uint8(gen_dhw))

                # ---- FRD: collect slices from (H, W, D) volumes in [0, 1] ----
                if frd_eval:
                    frd_real_images.extend(extract_slices_for_radiomics(gt_vol, frd_num_slices))
                    frd_fake_images.extend(extract_slices_for_radiomics(generated_vol, frd_num_slices))

                results.append({
                    "patient_id": patient_id,
                    **metrics_dict,
                    **sub_metrics,
                    **roi_metrics,
                })

                msg = (f"  {patient_id}: PSNR={metrics_dict['psnr']:.2f} dB | "
                       f"SSIM={metrics_dict['ssim']:.4f} | "
                       f"MAE={metrics_dict['mae']:.4f} | "
                       f"RMSE={metrics_dict['rmse']:.4f}")
                if sub_metrics:
                    msg += f"  | SubPSNR={sub_metrics['sub_psnr']:.2f}, SubSSIM={sub_metrics['sub_ssim']:.4f}"
                if roi_metrics:
                    msg += (f"  | Dice={roi_metrics['roi_dice']:.4f}"
                            f" CNR(GT)={roi_metrics['roi_cnr_gt']:.2f}"
                            f" CNR(Pred)={roi_metrics['roi_cnr_pred']:.2f}")
                print(msg)

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
                error_entry = {
                    "patient_id": patient_id,
                    "psnr": float('nan'),
                    "ssim": float('nan'),
                    "mae": float('nan'),
                    "rmse": float('nan'),
                }
                if subtraction_eval:
                    error_entry.update({
                        "sub_psnr": float('nan'),
                        "sub_ssim": float('nan'),
                        "sub_mae": float('nan'),
                        "sub_rmse": float('nan'),
                    })
                if roi_eval:
                    error_entry.update({
                        "roi_dice": float('nan'),
                        "roi_cnr_gt": float('nan'),
                        "roi_cnr_pred": float('nan'),
                        "roi_mae": float('nan'),
                        "roi_mae_hu": float('nan'),
                        "roi_psnr": float('nan'),
                    })
                results.append(error_entry)
                continue

    # ---- Compute 2D FID after full test set ----
    fid_score = None
    if fid_eval and fid_real_slices:
        print("\nComputing 2D FID (centre axial slices, InceptionV3)...")
        fid_score = compute_fid_2d(fid_real_slices, fid_fake_slices, opt.device)
        print(f"  FID (2D axial): {fid_score:.4f}")

    # ---- Compute FRD after full test set ----
    frd_score = None
    if frd_eval and frd_real_images:
        print(f"\nComputing FRD ({len(frd_real_images)} slices per set)...")
        frd_score = compute_frd(frd_real_images, frd_fake_images)
        if not np.isnan(frd_score):
            print(f"  FRD (radiomic): {frd_score:.6f}")
        else:
            print("  FRD: FAILED (pyradiomics unavailable or extraction error)")

    # Save metrics to CSV
    df = pd.DataFrame(results)
    df.to_csv(os.path.join(output_dir, "metrics.csv"), index=False)

    # Build and save JSON results (matching diffusion model format)
    metric_names = ["psnr", "ssim", "mae", "rmse"]
    if subtraction_eval:
        metric_names += ["sub_psnr", "sub_ssim", "sub_mae", "sub_rmse"]
    if roi_eval:
        metric_names += ["roi_dice", "roi_cnr_gt", "roi_cnr_pred", "roi_mae", "roi_mae_hu", "roi_psnr"]
    aggregate_metrics = {}
    for m in metric_names:
        vals = df[m].dropna()
        aggregate_metrics[f"{m}_mean"] = float(vals.mean())
        aggregate_metrics[f"{m}_std"] = float(vals.std())
        aggregate_metrics[f"{m}_min"] = float(vals.min())
        aggregate_metrics[f"{m}_max"] = float(vals.max())
    if fid_score is not None:
        aggregate_metrics["fid_2d"] = float(fid_score)
    if frd_score is not None and not np.isnan(frd_score):
        aggregate_metrics["frd"] = float(frd_score)

    json_results = {
        "model_name": opt.name,
        "checkpoint": f"{opt.name}/epoch_{opt.epoch}",
        "test_csv": test_csv,
        "num_samples": len(test_ds),
        "aggregate_metrics": aggregate_metrics,
        "per_patient_results": [
            {k: float(v) if isinstance(v, (np.floating, float)) else v for k, v in r.items()}
            for r in results
        ],
    }

    json_path = os.path.join(output_dir, "evaluation_results.json")
    with open(json_path, "w") as f:
        json.dump(json_results, f, indent=2)

    # Summary bar chart visualization (matching diffusion model)
    valid_results = [r for r in results if not np.isnan(r.get('psnr', float('nan')))]
    if valid_results:
        create_summary_visualization(
            valid_results,
            os.path.join(output_dir, "visualizations", "metrics_summary.png"),
        )

    # Print summary
    print("\n" + "=" * 60)
    print(f"TEST RESULTS ({len(test_ds)} Patients)")
    print("=" * 60)
    print(f"Average PSNR: {aggregate_metrics['psnr_mean']:.4f} +/- {aggregate_metrics['psnr_std']:.4f}")
    print(f"Average SSIM: {aggregate_metrics['ssim_mean']:.4f} +/- {aggregate_metrics['ssim_std']:.4f}")
    print(f"Average MAE:  {aggregate_metrics['mae_mean']:.4f} +/- {aggregate_metrics['mae_std']:.4f}")
    print(f"Average RMSE: {aggregate_metrics['rmse_mean']:.4f} +/- {aggregate_metrics['rmse_std']:.4f}")
    if subtraction_eval:
        print("-" * 60)
        print("SUBTRACTION METRICS  (source - generated) vs (source - ground_truth)")
        print("-" * 60)
        print(f"Sub PSNR:  {aggregate_metrics['sub_psnr_mean']:.2f} +/- {aggregate_metrics['sub_psnr_std']:.2f} dB")
        print(f"Sub SSIM:  {aggregate_metrics['sub_ssim_mean']:.4f} +/- {aggregate_metrics['sub_ssim_std']:.4f}")
        print(f"Sub MAE:   {aggregate_metrics['sub_mae_mean']:.4f} +/- {aggregate_metrics['sub_mae_std']:.4f}")
        print(f"Sub RMSE:  {aggregate_metrics['sub_rmse_mean']:.4f} +/- {aggregate_metrics['sub_rmse_std']:.4f}")
    if roi_eval:
        print("-" * 60)
        print(f"VESSEL ROI METRICS  (threshold = {vessel_threshold_hu} HU)")
        print("-" * 60)
        print(f"Dice:         {aggregate_metrics['roi_dice_mean']:.4f} +/- {aggregate_metrics['roi_dice_std']:.4f}")
        print(f"CNR (GT):     {aggregate_metrics['roi_cnr_gt_mean']:.2f} +/- {aggregate_metrics['roi_cnr_gt_std']:.2f}")
        print(f"CNR (Pred):   {aggregate_metrics['roi_cnr_pred_mean']:.2f} +/- {aggregate_metrics['roi_cnr_pred_std']:.2f}")
        print(f"ROI MAE:      {aggregate_metrics['roi_mae_mean']:.4f} +/- {aggregate_metrics['roi_mae_std']:.4f}")
        print(f"ROI MAE (HU): {aggregate_metrics['roi_mae_hu_mean']:.2f} +/- {aggregate_metrics['roi_mae_hu_std']:.2f} HU")
        print(f"ROI PSNR:     {aggregate_metrics['roi_psnr_mean']:.2f} +/- {aggregate_metrics['roi_psnr_std']:.2f} dB")
    if fid_score is not None:
        print("-" * 60)
        print(f"FID (2D axial): {fid_score:.4f}")
    if frd_score is not None and not np.isnan(frd_score):
        print("-" * 60)
        print(f"FRD (radiomic): {frd_score:.6f}")
    elif frd_eval:
        print("-" * 60)
        print("FRD: FAILED (see above for error details)")
    print(f"\nResults saved to: {output_dir}")
    print(f"  > Metrics CSV:  metrics.csv")
    print(f"  > Metrics JSON: evaluation_results.json")
    print(f"  > Visualizations: visualizations/ directory")
    print(f"  > Volumes:      volumes/ directory")
    if save_subtractions:
        print(f"  > Subtractions: subtractions/ directory")
    print("=" * 60)
