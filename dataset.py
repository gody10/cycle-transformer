"""
Dataset for CyTran baseline with exact data parity to the 3D Latent Diffusion model.

Normalization Contract (matches ctpa_medvae_latent_diffusion exactly):
    - HU windowing: [-1000, 1000] (width=2000, level=0)
    - Clipped and scaled to [0, 1]
    - Then rescaled to [-1, 1] for CyTran (Tanh output range)

No additional augmentation beyond a random spatial crop in training mode.
The diffusion model uses RandSpatialCropd / RandCropByBodyMaskd with
target size (256, 256, 64); we match the same target XY (256x256) here.

Two dataset classes:
    - ColteaPairedDataset3D: Returns full 3D volumes (for inference/stitching).
    - ColteaSliceDataset:    Returns individual 2D axial slices on the fly
                             (for CyTran training and validation).
"""

import os
import logging
from typing import Optional, List, Tuple

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset
import nibabel as nib
from monai.transforms import (
    Compose,
    LoadImaged,
    EnsureChannelFirstd,
    Orientationd,
    ScaleIntensityRanged,
    SpatialPadd,
    RandSpatialCropd,
    Resized,
    DivisiblePadd,
    ToTensord,
)

logger = logging.getLogger(__name__)

# ============================================================================
# Normalization constants — MUST match the diffusion model
# (ctpa_medvae_latent_diffusion with window_width=2000, window_level=0)
# ============================================================================
MIN_HU: float = -1000.0
MAX_HU: float = 1000.0

# Spatial targets — matches the diffusion model's default
TARGET_XY: int = 256
MIN_DEPTH: int = 64


# ============================================================================
# Shared NIfTI loading transforms
# ============================================================================

def _build_load_transforms() -> Compose:
    """
    Build the NIfTI loading and intensity normalisation pipeline.

    Steps (identical to diffusion model Mode 1 with width=2000, level=0):
        1. Load NIfTI
        2. EnsureChannelFirst -> (C, H, W, D) in MONAI convention
        3. Reorient to RAS
        4. ScaleIntensityRange [-1000, 1000] -> [0, 1], clip
        5. Resize XY to 256x256 (keep Z)
        6. Pad if Z < 64
        7. DivisiblePad k=16
    """
    return Compose([
        LoadImaged(keys=["source", "target"]),
        EnsureChannelFirstd(keys=["source", "target"]),
        Orientationd(keys=["source", "target"], axcodes="RAS"),
        ScaleIntensityRanged(
            keys=["source", "target"],
            a_min=MIN_HU,
            a_max=MAX_HU,
            b_min=0.0,
            b_max=1.0,
            clip=True,
        ),
        Resized(
            keys=["source", "target"],
            spatial_size=(TARGET_XY, TARGET_XY, -1),
            mode=["trilinear", "trilinear"],
        ),
        SpatialPadd(
            keys=["source", "target"],
            spatial_size=(TARGET_XY, TARGET_XY, MIN_DEPTH),
            method="symmetric",
        ),
        DivisiblePadd(
            keys=["source", "target"],
            k=16,
            method="symmetric",
        ),
        ToTensord(keys=["source", "target"]),
    ])


def _normalize_to_neg1_pos1(tensor: torch.Tensor) -> torch.Tensor:
    """[0, 1] -> [-1, 1]."""
    return tensor * 2.0 - 1.0


def _denormalize_to_01(tensor: torch.Tensor) -> torch.Tensor:
    """[-1, 1] -> [0, 1]."""
    return (tensor + 1.0) / 2.0


# ============================================================================
# Helper: filter valid patient folders
# ============================================================================

def _collect_valid_samples(
    csv_file: str,
    col_name: str,
    root_dir: str,
) -> List[dict]:
    """Read CSV, check that both arterial.nii.gz and native.nii.gz exist."""
    df = pd.read_csv(csv_file)
    patient_ids = df[col_name].astype(str).tolist()
    valid = []
    for pid in patient_ids:
        src = os.path.join(root_dir, pid, "arterial.nii.gz")
        tgt = os.path.join(root_dir, pid, "native.nii.gz")
        if os.path.exists(src) and os.path.exists(tgt):
            valid.append({"source": src, "target": tgt, "patient_id": pid})
        else:
            logger.warning(f"Skipping {pid}: missing arterial or native NIfTI")
    logger.info(f"Found {len(valid)}/{len(patient_ids)} valid patients from {csv_file}")
    return valid


# ============================================================================
# Dataset 1 — Full 3D volume (for inference & stitching)
# ============================================================================

class ColteaPairedDataset3D(Dataset):
    """
    Returns full 3D volumes after loading, reorientation, and intensity
    normalisation.  Used by inference_and_stitch.py.

    Each sample dict:
        source: (1, H, W, D) tensor in [0, 1]
        target: (1, H, W, D) tensor in [0, 1]
        patient_id: str
    """

    def __init__(
        self,
        csv_file: str,
        col_name: str,
        root_dir: str,
    ):
        self.valid_samples = _collect_valid_samples(csv_file, col_name, root_dir)
        self.transform = _build_load_transforms()

    def __len__(self) -> int:
        return len(self.valid_samples)

    def __getitem__(self, idx: int) -> dict:
        info = self.valid_samples[idx]
        data = self.transform({"source": info["source"], "target": info["target"]})
        return {
            "source": data["source"],   # (1, H, W, D) in [0, 1]
            "target": data["target"],
            "patient_id": info["patient_id"],
        }


# ============================================================================
# Dataset 2 — 2D slice dataset (for CyTran training / validation)
# ============================================================================

class ColteaSliceDataset(Dataset):
    """
    Loads 3D volumes once, then serves individual 2D axial slices.

    At construction we:
        1. Load every volume through the shared MONAI pipeline.
        2. Build a flat index (volume_idx, slice_idx) covering all slices.

    At __getitem__ we return a single 2D slice pair in [-1, 1] with shape
    (1, H, W) — ready for CyTran's 1-channel input.

    For CycleGAN/CyTran compatibility the returned dict uses keys
    {"A", "B", "A_paths", "B_paths"} where A=source (arterial), B=target (native).
    """

    def __init__(
        self,
        csv_file: str,
        col_name: str,
        root_dir: str,
        max_patients: Optional[int] = None,
    ):
        sample_list = _collect_valid_samples(csv_file, col_name, root_dir)
        if max_patients is not None:
            sample_list = sample_list[:max_patients]

        transform = _build_load_transforms()

        self.volumes: List[Tuple[torch.Tensor, torch.Tensor, str]] = []
        self.index_map: List[Tuple[int, int]] = []  # (vol_idx, slice_idx)

        logger.info(f"Pre-loading {len(sample_list)} volumes into memory ...")
        for i, info in enumerate(sample_list):
            data = transform({"source": info["source"], "target": info["target"]})
            src = data["source"]  # (1, H, W, D) in [0, 1]
            tgt = data["target"]
            self.volumes.append((src, tgt, info["patient_id"]))
            n_slices = src.shape[-1]  # D is last dim in MONAI
            for s in range(n_slices):
                self.index_map.append((i, s))

        logger.info(
            f"ColteaSliceDataset ready: {len(self.volumes)} volumes, "
            f"{len(self.index_map)} total slices"
        )

    def __len__(self) -> int:
        return len(self.index_map)

    def __getitem__(self, idx: int) -> dict:
        vol_idx, slice_idx = self.index_map[idx]
        src_vol, tgt_vol, pid = self.volumes[vol_idx]

        # Extract 2D axial slice: (1, H, W, D) -> (1, H, W)
        src_slice = src_vol[..., slice_idx]   # (1, H, W) in [0, 1]
        tgt_slice = tgt_vol[..., slice_idx]

        # Convert [0, 1] -> [-1, 1] (CyTran Tanh range)
        src_slice = _normalize_to_neg1_pos1(src_slice)
        tgt_slice = _normalize_to_neg1_pos1(tgt_slice)

        return {
            "A": src_slice,        # (1, H, W) in [-1, 1]
            "B": tgt_slice,        # (1, H, W) in [-1, 1]
            "A_paths": pid,
            "B_paths": pid,
        }
