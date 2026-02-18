#!/usr/bin/env python3
"""
Honest 3D Inference for CyTran Baseline.

Takes a reserved test set of 3D NIfTI volumes, breaks each into 2D axial
slices, runs CyTran inference on each slice *independently*, and stitches
them back into a 3D volume.

Constraint: NO post-processing smoothing (no Gaussian blur, no 3D consistency
checks).  The raw slice-to-slice output is saved as-is so that any striping
artifacts or Z-inconsistencies are preserved honestly.

Normalization:
    Input:  NIfTI loaded and windowed to [-1000, 1000] HU -> [0, 1] -> [-1, 1]
    Output: CyTran outputs in [-1, 1], converted back to [0, 1] for saving.
    Both diffusion and CyTran outputs end up in [0, 1] for fair comparison.

Usage:
    python inference_and_stitch.py \
        --dataroot ../data/Coltea_Processed_Nifti_Registered \
        --test_csv ../data/Coltea-Lung-CT-100W/test_data.csv \
        --test_col patient_id \
        --name coltea_cytran_baseline \
        --epoch best \
        --output_dir ./results/cytran_3d_volumes \
        --input_nc 1 --output_nc 1
"""

import os
import sys
import argparse
import logging
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import nibabel as nib
from tqdm import tqdm
from torch.utils.data import DataLoader

from options.test_options import TestOptions
from models import create_model
from dataset import ColteaPairedDataset3D, _normalize_to_neg1_pos1, _denormalize_to_01

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)


# Defaults are now defined in options/base_options.py (--test_csv, --csv_column)


def tensor_to_numpy_01(tensor: torch.Tensor) -> np.ndarray:
    """Convert tensor from [-1,1] to numpy in [0,1]."""
    arr = tensor.squeeze().cpu().numpy()
    arr = (arr + 1.0) / 2.0
    return np.clip(arr, 0.0, 1.0)


@torch.no_grad()
def process_volume(
    model,
    source_vol: torch.Tensor,
    target_vol: torch.Tensor,
    device: torch.device,
) -> np.ndarray:
    """
    Process a single 3D volume slice-by-slice through CyTran.

    Args:
        model: Loaded CyTran model (in eval mode).
        source_vol: (1, H, W, D) tensor in [0, 1] from ColteaPairedDataset3D.
        target_vol: (1, H, W, D) tensor in [0, 1] (used as dummy B input).
        device: CUDA / CPU device.

    Returns:
        generated_volume: (H, W, D) numpy array in [0, 1].
    """
    C, H, W, D = source_vol.shape
    generated_slices = []

    for d in range(D):
        # Extract single 2D axial slice: (1, H, W)
        src_slice = source_vol[:, :, :, d].unsqueeze(0)  # (1, 1, H, W)
        tgt_slice = target_vol[:, :, :, d].unsqueeze(0)

        # [0, 1] -> [-1, 1] for CyTran
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
        # fake_B is the A->B translation (arterial -> native)
        gen_slice = visuals.get("fake_B", visuals.get("fake"))

        # [-1, 1] -> [0, 1]
        gen_np = tensor_to_numpy_01(gen_slice)  # (H, W)
        generated_slices.append(gen_np)

    # Stack along depth: (H, W, D)
    generated_volume = np.stack(generated_slices, axis=-1)
    return generated_volume


def main():
    # Parse CyTran test options
    opt = TestOptions().parse()
    opt.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    # Fixed test settings
    opt.num_threads = 0
    opt.batch_size = 1
    opt.serial_batches = True
    opt.no_flip = True

    # Data config
    test_csv = opt.test_csv
    test_col = opt.csv_column
    data_root = opt.dataroot

    # Output directory
    output_dir = os.path.join(
        opt.results_dir, opt.name, f"stitched_3d_epoch_{opt.epoch}"
    )
    os.makedirs(output_dir, exist_ok=True)

    logger.info("=" * 60)
    logger.info("CyTran 3D Inference & Stitch (HONEST — no post-processing)")
    logger.info("=" * 60)
    logger.info(f"Checkpoint: {opt.name}, epoch: {opt.epoch}")
    logger.info(f"Data root:  {data_root}")
    logger.info(f"Test CSV:   {test_csv}")
    logger.info(f"Output:     {output_dir}")
    logger.info(f"Device:     {opt.device}")

    # Load 3D volume dataset
    test_ds = ColteaPairedDataset3D(test_csv, test_col, data_root)
    test_loader = DataLoader(
        test_ds, batch_size=1, shuffle=False, num_workers=0, pin_memory=True
    )
    logger.info(f"Test patients: {len(test_ds)}")

    # Create and load model
    model = create_model(opt)
    model.setup(opt)
    if opt.eval:
        model.eval()

    # Process each patient
    results = []

    for i, batch in enumerate(tqdm(test_loader, desc="Inference")):
        source_vol = batch["source"].squeeze(0)  # (1, H, W, D) in [0, 1]
        target_vol = batch["target"].squeeze(0)
        patient_id = batch["patient_id"][0]

        try:
            # Slice-by-slice inference (no smoothing, no 3D consistency)
            gen_vol = process_volume(model, source_vol, target_vol, opt.device)

            # Ground truth in [0, 1]
            gt_vol = target_vol.squeeze(0).cpu().numpy()  # (H, W, D) in [0, 1]
            src_vol_np = source_vol.squeeze(0).cpu().numpy()

            # Save NIfTI volumes
            nib.save(
                nib.Nifti1Image(gen_vol.astype(np.float32), affine=np.eye(4)),
                os.path.join(output_dir, f"{patient_id}_cytran_pred.nii.gz"),
            )
            nib.save(
                nib.Nifti1Image(gt_vol.astype(np.float32), affine=np.eye(4)),
                os.path.join(output_dir, f"{patient_id}_ground_truth.nii.gz"),
            )
            nib.save(
                nib.Nifti1Image(src_vol_np.astype(np.float32), affine=np.eye(4)),
                os.path.join(output_dir, f"{patient_id}_source.nii.gz"),
            )

            results.append(
                {"patient_id": patient_id, "status": "success", "depth": gen_vol.shape[-1]}
            )
            logger.info(
                f"  {patient_id}: volume shape {gen_vol.shape}, saved."
            )

        except Exception as e:
            logger.error(f"  {patient_id}: FAILED — {e}")
            results.append({"patient_id": patient_id, "status": f"error: {e}"})

    # Save manifest
    df = pd.DataFrame(results)
    manifest_path = os.path.join(output_dir, "inference_manifest.csv")
    df.to_csv(manifest_path, index=False)

    logger.info("=" * 60)
    logger.info(f"Done. {len(df[df['status'] == 'success'])}/{len(df)} volumes processed.")
    logger.info(f"Volumes saved to: {output_dir}")
    logger.info("NOTE: No post-processing applied. Raw slice-by-slice output.")
    logger.info("=" * 60)


if __name__ == "__main__":
    main()
