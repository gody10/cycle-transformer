"""Training script for CyTran baseline with validation checkpoints.

Uses ColteaSliceDataset with diffusion-matched HU windowing [-1000, 1000].
Includes validation loop with L1 + PSNR tracking and best-model saving.
"""

import time
import logging
import os
import torch
import torch.nn as nn
import numpy as np
from tqdm import tqdm
from torch.utils.data import DataLoader

from options.train_options import TrainOptions
from models import create_model
from util.visualizer import Visualizer

from dataset import ColteaSliceDataset


# Defaults are now defined in options/base_options.py (--train_csv, --val_csv, --csv_column)


def setup_logging(opt):
    """Setup logging to file and console with detailed tracking."""
    log_dir = os.path.join(opt.checkpoints_dir, opt.name)
    os.makedirs(log_dir, exist_ok=True)
    log_file = os.path.join(log_dir, 'train_log.txt')

    # Clear any existing handlers to avoid duplicate logs
    for handler in logging.root.handlers[:]:
        logging.root.removeHandler(handler)

    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(log_file, mode='a'),  # Append mode for resume
            logging.StreamHandler()
        ]
    )

    logger = logging.getLogger(__name__)

    # Log configuration at start
    logger.info("=" * 60)
    logger.info("TRAINING SESSION STARTED")
    logger.info("=" * 60)
    logger.info(f"Log file: {log_file}")
    logger.info(f"Checkpoints dir: {opt.checkpoints_dir}/{opt.name}")

    # Log key training parameters
    logger.info("-" * 40)
    logger.info("Training Configuration:")
    logger.info(f"  Model: {opt.model}")
    logger.info(f"  Batch size: {opt.batch_size}")
    logger.info(f"  Learning rate: {opt.lr}")
    logger.info(f"  Epochs: {opt.n_epochs} + {opt.n_epochs_decay} decay")
    logger.info(f"  Input channels: {opt.input_nc}")
    logger.info(f"  Output channels: {opt.output_nc}")
    logger.info(f"  GAN mode: {opt.gan_mode}")
    logger.info("-" * 40)

    return logger


def compute_val_metrics(fake, real):
    """
    Computes metrics between generated and real images.
    Input tensors should be in range [-1, 1].
    """
    # 1. L1 Loss (MAE) - Good for optimization tracking
    l1_loss = nn.L1Loss()(fake, real)

    # 2. PSNR - Good for medical image quality
    # Convert to [0, 1] for PSNR calculation
    fake_norm = (fake + 1) / 2.0
    real_norm = (real + 1) / 2.0
    mse = nn.MSELoss()(fake_norm, real_norm)
    if mse == 0:
        psnr = 100
    else:
        psnr = 20 * torch.log10(1.0 / torch.sqrt(mse))

    return l1_loss, psnr


if __name__ == "__main__":
    opt = TrainOptions().parse()

    # Detect device
    if torch.cuda.is_available():
        opt.device = torch.device("cuda:0")
    else:
        opt.device = torch.device("cpu")

    # Setup logging
    logger = setup_logging(opt)

    # CSV paths and data root from CLI (defaults in base_options.py)
    train_csv = opt.train_csv
    train_col = opt.csv_column
    val_csv = opt.val_csv
    val_col = opt.csv_column
    data_root = opt.dataroot

    patience = getattr(opt, 'patience', 50)
    use_validation = getattr(opt, 'use_validation', False)

    # Create slice-based datasets with diffusion-matched normalization
    logger.info("Loading training volumes and building slice index ...")
    train_ds = ColteaSliceDataset(train_csv, train_col, data_root)
    train_loader = DataLoader(
        train_ds,
        batch_size=opt.batch_size,
        shuffle=True,
        num_workers=int(opt.num_threads),
        pin_memory=True
    )

    dataset_size = len(train_ds)
    logger.info(f"The number of training slices = {dataset_size}")

    # Create validation loader
    val_loader = None
    if use_validation and os.path.exists(val_csv):
        logger.info("Loading validation volumes ...")
        val_ds = ColteaSliceDataset(val_csv, val_col, data_root)
        val_loader = DataLoader(
            val_ds,
            batch_size=opt.batch_size,
            shuffle=False,
            num_workers=max(1, int(opt.num_threads) // 2),
            pin_memory=True
        )
        logger.info(f"The number of validation slices = {len(val_ds)}")

    model = create_model(opt)
    model.setup(opt)
    visualizer = Visualizer(opt)

    total_iters = 0
    best_val_loss = float('inf')
    early_stopping_counter = 0

    start_msg = f"Starting Training on {opt.device} | Model: {opt.model}"
    logger.info(start_msg)
    if use_validation:
        logger.info(f"Validation enabled with patience={patience}")

    total_epochs = opt.n_epochs + opt.n_epochs_decay

    for epoch in range(opt.epoch_count, total_epochs + 1):
        epoch_start_time = time.time()
        epoch_iter = 0
        epoch_loss = 0.0
        num_batches = 0

        visualizer.reset()

        # --- TRAINING LOOP ---
        progress_bar = tqdm(train_loader, desc=f"Epoch {epoch}/{total_epochs} [Train]")

        for batch in progress_bar:
            iter_start_time = time.time()

            data = batch  # ColteaSliceDataset returns CyTran-ready format
            total_iters += opt.batch_size
            epoch_iter += opt.batch_size

            model.set_input(data)
            model.optimize_parameters()

            # Track epoch loss
            losses = model.get_current_losses()
            batch_loss = sum(losses.values())
            epoch_loss += batch_loss
            num_batches += 1

            progress_bar.set_postfix(loss=f"{batch_loss:.4f}")

            if total_iters % opt.display_freq == 0:
                save_result = total_iters % opt.update_html_freq == 0
                model.compute_visuals()
                visualizer.display_current_results(model.get_current_visuals(), epoch, save_result)

            if total_iters % opt.print_freq == 0:
                t_comp = (time.time() - iter_start_time) / opt.batch_size
                visualizer.print_current_losses(epoch, epoch_iter, losses, t_comp, 0)
                if opt.display_id > 0:
                    visualizer.plot_current_losses(epoch, float(epoch_iter) / dataset_size, losses)

            if total_iters % opt.save_latest_freq == 0:
                print(f"saving the latest model (epoch {epoch}, total_iters {total_iters})")
                save_suffix = f"iter_{total_iters}" if opt.save_by_iter else "latest"
                model.save_networks(save_suffix)

        # Average Training Loss
        avg_train_loss = epoch_loss / num_batches if num_batches > 0 else 0
        current_lr = model.optimizers[0].param_groups[0]['lr'] if model.optimizers else opt.lr

        # Log Training Stats
        epoch_losses = model.get_current_losses()
        loss_str = " | ".join([f"{k}: {v:.4f}" for k, v in epoch_losses.items()])
        logger.info(f"Epoch {epoch} Losses: {loss_str}")
        logger.info(f"Epoch {epoch} - Avg Train Loss: {avg_train_loss:.6f} | LR: {current_lr:.6f}")

        # --- VALIDATION LOOP ---
        if val_loader is not None:
            model.eval()
            val_l1_loss = 0.0
            val_psnr = 0.0
            val_batches = 0

            with torch.no_grad():
                for batch in tqdm(val_loader, desc=f"Epoch {epoch} [Valid]", leave=False):
                    model.set_input(batch)  # Already in CyTran format
                    model.test()  # Generates fake_B (A->B)

                    # Get images
                    visuals = model.get_current_visuals()
                    fake_B = visuals['fake_B']  # Generated Native
                    real_B = visuals['real_B']  # Real Native

                    # Compute metrics
                    l1, psnr = compute_val_metrics(fake_B, real_B)

                    val_l1_loss += l1.item()
                    val_psnr += psnr.item()
                    val_batches += 1

            # Switch back to train mode
            for name in model.model_names:
                if isinstance(name, str):
                    getattr(model, "net" + name).train()

            # Compute Averages
            avg_val_l1 = val_l1_loss / val_batches if val_batches > 0 else 0
            avg_val_psnr = val_psnr / val_batches if val_batches > 0 else 0

            logger.info(f"Epoch {epoch} - Val L1 Loss: {avg_val_l1:.6f} | Val PSNR: {avg_val_psnr:.2f} dB")

            # Save Best Model Logic (Minimizing L1 Loss)
            if avg_val_l1 < best_val_loss:
                improvement = best_val_loss - avg_val_l1
                best_val_loss = avg_val_l1
                early_stopping_counter = 0

                model.save_networks('best')
                logger.info(f" -> New Best Model Saved! (L1: {best_val_loss:.6f})")
            else:
                early_stopping_counter += 1
                logger.info(f" -> No improvement ({early_stopping_counter}/{patience})")

            # Early Stopping
            if early_stopping_counter >= patience:
                logger.info(f"Early stopping triggered after {epoch} epochs.")
                break

        # Update Learning Rate
        model.update_learning_rate()

        # Regular Epoch Saving
        if epoch % opt.save_epoch_freq == 0:
            print(f"saving the model at the end of epoch {epoch}, iters {total_iters}")
            model.save_networks("latest")
            model.save_networks(epoch)

        epoch_time = time.time() - epoch_start_time
        logger.info(f"End of epoch {epoch} / {total_epochs} \t Time Taken: {epoch_time:.0f} sec")

    model.save_networks('final')
    logger.info("Training completed. Final model saved.")
