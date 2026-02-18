import os
import re
import argparse
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

def parse_log_file(log_path):
    """Parses the CyTran train_log.txt to extract loss metrics."""

    if not os.path.exists(log_path):
        print(f"Error: Log file not found at {log_path}")
        return None

    # Storage for metrics
    epochs = []
    losses = {
        "G_A": [], "G_B": [],
        "D_A": [], "D_B": [],
        "cycle_A": [], "cycle_B": [],
        "idt_A": [], "idt_B": [],
        "val_loss": []
    }

    # Regex patterns
    # Matches: "Epoch 5 Losses: D_A: 0.234 | G_A: 0.456 | ..."
    loss_pattern = re.compile(r"Epoch (\d+) Losses: (.*)")
    # Matches: "Epoch 5 - Val Loss: 0.123456" or "Epoch 5 - Val L1 Loss: 0.123456"
    val_pattern = re.compile(r"Epoch (\d+) - Val (?:L1 )?Loss: ([\d\.]+)")

    print(f"Reading log: {log_path}...")

    with open(log_path, 'r') as f:
        for line in f:
            # 1. Parse Training Losses
            match_loss = loss_pattern.search(line)
            if match_loss:
                epoch = int(match_loss.group(1))
                if epoch not in epochs:
                    epochs.append(epoch)

                # Parse the key-value pairs part
                content = match_loss.group(2)
                pairs = content.split(" | ")

                # Extract values into dictionary
                current_metrics = {}
                for pair in pairs:
                    if ":" in pair:
                        key, val = pair.split(":")
                        current_metrics[key.strip()] = float(val.strip())

                # Append to main lists (handling missing keys gracefully)
                for key in losses:
                    if key != "val_loss": # distinct handling for val
                        val = current_metrics.get(key, None)
                        if val is not None:
                            losses[key].append(val)

            # 2. Parse Validation Loss
            match_val = val_pattern.search(line)
            if match_val:
                val_loss = float(match_val.group(2))
                losses["val_loss"].append(val_loss)

    # Align lengths (Validation might be missing for some epochs)
    min_len = len(epochs)
    # Truncate lists to match epoch count just in case of partial writes
    for k in losses:
        losses[k] = losses[k][:min_len]

    return epochs, losses

def plot_training_curves(epochs, losses, save_path=None):
    """Generates a summary plot of the training session."""

    fig, axes = plt.subplots(3, 1, figsize=(12, 15), sharex=True)

    # --- Plot 1: Generator Losses (The "Performance" score) ---
    # G_A/G_B = GAN loss (fooling the discriminator)
    # Cycle = Consistency loss (reconstruction)
    if losses["G_A"]:
        axes[0].plot(epochs, losses["G_A"], label="G_A (Art->Nat)", color="tab:blue", linewidth=2)
        axes[0].plot(epochs, losses["cycle_A"], label="Cycle A (Recon)", color="tab:blue", linestyle="--", alpha=0.7)
    if losses["G_B"]:
        axes[0].plot(epochs, losses["G_B"], label="G_B (Nat->Art)", color="tab:orange", linewidth=2)
        axes[0].plot(epochs, losses["cycle_B"], label="Cycle B (Recon)", color="tab:orange", linestyle="--", alpha=0.7)

    axes[0].set_title("Generator & Cycle Consistency Losses")
    axes[0].set_ylabel("Loss")
    axes[0].legend(loc="upper right")
    axes[0].grid(True, alpha=0.3)

    # --- Plot 2: Discriminator Losses (The "Critic" score) ---
    # Should ideally hover around 0.25 - 0.5. If 0.0, it's too strong.
    if losses["D_A"]:
        axes[1].plot(epochs, losses["D_A"], label="D_A (Checks Native)", color="tab:green")
    if losses["D_B"]:
        axes[1].plot(epochs, losses["D_B"], label="D_B (Checks Arterial)", color="tab:red")

    axes[1].set_title("Discriminator Losses")
    axes[1].set_ylabel("Loss (MSE)")
    axes[1].legend(loc="upper right")
    axes[1].grid(True, alpha=0.3)

    # --- Plot 3: Validation Loss (The "Reality" check) ---
    if losses["val_loss"]:
        # Plot only indices where we actually have validation data
        val_epochs = epochs[:len(losses["val_loss"])]
        axes[2].plot(val_epochs, losses["val_loss"], label="Validation Loss", color="black", linewidth=2.5)

        # Highlight minimum
        min_val = min(losses["val_loss"])
        min_idx = losses["val_loss"].index(min_val)
        axes[2].scatter(val_epochs[min_idx], min_val, color="red", zorder=5)
        axes[2].annotate(f"Best: {min_val:.4f}", (val_epochs[min_idx], min_val),
                         xytext=(10, 10), textcoords='offset points')
    else:
        axes[2].text(0.5, 0.5, "No Validation Data Found", ha='center', transform=axes[2].transAxes)

    axes[2].set_title("Validation Loss (Total)")
    axes[2].set_xlabel("Epochs")
    axes[2].set_ylabel("Loss")
    axes[2].grid(True, alpha=0.3)

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150)
        print(f"Plot saved to: {save_path}")
    else:
        plt.show()

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--name", type=str, required=True, help="Name of the experiment (e.g., coltea_cytran_baseline)")
    parser.add_argument("--checkpoints_dir", type=str, default="./checkpoints", help="Root checkpoint directory")
    args = parser.parse_args()

    # Construct path to the log file based on your folder structure
    log_file = os.path.join(args.checkpoints_dir, args.name, "train_log.txt")
    output_img = os.path.join(args.checkpoints_dir, args.name, "loss_curves.png")

    result = parse_log_file(log_file)

    if result:
        epochs_data, losses_data = result
        if len(epochs_data) > 0:
            plot_training_curves(epochs_data, losses_data, save_path=output_img)
        else:
            print("Log file found, but no epoch data could be parsed.")
