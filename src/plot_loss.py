"""Plot kurva loss dari logs/train_history.json → report/figures/loss_curve.png"""
import json
import os
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.config import Config


def plot_loss_curve(cfg: Config):
    hist_path = os.path.join(cfg.log_dir, "train_history.json")
    with open(hist_path) as f:
        hist = json.load(f)

    epochs = [h["epoch"] for h in hist]
    losses = [h["loss"] for h in hist]

    fig, ax = plt.subplots(figsize=(7, 4), dpi=150)
    ax.plot(epochs, losses, marker="o", markersize=3, linewidth=1.5, color="#2b6cb0")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("MSE loss (noise prediction)")
    ax.set_title(f"Kurva loss training — {len(epochs)} epoch, loss akhir {losses[-1]:.5f}")
    ax.grid(alpha=0.3)
    ax.set_xticks(range(1, len(epochs) + 1, max(1, len(epochs) // 10)))

    os.makedirs(cfg.fig_dir, exist_ok=True)
    out = os.path.join(cfg.fig_dir, "loss_curve.png")
    fig.tight_layout()
    fig.savefig(out)
    plt.close(fig)
    print(f"Kurva loss disimpan ke: {out}")
    print(f"Epoch 1: {losses[0]:.5f} → Epoch {epochs[-1]}: {losses[-1]:.5f} "
          f"({(1 - losses[-1]/losses[0])*100:.1f}% turun)")


if __name__ == "__main__":
    plot_loss_curve(Config())
