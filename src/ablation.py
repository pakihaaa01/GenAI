import copy
import json
import math
import sys
import time
from pathlib import Path
from typing import Any, Dict, List

# Bootstrap sys.path di level modul agar aman dijalankan via `python src/ablation.py`
ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import torch
import torch.nn.functional as F

from src.config import Config
from src.data import get_dataloader
from src.diffusion import DDPM
from src.model import DDPMUNet
from src.train import set_seed

# Dibatasi 2 epoch: satu epoch terukur 915 dtk di CPU, 3 epoch x 2 varian > 1 jam.
ABLATION_EPOCHS = 2

class CosineDDPM(DDPM):
    """Subclass DDPM dengan cosine noise schedule (Nichol & Dhariwal, 2021)."""

    def __init__(
        self,
        timesteps: int = 1000,
        s: float = 0.008,
        clip_denoised: bool = True,
    ) -> None:
        super().__init__(
            timesteps=timesteps,
            beta_start=1e-4,
            beta_end=0.02,
            clip_denoised=clip_denoised,
        )

        # Hitung alpha_bar_t dalam float64 untuk presisi numerik di dekat t = 0
        steps = timesteps + 1
        t = torch.linspace(0, timesteps, steps, dtype=torch.float64)
        f_t = torch.cos(((t / timesteps) + s) / (1.0 + s) * (math.pi / 2.0)) ** 2
        alphas_cumprod_raw = f_t / f_t[0]

        betas = 1.0 - (alphas_cumprod_raw[1:] / alphas_cumprod_raw[:-1])
        betas = torch.clamp(betas, min=1e-4, max=0.999).to(torch.float32)

        # Turunkan seluruh buffer dari betas pasca-clamp agar 100% konsisten
        alphas = 1.0 - betas
        alphas_cumprod = torch.cumprod(alphas, dim=0)
        alphas_cumprod_prev = F.pad(alphas_cumprod[:-1], (1, 0), value=1.0)

        sqrt_alphas_cumprod = torch.sqrt(alphas_cumprod)
        sqrt_one_minus_alphas_cumprod = torch.sqrt(1.0 - alphas_cumprod)
        sqrt_recip_alphas_cumprod = torch.sqrt(1.0 / alphas_cumprod)
        sqrt_recipm1_alphas_cumprod = torch.sqrt(1.0 / alphas_cumprod - 1.0)

        posterior_variance = (
            betas * (1.0 - alphas_cumprod_prev) / (1.0 - alphas_cumprod)
        )
        posterior_log_variance_clipped = torch.log(
            torch.clamp(posterior_variance, min=1e-20)
        )
        posterior_mean_coef1 = (
            betas * torch.sqrt(alphas_cumprod_prev) / (1.0 - alphas_cumprod)
        )
        posterior_mean_coef2 = (
            (1.0 - alphas_cumprod_prev) * torch.sqrt(alphas) / (1.0 - alphas_cumprod)
        )

        updated_buffers = {
            "betas": betas,
            "alphas": alphas,
            "alphas_cumprod": alphas_cumprod,
            "alphas_cumprod_prev": alphas_cumprod_prev,
            "sqrt_alphas_cumprod": sqrt_alphas_cumprod,
            "sqrt_one_minus_alphas_cumprod": sqrt_one_minus_alphas_cumprod,
            "sqrt_recip_alphas_cumprod": sqrt_recip_alphas_cumprod,
            "sqrt_recipm1_alphas_cumprod": sqrt_recipm1_alphas_cumprod,
            "posterior_variance": posterior_variance,
            "posterior_log_variance_clipped": posterior_log_variance_clipped,
            "posterior_mean_coef1": posterior_mean_coef1,
            "posterior_mean_coef2": posterior_mean_coef2,
        }

        existing_buffers = dict(self.named_buffers())
        for buf_name, buf_tensor in updated_buffers.items():
            if buf_name not in existing_buffers:
                raise KeyError(f"Buffer '{buf_name}' tidak ditemukan pada induk DDPM.")
            self.register_buffer(buf_name, buf_tensor)

def run_variant(name: str, cfg: Config, use_cosine: bool) -> Dict[str, Any]:
    """Jalankan training 3 epoch dengan seed & urutan batch identik per varian."""
    variant_cfg = copy.deepcopy(cfg)
    variant_cfg.epochs = ABLATION_EPOCHS
    variant_cfg.num_workers = 0
    variant_cfg.beta_schedule = "cosine" if use_cosine else "linear"

    # Isolasi direktori keluaran agar checkpoints/ddpm_best.pt & logs/train_history.json aman
    variant_ckpt_dir = Path(cfg.ckpt_dir) / "ablation" / name
    variant_log_dir = Path(cfg.log_dir) / "ablation"
    variant_ckpt_dir.mkdir(parents=True, exist_ok=True)
    variant_log_dir.mkdir(parents=True, exist_ok=True)

    variant_cfg.ckpt_dir = str(variant_ckpt_dir)
    variant_cfg.log_dir = str(variant_log_dir)

    # Reset seed sebelum inisialisasi dataloader, model, dan sampling noise
    set_seed(variant_cfg.seed)
    device = torch.device("cpu")

    train_loader = get_dataloader(variant_cfg, train=True)

    model = DDPMUNet(
        in_channels=variant_cfg.channels,
        base_ch=variant_cfg.unet_base,
        time_emb_dim=variant_cfg.time_embed_dim,
    ).to(device)

    if use_cosine:
        diffusion: DDPM = CosineDDPM(timesteps=variant_cfg.timesteps).to(device)
    else:
        diffusion = DDPM(
            timesteps=variant_cfg.timesteps,
            beta_start=variant_cfg.beta_start,
            beta_end=variant_cfg.beta_end,
        ).to(device)

    num_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    optimizer = torch.optim.Adam(model.parameters(), lr=variant_cfg.lr)

    epoch_losses: List[float] = []
    epoch_times: List[float] = []
    best_loss = float("inf")

    print(f"\n[Ablation] Menjalankan varian '{name}' ({variant_cfg.beta_schedule})...")

    for epoch in range(1, variant_cfg.epochs + 1):
        model.train()
        t_start = time.perf_counter()
        running_loss = 0.0
        total_samples = 0

        for batch in train_loader:
            x0 = batch[0] if isinstance(batch, (list, tuple)) else batch
            x0 = x0.to(device)
            batch_size = x0.shape[0]

            t = torch.randint(
                0, variant_cfg.timesteps, (batch_size,), device=device
            ).long()

            optimizer.zero_grad(set_to_none=True)
            loss = diffusion.p_losses(model, x0, t)
            loss.backward()
            optimizer.step()

            running_loss += loss.item() * batch_size
            total_samples += batch_size

        elapsed = time.perf_counter() - t_start
        avg_loss = running_loss / max(total_samples, 1)

        epoch_losses.append(avg_loss)
        epoch_times.append(elapsed)

        ckpt_payload = {
            "epoch": epoch,
            "variant": name,
            "beta_schedule": variant_cfg.beta_schedule,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "loss": avg_loss,
        }
        torch.save(ckpt_payload, variant_ckpt_dir / f"ddpm_epoch_{epoch:03d}.pt")

        if avg_loss < best_loss:
            best_loss = avg_loss
            torch.save(ckpt_payload, variant_ckpt_dir / "ddpm_best.pt")

        print(
            f"  Epoch {epoch}/{variant_cfg.epochs} | "
            f"Loss: {avg_loss:.5f} | Waktu: {elapsed:.2f} dtk"
        )

    log_path = variant_log_dir / f"{name}_history.json"
    history_payload = {
        "variant": name,
        "beta_schedule": variant_cfg.beta_schedule,
        "seed": variant_cfg.seed,
        "timesteps": variant_cfg.timesteps,
        "unet_base": variant_cfg.unet_base,
        "num_params": num_params,
        "epoch_losses": epoch_losses,
        "epoch_times_sec": epoch_times,
        "total_time_sec": sum(epoch_times),
    }
    with open(log_path, "w", encoding="utf-8") as f:
        json.dump(history_payload, f, indent=2)

    return history_payload

def plot_ablation(results: List[Dict[str, Any]], save_path: Path) -> None:
    """Plot kurva training loss kedua varian dalam satu gambar PNG."""
    save_path = Path(save_path)
    save_path.parent.mkdir(parents=True, exist_ok=True)

    fig, ax = plt.subplots(figsize=(8.5, 5.0), dpi=150)

    styles = {
        "linear": {"color": "#1f77b4", "marker": "o", "linestyle": "-"},
        "cosine": {"color": "#d95f02", "marker": "s", "linestyle": "--"},
    }

    for res in results:
        name = res["variant"]
        losses = res["epoch_losses"]
        epochs = list(range(1, len(losses) + 1))
        st = styles.get(name, {"color": "#2ca02c", "marker": "^", "linestyle": "-."})

        label = f"{name.capitalize()} Schedule (T={res['timesteps']})"
        ax.plot(
            epochs,
            losses,
            label=label,
            color=st["color"],
            marker=st["marker"],
            linestyle=st["linestyle"],
            linewidth=2.2,
            markersize=7,
        )

        # Anotasi nilai loss eksak pada setiap titik epoch
        offset_y = 8 if name == "linear" else -14
        for ep, val in zip(epochs, losses):
            ax.annotate(
                f"{val:.5f}",
                (ep, val),
                textcoords="offset points",
                xytext=(0, offset_y),
                ha="center",
                fontsize=8.5,
                color=st["color"],
                fontweight="bold",
            )

    ax.set_title(
        "Ablation Study: Linear vs Cosine Noise Schedule "
        f"({max(len(r['epoch_losses']) for r in results)} Epoch, Fashion-MNIST)",
        fontsize=11.5,
        fontweight="bold",
        pad=12,
    )
    ax.set_xlabel("Epoch", fontsize=10.5)
    ax.set_ylabel("Mean Squared Error (MSE) Loss", fontsize=10.5)
    ax.set_xticks(range(1, max(len(r["epoch_losses"]) for r in results) + 1))
    ax.grid(True, linestyle=":", alpha=0.6)
    ax.legend(frameon=True, fontsize=9.5)

    fig.tight_layout()
    fig.savefig(save_path, bbox_inches="tight")
    plt.close(fig)
    print(f"\n[Ablation] Grafik perbandingan disimpan ke: {save_path}")

def _print_comparison_table(results: List[Dict[str, Any]]) -> None:
    """Cetak tabel ringkasan perbandingan ke konsol."""
    n_epochs = max(len(r["epoch_losses"]) for r in results)
    ep_cols = " | ".join(f"{'Epoch ' + str(i):>9}" for i in range(1, n_epochs + 1))
    header = (
        f"{'Varian':<10} | {'Params':>9} | {ep_cols} | "
        f"{'Avg Time/Ep':>11} | {'Total Time':>10}"
    )
    sep = "-" * len(header)
    print("\n" + sep)
    print(f"HASIL ABLATION STUDY NOISE SCHEDULE ({n_epochs} EPOCH)")
    print(sep)
    print(header)
    print(sep)

    for res in results:
        losses = res["epoch_losses"]
        times = res["epoch_times_sec"]
        avg_time = sum(times) / len(times)
        loss_cells = " | ".join(f"{v:>9.5f}" for v in losses)
        print(
            f"{res['variant']:<10} | "
            f"{res['num_params']:>9,d} | "
            f"{loss_cells} | "
            f"{avg_time:>9.2f} s | "
            f"{res['total_time_sec']:>8.2f} s"
        )
    print(sep)

def main() -> None:
    cfg = Config()
    cfg.epochs = ABLATION_EPOCHS
    cfg.num_workers = 0

    linear_res = run_variant(name="linear", cfg=cfg, use_cosine=False)
    cosine_res = run_variant(name="cosine", cfg=cfg, use_cosine=True)

    results = [linear_res, cosine_res]
    _print_comparison_table(results)

    fig_path = Path(cfg.fig_dir) / "ablation_noise_schedule.png"
    plot_ablation(results, save_path=fig_path)

if __name__ == "__main__":
    main()
