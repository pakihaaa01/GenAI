"""Grid perbandingan visual dua varian noise schedule (linear vs cosine).

Men-sampling 6 gambar dari checkpoint linear dan 6 dari checkpoint cosine dengan
seed yang sama, lalu menyusunnya jadi PNG 2 baris x 6 kolom untuk lampiran
Bab IV.

Kode awal dari sesi vibe coding dengan Gemini, lalu diverifikasi.
"""

from pathlib import Path
import json
import sys
import time
from typing import Any, Dict

# Bootstrap sys.path agar import `from src...` berjalan saat skrip dijalankan langsung
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import torch

from src.ablation import CosineDDPM
from src.config import Config
from src.diffusion import DDPM
from src.evaluate import load_checkpoint_and_models
from src.model import DDPMUNet
from src.sample import generate_with_fixed_noise


LINEAR_CKPT_DIR = PROJECT_ROOT / "checkpoints" / "ablation" / "linear"
COSINE_CKPT_DIR = PROJECT_ROOT / "checkpoints" / "ablation" / "cosine"
COSINE_HISTORY_PATH = PROJECT_ROOT / "logs" / "ablation" / "cosine_history.json"
OUTPUT_PATH = PROJECT_ROOT / "report" / "figures" / "ablation_schedule_grid.png"

LINEAR_MMD = 0.18872
COSINE_MMD = 0.25580
NUM_SAMPLES = 6
FIXED_SEED = 1234


def _find_checkpoint_file(ckpt_dir: Path) -> Path:
    if ckpt_dir.is_file():
        return ckpt_dir
    if not ckpt_dir.exists():
        raise FileNotFoundError(f"Direktori checkpoint tidak ditemukan: {ckpt_dir}")

    preferred_names = [
        "ddpm_best.pt",
        "best.pt",
        "checkpoint_best.pt",
        "model_best.pt",
        "last.pt",
        "checkpoint_last.pt",
    ]
    for name in preferred_names:
        candidate = ckpt_dir / name
        if candidate.exists():
            return candidate

    pt_files = sorted(ckpt_dir.glob("*.pt"))
    if not pt_files:
        raise FileNotFoundError(f"Tidak ada file .pt di dalam {ckpt_dir}")
    return pt_files[-1]


def _normalize_to_01(images: torch.Tensor) -> torch.Tensor:
    """Konversi tensor gambar ke rentang [0.0, 1.0] secara aman untuk visualisasi."""
    img = images.detach().cpu().float()
    if img.ndim == 4 and img.shape[1] > 1:
        img = img.mean(dim=1, keepdim=True)

    if img.min() < -0.05:
        img = (img + 1.0) * 0.5
    elif img.max() > 1.0:
        img = img / 255.0

    return torch.clamp(img, 0.0, 1.0)


def _load_cosine_metadata(history_path: Path, ckpt: Dict[str, Any]) -> Dict[str, Any]:
    """Ambil metadata dari cosine_history.json, lalu fallback ke Config() default."""
    cfg_default = Config()
    meta: Dict[str, Any] = {
        "variant": ckpt.get("variant", "cosine"),
        "beta_schedule": ckpt.get("beta_schedule", "cosine"),
        "seed": FIXED_SEED,
        "channels": cfg_default.channels,
        "image_size": cfg_default.image_size,
        "unet_base": cfg_default.unet_base,
        "time_embed_dim": cfg_default.time_embed_dim,
        "timesteps": cfg_default.timesteps,
        "beta_start": cfg_default.beta_start,
        "beta_end": cfg_default.beta_end,
    }

    if history_path.exists():
        with history_path.open("r", encoding="utf-8") as f:
            history_data = json.load(f)

        # Jika file JSON berupa dict metadata langsung
        if isinstance(history_data, dict):
            for key in ("variant", "beta_schedule", "seed", "channels", "image_size", "unet_base", "time_embed_dim", "timesteps", "beta_start", "beta_end"):
                if key in history_data and history_data[key] is not None:
                    meta[key] = history_data[key]
            # Jika metadata bersarang di dalam key 'config' atau 'meta'
            for nested_key in ("config", "meta", "metadata"):
                nested = history_data.get(nested_key)
                if isinstance(nested, dict):
                    for key in ("variant", "beta_schedule", "seed", "channels", "image_size", "unet_base", "time_embed_dim", "timesteps", "beta_start", "beta_end"):
                        if key in nested and nested[key] is not None:
                            meta[key] = nested[key]

    return meta


def _build_unet_from_meta(meta: Dict[str, Any], device: torch.device) -> DDPMUNet:
    model = DDPMUNet(
        in_channels=meta["channels"],
        base_ch=meta["unet_base"],
        time_emb_dim=meta["time_embed_dim"],
    )
    return model.to(device)


def _build_cosine_diffusion(meta: Dict[str, Any], device: torch.device) -> CosineDDPM:
    diffusion = CosineDDPM(
        timesteps=meta["timesteps"],
        s=0.008,
        clip_denoised=True,
    )
    return diffusion.to(device)


def _run_fixed_noise_sampling(
    model: torch.nn.Module,
    diffusion: DDPM,
    img_size: int,
    channels: int,
    device: torch.device,
    num_samples: int,
    fixed_seed: int,
) -> torch.Tensor:
    model.eval()
    torch.manual_seed(fixed_seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(fixed_seed)

    with torch.no_grad():
        samples = generate_with_fixed_noise(
            model=model,
            diffusion=diffusion,
            num_samples=num_samples,
            img_size=img_size,
            channels=channels,
            device=device,
            fixed_seed=fixed_seed,
        )

    if isinstance(samples, tuple):
        samples = samples[0]

    return _normalize_to_01(samples)[:num_samples]


def _sample_linear_checkpoint(
    ckpt_path: Path,
    device: torch.device,
    num_samples: int,
    fixed_seed: int,
) -> torch.Tensor:
    """Sampling model linear menggunakan helper bawaan load_checkpoint_and_models."""
    cfg, model, diffusion, _ = load_checkpoint_and_models(str(ckpt_path), device=device)
    model.to(device)
    diffusion.to(device)

    return _run_fixed_noise_sampling(
        model=model,
        diffusion=diffusion,
        img_size=cfg.image_size,
        channels=cfg.channels,
        device=device,
        num_samples=num_samples,
        fixed_seed=fixed_seed,
    )


def _sample_cosine_checkpoint(
    ckpt_path: Path,
    history_path: Path,
    device: torch.device,
    num_samples: int,
    fixed_seed: int,
) -> torch.Tensor:
    """Bangun ulang DDPMUNet + CosineDDPM dari checkpoint .pt + metadata cosine."""
    ckpt = torch.load(str(ckpt_path), map_location=device)
    meta = _load_cosine_metadata(history_path=history_path, ckpt=ckpt)

    model = _build_unet_from_meta(meta=meta, device=device)
    model.load_state_dict(ckpt["model_state_dict"])

    diffusion = _build_cosine_diffusion(meta=meta, device=device)

    return _run_fixed_noise_sampling(
        model=model,
        diffusion=diffusion,
        img_size=meta["image_size"],
        channels=meta["channels"],
        device=device,
        num_samples=num_samples,
        fixed_seed=fixed_seed,
    )


def build_ablation_grid(
    linear_samples: torch.Tensor,
    cosine_samples: torch.Tensor,
    save_path: Path,
) -> None:
    save_path.parent.mkdir(parents=True, exist_ok=True)

    rows = [
        ("Linear", LINEAR_MMD, linear_samples),
        ("Cosine", COSINE_MMD, cosine_samples),
    ]

    fig, axes = plt.subplots(
        nrows=2,
        ncols=NUM_SAMPLES,
        figsize=(NUM_SAMPLES * 1.8, 4.1),
        gridspec_kw={"wspace": 0.06, "hspace": 0.28},
    )

    for row_idx, (schedule_name, mmd_val, imgs) in enumerate(rows):
        row_label = f"{schedule_name} | MMD: {mmd_val:.5f}"
        for col_idx in range(NUM_SAMPLES):
            ax = axes[row_idx, col_idx]
            img_2d = imgs[col_idx].squeeze().numpy()
            ax.imshow(img_2d, cmap="gray", vmin=0.0, vmax=1.0, interpolation="nearest")
            ax.set_xticks([])
            ax.set_yticks([])

            if col_idx == 0:
                ax.text(
                    0.0,
                    1.04,
                    row_label,
                    transform=ax.transAxes,
                    ha="left",
                    va="bottom",
                    fontsize=9,
                    fontweight="bold",
                )

    fig.subplots_adjust(left=0.02, right=0.98, top=0.90, bottom=0.03, wspace=0.06, hspace=0.30)
    fig.savefig(save_path, dpi=200)
    plt.close(fig)


def main() -> None:
    t_start = time.perf_counter()
    device = torch.device("cpu")

    linear_ckpt = _find_checkpoint_file(LINEAR_CKPT_DIR)
    cosine_ckpt = _find_checkpoint_file(COSINE_CKPT_DIR)

    t_lin_0 = time.perf_counter()
    linear_samples = _sample_linear_checkpoint(
        ckpt_path=linear_ckpt,
        device=device,
        num_samples=NUM_SAMPLES,
        fixed_seed=FIXED_SEED,
    )
    t_lin_elapsed = time.perf_counter() - t_lin_0

    t_cos_0 = time.perf_counter()
    cosine_samples = _sample_cosine_checkpoint(
        ckpt_path=cosine_ckpt,
        history_path=COSINE_HISTORY_PATH,
        device=device,
        num_samples=NUM_SAMPLES,
        fixed_seed=FIXED_SEED,
    )
    t_cos_elapsed = time.perf_counter() - t_cos_0

    t_plot_0 = time.perf_counter()
    build_ablation_grid(linear_samples, cosine_samples, OUTPUT_PATH)
    t_plot_elapsed = time.perf_counter() - t_plot_0

    total_elapsed = time.perf_counter() - t_start
    print("=== Ringkasan Waktu Eksekusi (src/figures_ablation.py) ===")
    print(f"Checkpoint Linear : {linear_ckpt}")
    print(f"Checkpoint Cosine : {cosine_ckpt}")
    print(f"History Cosine    : {COSINE_HISTORY_PATH}")
    print(f"Sampling Linear   : {t_lin_elapsed:.2f} s")
    print(f"Sampling Cosine   : {t_cos_elapsed:.2f} s")
    print(f"Render & Simpan   : {t_plot_elapsed:.2f} s")
    print(f"Total Waktu       : {total_elapsed:.2f} s")
    print(f"Output PNG        : {OUTPUT_PATH}")


if __name__ == "__main__":
    main()