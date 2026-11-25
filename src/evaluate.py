from pathlib import Path
import sys

# Bootstrap sys.path di level modul agar import absolut `from src...` selalu bekerja
# saat dijalankan langsung sebagai `python src/evaluate.py`
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np
import torch
import torch.nn.functional as F
from matplotlib import pyplot as plt

from src.config import Config
from src.data import get_dataloader
from src.diffusion import DDPM
from src.model import DDPMUNet
from src.sample import generate_with_fixed_noise


def _resolve_path(path_str: str | Path) -> Path:
    p = Path(path_str)
    return p if p.is_absolute() else (PROJECT_ROOT / p)


def _extract_real_images(cfg: Config, max_samples: int) -> torch.Tensor:
    """Mengambil tensor gambar asli (test set) ternormalisasi ke [0, 1], shape (N, C, H, W)."""
    # Pastikan num_workers=0 sesuai batasan Windows
    cfg_eval = Config(**{**vars(cfg), "num_workers": 0})
    loader = get_dataloader(cfg_eval, train=False)

    chunks: list[torch.Tensor] = []
    collected = 0
    for batch in loader:
        imgs = batch[0] if isinstance(batch, (tuple, list)) else batch
        imgs = imgs.detach().cpu().float()
        # Jika dataset berasal dari ToTensor(), nilai sudah [0, 1]; jika [-1, 1], petakan ke [0, 1]
        if imgs.min() < 0.0:
            imgs = (imgs + 1.0) * 0.5
        imgs = imgs.clamp(0.0, 1.0)

        remaining = max_samples - collected
        if remaining <= 0:
            break
        if imgs.size(0) > remaining:
            imgs = imgs[:remaining]
        chunks.append(imgs)
        collected += imgs.size(0)
        if collected >= max_samples:
            break

    return torch.cat(chunks, dim=0)


def compute_rbf_mmd(x: torch.Tensor, y: torch.Tensor, sigmas: tuple[float, ...] = (0.5, 1.0, 2.0, 5.0, 10.0)) -> float:
    """Menghitung unbiased/biased RBF-MMD^2 antara dua himpunan tensor ber-shape (N, D)."""
    x_flat = x.reshape(x.size(0), -1).float()
    y_flat = y.reshape(y.size(0), -1).float()

    xx = torch.cdist(x_flat, x_flat, p=2.0).pow(2)
    yy = torch.cdist(y_flat, y_flat, p=2.0).pow(2)
    xy = torch.cdist(x_flat, y_flat, p=2.0).pow(2)

    mmd2 = 0.0
    for s in sigmas:
        bandwidth = 2.0 * (s ** 2)
        k_xx = torch.exp(-xx / bandwidth).mean()
        k_yy = torch.exp(-yy / bandwidth).mean()
        k_xy = torch.exp(-xy / bandwidth).mean()
        mmd2 += float((k_xx + k_yy - 2.0 * k_xy).item())

    return max(0.0, mmd2 / len(sigmas))


def _hist_wasserstein_1d(vals_a: np.ndarray, vals_b: np.ndarray, bins: int = 32) -> float:
    """Jarak Wasserstein-1 (CDF L1) antara distribusi nilai skalar 1D pada rentang [0, 1]."""
    hist_a, _ = np.histogram(vals_a, bins=bins, range=(0.0, 1.0), density=True)
    hist_b, _ = np.histogram(vals_b, bins=bins, range=(0.0, 1.0), density=True)
    bin_width = 1.0 / bins
    cdf_a = np.cumsum(hist_a) * bin_width
    cdf_b = np.cumsum(hist_b) * bin_width
    return float(np.sum(np.abs(cdf_a - cdf_b)) * bin_width)


def _sobel_gradient_magnitude(imgs: torch.Tensor) -> np.ndarray:
    """Menghitung magnitudo gradien Sobel ternormalisasi [0, 1] untuk tensor (N, C, H, W)."""
    c = imgs.size(1)
    kx = torch.tensor([[-1.0, 0.0, 1.0], [-2.0, 0.0, 2.0], [-1.0, 0.0, 1.0]], dtype=torch.float32).view(1, 1, 3, 3)
    ky = torch.tensor([[-1.0, -2.0, -1.0], [0.0, 0.0, 0.0], [1.0, 2.0, 1.0]], dtype=torch.float32).view(1, 1, 3, 3)
    kx = kx.repeat(c, 1, 1, 1)
    ky = ky.repeat(c, 1, 1, 1)

    gx = F.conv2d(imgs.float(), kx, padding=1, groups=c)
    gy = F.conv2d(imgs.float(), ky, padding=1, groups=c)
    mag = torch.sqrt(gx.pow(2) + gy.pow(2) + 1e-8)
    # Normalisasi per-dataset maksimum agar skala berada di [0, 1]
    denom = float(mag.max().item()) if float(mag.max().item()) > 1e-8 else 1.0
    return (mag / denom).clamp(0.0, 1.0).numpy().ravel()


def load_checkpoint_and_models(ckpt_path: Path, device: torch.device) -> tuple[Config, DDPMUNet, DDPM, dict]:
    """Memuat checkpoint, membangun Config, DDPMUNet, dan DDPM."""
    ckpt = torch.load(ckpt_path, map_location=device)
    raw_cfg = ckpt.get("config", {})
    if isinstance(raw_cfg, Config):
        cfg = raw_cfg
    elif isinstance(raw_cfg, dict):
        valid_fields = {k: v for k, v in raw_cfg.items() if hasattr(Config, k)}
        cfg = Config(**valid_fields)
    else:
        cfg = Config()

    cfg.num_workers = 0

    model = DDPMUNet(
        in_channels=cfg.channels,
        base_ch=cfg.unet_base,
        time_emb_dim=cfg.time_embed_dim,
    ).to(device)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()

    diffusion = DDPM(
        timesteps=cfg.timesteps,
        beta_start=cfg.beta_start,
        beta_end=cfg.beta_end,
        clip_denoised=True,
    ).to(device)

    return cfg, model, diffusion, ckpt


def save_evaluation_figure(
    fig_dir: Path,
    real_imgs: torch.Tensor,
    gen_imgs: torch.Tensor,
    metrics: dict[str, float],
) -> Path:
    """Menyimpan ringkasan visual (grid sampel generated + perbandingan histogram piksel & gradien)."""
    fig_dir.mkdir(parents=True, exist_ok=True)
    out_path = fig_dir / "evaluation_metrics.png"

    fig, axes = plt.subplots(1, 3, figsize=(14, 4.2))

    # Panel 1: Grid 4x4 hasil sampling
    grid_n = min(16, gen_imgs.size(0))
    side = int(np.sqrt(grid_n))
    grid_n = side * side
    canvas = np.zeros((side * gen_imgs.size(2), side * gen_imgs.size(3)), dtype=np.float32)
    for idx in range(grid_n):
        r, c = divmod(idx, side)
        patch = gen_imgs[idx, 0].numpy()
        h, w = patch.shape
        canvas[r * h : (r + 1) * h, c * w : (c + 1) * w] = patch

    axes[0].imshow(canvas, cmap="gray", vmin=0.0, vmax=1.0)
    axes[0].set_title(f"Generated Samples ({side}x{side})")
    axes[0].axis("off")

    # Panel 2: Histogram intensitas piksel (Real vs Generated)
    real_px = real_imgs.numpy().ravel()
    gen_px = gen_imgs.numpy().ravel()
    axes[1].hist(real_px, bins=32, range=(0, 1), density=True, alpha=0.6, label="Real (Test)")
    axes[1].hist(gen_px, bins=32, range=(0, 1), density=True, alpha=0.6, label="Generated")
    axes[1].set_title(f"Pixel Intensity (W1={metrics['pixel_w1']:.4f})")
    axes[1].set_xlabel("Intensity [0, 1]")
    axes[1].set_ylabel("Density")
    axes[1].legend()

    # Panel 3: Histogram gradien Sobel (Real vs Generated)
    real_grad = _sobel_gradient_magnitude(real_imgs)
    gen_grad = _sobel_gradient_magnitude(gen_imgs)
    axes[2].hist(real_grad, bins=32, range=(0, 1), density=True, alpha=0.6, label="Real (Test)")
    axes[2].hist(gen_grad, bins=32, range=(0, 1), density=True, alpha=0.6, label="Generated")
    axes[2].set_title(f"Sobel Edge Mag (W1={metrics['grad_w1']:.4f})")
    axes[2].set_xlabel("Normalized Gradient")
    axes[2].legend()

    fig.suptitle(
        f"Mini DDPM Eval | RBF-MMD (Real vs Gen): {metrics['mmd_real_gen']:.4f} "
        f"(Real Split Ref: {metrics['mmd_real_split']:.4f})",
        fontsize=10,
    )
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    return out_path


def evaluate(
    ckpt_rel_path: str = "checkpoints/ddpm_best.pt",
    num_eval_samples: int = 256,
    seed: int = 1234,
) -> dict[str, float]:
    device = torch.device("cpu")
    ckpt_path = _resolve_path(ckpt_rel_path)
    cfg, model, diffusion, ckpt = load_checkpoint_and_models(ckpt_path, device)

    # Ambil 2x sampel test untuk membandingkan Real-vs-Generated dan baseline Real-vs-Real
    real_all = _extract_real_images(cfg, max_samples=num_eval_samples * 2)
    half = real_all.size(0) // 2
    real_a = real_all[:half]
    real_b = real_all[half : half * 2]

    gen_imgs = generate_with_fixed_noise(
        model=model,
        diffusion=diffusion,
        num_samples=half,
        img_size=cfg.image_size,
        channels=cfg.channels,
        device=device,
        fixed_seed=seed,
    ).detach().cpu().float().clamp(0.0, 1.0)

    mmd_real_gen = compute_rbf_mmd(real_a, gen_imgs)
    mmd_real_split = compute_rbf_mmd(real_a, real_b)

    pixel_w1 = _hist_wasserstein_1d(real_a.numpy().ravel(), gen_imgs.numpy().ravel(), bins=32)
    pixel_w1_ref = _hist_wasserstein_1d(real_a.numpy().ravel(), real_b.numpy().ravel(), bins=32)

    grad_w1 = _hist_wasserstein_1d(_sobel_gradient_magnitude(real_a), _sobel_gradient_magnitude(gen_imgs), bins=32)
    grad_w1_ref = _hist_wasserstein_1d(_sobel_gradient_magnitude(real_a), _sobel_gradient_magnitude(real_b), bins=32)

    metrics = {
        "mmd_real_gen": mmd_real_gen,
        "mmd_real_split": mmd_real_split,
        "pixel_w1": pixel_w1,
        "pixel_w1_ref": pixel_w1_ref,
        "grad_w1": grad_w1,
        "grad_w1_ref": grad_w1_ref,
    }

    fig_dir = _resolve_path(cfg.fig_dir)
    saved_fig = save_evaluation_figure(fig_dir, real_a, gen_imgs, metrics)

    ckpt_epoch = ckpt.get("epoch", "N/A")
    ckpt_loss = ckpt.get("loss", "N/A")

    print("=== Mini DDPM Evaluation Summary (CPU) ===")
    print(f"Checkpoint      : {ckpt_path} (epoch={ckpt_epoch}, loss={ckpt_loss})")
    print(f"Eval Samples    : {half} per set (image_size={cfg.image_size}, channels={cfg.channels})")
    print(f"RBF-MMD (Real vs Gen)   : {mmd_real_gen:.6f}  [Real Split Ref: {mmd_real_split:.6f}]")
    print(f"Pixel W1 (Real vs Gen)  : {pixel_w1:.6f}  [Real Split Ref: {pixel_w1_ref:.6f}]")
    print(f"Sobel Grad W1 (R vs G)  : {grad_w1:.6f}  [Real Split Ref: {grad_w1_ref:.6f}]")
    print(f"Saved figure    : {saved_fig}")

    return metrics


if __name__ == "__main__":
    evaluate()
