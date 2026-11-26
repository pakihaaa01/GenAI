# Ekspor Percakapan Gemini — Sesi #6

- **Judul percakapan:** Evaluasi Mini DDPM Fashion-MNIST CPU
- **Tautan share:** https://share.gemini.google/QMM0U3W3ZDe0
- **Model:** Gemini (gemini.google.com)
- **Operator:** Mia Rosiana Santoso
- **Cara memperoleh:** dibuka langsung dari tautan share (26/11/2026), teks halaman disalin utuh

> Teks di bawah adalah isi halaman share apa adanya (tanggal dibuat/dipublikasikan tercantum di baris pertama). Pesan pengguna tampil terlipat oleh Gemini; teks prompt lengkap ada di `LOG-INTERAKSI.md`.

---

```text
Gemini
Evaluasi Mini DDPM Fashion-MNIST CPU
https://share.gemini.google/7x08wNfwu2Zw
Created with Bantuan sehari-hari November 25, 2026 at 04:04 PM
Published November 25, 2026 at 08:04 PM
You said konteks: aku punya proyek Mini DDPM from scratch untuk Fashion-MNIST 28x28 grayscale, sudah selesai …

konteks: aku punya proyek Mini DDPM from scratch untuk Fashion-MNIST 28x28 grayscale, sudah selesai training 25 epoch (loss 0.16831 -> 0.04297, model 793.953 parameter, base_ch=32). sekarang aku mau bikin src/evaluate.py yang ngasih ANGKA, bukan cuma liat gambar. jalan di CPU, Windows, python 3.13, torch 2.14.0+cpu.







yang aku bingung:




1. FID katanya standar buat evaluasi diffusion. tapi InceptionV3 maunya input 299x299 RGB, punya aku 28x28 grayscale. kalau aku resize ke 299 terus channel-nya direplikasi 3x, FID-nya masih bermakna atau malah menyesatkan? aku takut angkanya kelihatan ilmiah tapi sebenarnya ngukur hal yang salah.




2. kalau FID nggak cocok, metrik apa yang lebih jujur buat kasus ini? aku kepikiran histogram intensitas piksel antara data asli dan hasil sampling, tapi itu cukup ilmiah nggak sih buat laporan? atau ada metrik lain yang lebih defensible dan masih bisa dihitung di CPU tanpa download model besar?




3. dari angka yang keluar nanti, gimana cara nyimpulin modelnya bagus atau belum? aku butuh ambang atau pembanding, bukan cuma "angkanya kecil".







signature yang ada di repo (jangan diubah, ini verbatim):







# src/diffusion.py




class DDPM(nn.Module):




def __init__(self, timesteps: int = 1000, beta_start: float = 1e-4,




beta_end: float = 0.02, clip_denoised: bool = True) -> None




def q_sample(self, x0: torch.Tensor, t: torch.Tensor,




noise: torch.Tensor | None = None) -> torch.Tensor




def p_losses(self, model: nn.Module, x0: torch.Tensor, t: torch.Tensor,




noise: torch.Tensor | None = None) -> torch.Tensor




def p_sample_loop(self, model: nn.Module, shape: tuple[int, int, int, int],




device: torch.device | str = "cpu") -> torch.Tensor







# src/model.py




class DDPMUNet(nn.Module):




def __init__(self, in_channels: int = 1, base_ch: int = 48,




time_emb_dim: int = 192)







# src/sample.py




def generate_with_fixed_noise(model: DDPMUNet, diffusion: DDPM, num_samples: int,




img_size: int, channels: int, device: torch.device,




fixed_seed: int = 1234) -> torch.Tensor




# return sudah didenormalisasi ke [0, 1]







# src/data.py




def get_dataset(cfg, train: bool = True) -> Dataset




def get_dataloader(cfg, train: bool = True)







# isi checkpoint (src/train.py): {"epoch", "model_state_dict",




# "optimizer_state_dict", "loss", "config"} -> checkpoints/ddpm_best.pt







# field Config yang ada: data_root, subset_size, image_size=28, channels=1,




# num_workers=0, unet_base=32, time_embed_dim=128, num_groups=8, timesteps=1000,




# beta_start=1e-4, beta_end=0.02, beta_schedule="linear", batch_size=128,




# epochs=25, lr=2e-4, seed=42, sample_every=5, num_sample_images=64,




# ckpt_dir, log_dir, fig_dir="report/figures"







batasan:




- cuma bikin src/evaluate.py, jangan ubah file lain (config.py, data.py,




diffusion.py, model.py, train.py, sample.py, plot_loss.py sudah diverifikasi)




- jangan pakai library diffusion siap pakai (diffusers dll)




- import absolut from src.x import y, dan bootstrap sys.path di level modul




(bukan di dalam __main__), karena skrip dijalankan `python src/evaluate.py`




- num_workers=0 (Windows)




- keluaran: cetak minimal satu angka metrik ke konsol, simpan minimal satu




gambar ke report/figures/

1. Apakah FID (InceptionV3) Bermakna untuk Fashion-MNIST 28×28 Grayscale?

Kekhawatiranmu valid: FID berbasis InceptionV3 pada Fashion-MNIST 28×28 grayscale memang kurang ideal dan berisiko menyesatkan jika dijadikan satu-satunya acuan.

Domain mismatch: InceptionV3 dilatih di ImageNet (foto natural berwarna 299×299), bukan citra pakaian grayscale beresolusi rendah (28×28). Fitur lapisan pool3 InceptionV3 menangkap tekstur/objek foto natural, sehingga jarak distribusi fitur pada gambar baju/sepatu sederhana bisa bias atau kurang sensitif terhadap artefak khas grayscale 28×28.

Efek upsampling & replikasi channel: Resize bikubik/bilinear dari 28×28 ke 299×299 menambah interpolasi halus dan tidak menciptakan detail frekuensi tinggi baru, sementara replikasi 3 channel membuat korelasi antar-channel artifisial (R == G == B).

Biaya di CPU: Menjalankan InceptionV3 untuk ribuan sampel di CPU Windows cukup lambat dan butuh unduhan bobot pretrained.

Kalau tetap ingin memakai metrik fitur deep, gunakan classifier kecil yang dilatih langsung pada Fashion-MNIST (misalnya CNN ringan 28×28 grayscale milikmu sendiri atau feature extractor Fashion-MNIST) daripada InceptionV3.

2. Metrik Apa yang Lebih Jujur & Cepat di CPU Tanpa Model Besar?

Histogram intensitas piksel saja kurang cukup untuk laporan teknis karena hanya melihat distribusi marginal kecerahan piksel—gambar acak dengan histogram yang mirip bisa tetap terlihat berantakan secara spasial.

Kombinasi metrik berikut jauh lebih defensible secara metodologis dan cepat di CPU:

1-NN Classifier Two-Sample Test (C2ST) / Nearest-Neighbor Accuracy (atau MMD dengan RBF Kernel pada fitur tereduksi / flattened + PCA/downsampled):

MMD (Maximum Mean Discrepancy) dengan RBF kernel (di CPU cukup pakai subset misalnya 500–1.000 gambar asli vs 500–1.000 gambar generated agar cepat): mengukur jarak distribusi antar himpunan sampel secara non-parametrik tanpa asumsi Gaussian ketat seperti FID.

SWD (Sliced Wasserstein Distance) pada patch lokal (misalnya patch 7×7 atau 14×14, atau multi-scale): sangat cocok untuk citra karena membandingkan distribusi proyeksi 1D lintas banyak arah dan menangkap struktur lokal + kontras.

Histogram Intensitas Piksel + Distribusi Gradien / Edge (Sobel):

Bandingkan histogram intensitas [0, 1] (misalnya dengan Wasserstein-1 distance / Earth Mover's Distance atau KL/JS divergence dengan smoothing kecil) dan distribusi magnitudo gradien Sobel (mengukur ketajaman tepi pakaian/sepatu vs blur). Ini menjawab kekhawatiranmu soal histogram sambil menutup kelemahan "buta struktur spasial".

Train Loss & Sample Quality Sanity Check:

Loss 0.16831 -> 0.04297 (MSE prediksi noise pada 
𝜖
ϵ-prediction) menunjukkan konvergensi stabil pada arsitektur unet_base=32 (793.953 parameter), tetapi loss kecil tidak otomatis menjamin sampel bebas blur atau mode dropping—karena itu metrik berbasis sampel tetap diperlukan.

3. Cara Menyimpulkan Model Bagus atau Belum dari Angka?

Karena metrik non-FID (seperti MMD, SWD, atau jarak histogram) tidak punya ambang universal mutlak yang sama lintas dataset dan implementasi, cara paling ilmiah menyimpulkannya adalah perbandingan relatif (relative benchmarking):

Bandingkan terhadap Baseline & Upper/Lower Bound internal:

Real vs. Real (split test set A vs test set B): ini batas bawah praktis (floor)—seberapa jauh dua kelompok gambar asli Fashion-MNIST terpisah pada metrik tersebut. Model yang sangat bagus akan mendekati lantai ini.

Real vs. Pure Noise / Untrained Model: ini batas atas buruk (ceiling)—nilai saat model belum terlatih atau noise acak.

Real vs. Generated (dari checkpoints/ddpm_best.pt): model dinilai menjanjikan/layak bila jaraknya jauh lebih kecil daripada untrained/noise dan mendekati jarak Real vs. Real (misalnya rasio jarak terhadap Real vs. Real makin kecil, serta visualisasi grid & distribusi piksel/gradien selaras dengan test set).

Cek trade-off kecepatan CPU: pada CPU, p_sample_loop dengan timesteps=1000 memakan waktu lumayan lama per gambar. Gunakan jumlah sampel evaluasi yang masuk akal di CPU (misalnya num_eval_samples = 256 atau 512) dan catat waktu eksekusinya di laporan.

Implementasi src/evaluate.py

Kode di bawah ini:

Mem-bootstrap sys.path ke root proyek di level modul (bukan di dalam if __name__ == "__main__":) dan menggunakan import absolut from src.....

Memakai num_workers=0, membaca checkpoints/ddpm_best.pt, memuat DDPMUNet (in_channels=cfg.channels, base_ch=cfg.unet_base, time_emb_dim=cfg.time_embed_dim) dan DDPM (timesteps=cfg.timesteps, beta_start=cfg.beta_start, beta_end=cfg.beta_end), lalu memanggil generate_with_fixed_noise(...).

Menghitung RBF-MMD, Pixel Histogram Wasserstein-1, dan Sobel Gradient Wasserstein-1 (membandingkan Real Test vs. Generated serta Real Test Split A vs. Split B sebagai pembanding), mencetak angka ke konsol, dan menyimpan plot ringkasan ke report/figures/evaluation_metrics.png.

Python
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

Catatan Kecil Sebelum Menjalankan

Jika struktur Config di src/config.py menggunakan dataclass tanpa argumen wajib atau punya nama atribut berbeda, pastikan Config(**valid_fields) sesuai dengan konstruktor aslinya (kode di atas sudah memfilter atribut dengan hasattr(Config, k) agar aman).

Jika num_eval_samples=256 di CPU terasa terlalu lama karena p_sample_loop berjalan 1.000 timestep per sampel, kamu bisa mengubah argumen num_eval_samples menjadi 64 atau 128 saat pengujian cepat lalu naikkan saat siap menghasilkan angka final.

Sign in

Gemini may display inaccurate info, including about people, so double-check its responses.

Google Privacy Policy
Opens in a new window
Google Terms of Service
Opens in a new window
Your privacy & Gemini Apps
Opens in a new window
Sign in
```
