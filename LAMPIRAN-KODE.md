### `src/__init__.py`
```python
```

### `src/config.py`
```python
"""Konfigurasi terpusat Mini DDPM — Fashion-MNIST.

Semua hyperparameter ada di sini supaya eksperimen (Fase 4) cukup mengubah satu file
dan hasilnya bisa dibandingkan secara adil. Nilai default = hasil benchmark CPU
(Ryzen 7 8845HS, 16 thread): batch 128, ±1,3 menit/epoch pada subset 15k.
"""

from dataclasses import dataclass, asdict, field
from pathlib import Path

# Root proyek = satu level di atas src/. Dipakai agar folder output selalu
# dibuat di lokasi yang sama, tak peduli dari direktori mana skrip dijalankan.
PROJECT_ROOT = Path(__file__).resolve().parent.parent


@dataclass
class Config:
    # ---- Data ----
    dataset: str = "FashionMNIST"
    data_root: str = "data"
    subset_size: int = 15_000          # dari 60.000 gambar train
    image_size: int = 28
    channels: int = 1
    num_workers: int = 0               # Windows: 0 paling aman

    # ---- Model ----
    unet_base: int = 32                # channel pada resolusi penuh
    time_embed_dim: int = 128
    num_groups: int = 8                # GroupNorm

    # ---- Diffusion ----
    timesteps: int = 1000              # T
    beta_start: float = 1e-4
    beta_end: float = 0.02
    beta_schedule: str = "linear"

    # ---- Training ----
    batch_size: int = 128
    epochs: int = 25
    lr: float = 2e-4
    seed: int = 42
    sample_every: int = 5              # simpan grid gambar tiap N epoch
    num_sample_images: int = 64

    # ---- Path ----
    ckpt_dir: str = "checkpoints"
    log_dir: str = "logs"
    fig_dir: str = "report/figures"

    def __post_init__(self):
        for d in (self.data_root, self.ckpt_dir, self.log_dir, self.fig_dir):
            (PROJECT_ROOT / d).mkdir(parents=True, exist_ok=True)

    def to_dict(self) -> dict:
        return asdict(self)

    def run_name(self) -> str:
        """Nama unik per eksperimen — dipakai untuk nama file log & checkpoint."""
        return (
            f"bs{self.batch_size}_lr{self.lr:g}_ep{self.epochs}"
            f"_base{self.unet_base}_T{self.timesteps}"
        )


if __name__ == "__main__":
    cfg = Config()
    print("Konfigurasi aktif:")
    for k, v in cfg.to_dict().items():
        print(f"  {k:18s} = {v}")
    print(f"\nrun_name = {cfg.run_name()}")
```

### `src/data.py`
```python
"""Data pipeline Fashion-MNIST untuk Mini DDPM.

Kode di file ini dihasilkan lewat sesi vibe coding dengan Gemini,
lalu disesuaikan seperlunya. Lihat LOG-INTERAKSI.md untuk transkrip prompt.
"""

import sys
from pathlib import Path

import torch
from torch.utils.data import Dataset, Subset, DataLoader
from torchvision import datasets, transforms

# Pastikan root proyek ada di sys.path agar `python src/data.py` juga jalan
_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from src.config import Config, PROJECT_ROOT


def _resolve_data_root(cfg) -> Path:
    """Pastikan path dataset selalu absolut dan menempel di PROJECT_ROOT."""
    root = Path(cfg.data_root)
    if not root.is_absolute():
        root = (PROJECT_ROOT / root).resolve()
    root.mkdir(parents=True, exist_ok=True)
    return root


def get_dataset(cfg, train: bool = True) -> Dataset:
    """
    Memuat dataset Fashion-MNIST dengan normalisasi [-1, 1]
    dan pengambilan subset secara deterministik.
    """
    # 1. Pipeline transformasi tanpa augmentasi acak
    # ToTensor() mengubah piksel [0, 255] -> [0.0, 1.0]
    # Normalize((0.5,), (0.5,)) mengubah [0.0, 1.0] -> [-1.0, 1.0] dengan rumus (x - 0.5) / 0.5
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=(0.5,), std=(0.5,))
    ])

    # 2. Memuat dataset asli (1 channel grayscale, ukuran 28x28)
    data_path = _resolve_data_root(cfg)

    dataset = datasets.FashionMNIST(
        root=str(data_path),
        train=train,
        download=True,
        transform=transform
    )

    # 3. Pengambilan subset deterministik khusus data latih (misal: 15.000 sampel)
    subset_size = getattr(cfg, "subset_size", None)
    if train and subset_size is not None and subset_size < len(dataset):
        seed = getattr(cfg, "seed", 42)

        # Generator lokal agar urutan indeks selalu sama di setiap run
        generator = torch.Generator().manual_seed(seed)
        indices = torch.randperm(len(dataset), generator=generator)[:subset_size].tolist()

        dataset = Subset(dataset, indices)

    return dataset


def get_dataloader(cfg, train: bool = True) -> DataLoader:
    """
    Membungkus dataset Fashion-MNIST ke dalam PyTorch DataLoader
    dengan dukungan deterministik dan ukuran batch yang konsisten.
    """
    dataset = get_dataset(cfg, train=train)

    # Generator khusus DataLoader agar pengacakan batch tetap reproducible
    generator = torch.Generator()
    generator.manual_seed(getattr(cfg, "seed", 42))

    loader = DataLoader(
        dataset=dataset,
        batch_size=cfg.batch_size,
        shuffle=train,          # Acak saat training, urut saat evaluasi
        drop_last=train,        # Buang batch sisa saat training agar jumlah & ukuran batch konsisten
        num_workers=getattr(cfg, "num_workers", 0),
        pin_memory=False,       # False karena training berjalan penuh di CPU
        generator=generator if train else None,
    )

    return loader


def check_batch(loader: DataLoader, expected_batches: int = 117,
                expected_batch_size: int = 128) -> None:
    """
    Sanity check pada DataLoader: dimensi tensor, distribusi nilai piksel,
    jumlah batch, dan kelengkapan label.
    """
    num_batches = len(loader)
    images, labels = next(iter(loader))

    px_min = images.min().item()
    px_max = images.max().item()
    px_mean = images.mean().item()
    px_std = images.std().item()
    unique_labels = sorted(labels.unique().tolist())

    print("=== SANITY CHECK DATALOADER ===")
    print(f"Jumlah batch      : {num_batches}")
    print(f"Shape satu batch  : {tuple(images.shape)}")
    print(f"Min / Max piksel  : {px_min:.4f} / {px_max:.4f}")
    print(f"Mean / Std piksel : {px_mean:.4f} / {px_std:.4f}")
    print(f"Label unik ({len(unique_labels)})  : {unique_labels}")
    print("===============================")

    issues = []

    if num_batches != expected_batches:
        issues.append(f"Jumlah batch adalah {num_batches}, diharapkan {expected_batches}.")

    expected_shape = (expected_batch_size, 1, 28, 28)
    if tuple(images.shape) != expected_shape:
        issues.append(f"Shape batch adalah {tuple(images.shape)}, diharapkan {expected_shape}.")

    if px_min < -1.0001 or px_max > 1.0001:
        issues.append(f"Rentang piksel di luar [-1, 1]: [{px_min:.4f}, {px_max:.4f}].")

    if not (-0.8 <= px_mean <= 0.2):
        issues.append(f"Mean piksel ({px_mean:.4f}) terlalu ekstrem, periksa kembali rumus normalisasi.")

    if px_std <= 0.1:
        issues.append(f"Std piksel terlalu kecil ({px_std:.4f}), gambar berpotensi kosong/seragam.")

    if unique_labels != list(range(10)):
        issues.append(f"Label unik tidak lengkap (0-9): ditemukan {unique_labels}.")

    if not issues:
        print("STATUS: LULUS")
    else:
        print("STATUS: GAGAL")
        for idx, issue in enumerate(issues, 1):
            print(f"  {idx}. {issue}")


if __name__ == "__main__":
    cfg = Config()
    train_loader = get_dataloader(cfg, train=True)
    check_batch(train_loader, expected_batches=117, expected_batch_size=128)
```

### `src/diffusion.py`
```python
"""Denoising Diffusion Probabilistic Model (DDPM) from scratch.

Diimplementasikan mengikuti Ho et al. (2020), tanpa library diffusion siap pakai.
Kode awal dihasilkan lewat sesi vibe coding dengan Gemini lalu diverifikasi.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class DDPM(nn.Module):
    """
    Implementasi Denoising Diffusion Probabilistic Models (DDPM) dari nol
    berdasarkan Ho et al. (2020).
    """

    def __init__(
        self,
        timesteps: int = 1000,
        beta_start: float = 1e-4,
        beta_end: float = 0.02,
        clip_denoised: bool = True,
    ) -> None:
        super().__init__()
        self.timesteps = timesteps
        self.clip_denoised = clip_denoised

        # 1. Linear noise schedule: beta_1 sampai beta_T
        betas = torch.linspace(beta_start, beta_end, timesteps, dtype=torch.float32)

        # 2. Definisi alpha_t dan cumulative product alpha_bar_t
        alphas = 1.0 - betas
        alphas_cumprod = torch.cumprod(alphas, dim=0)

        # alpha_bar_{t-1}, dengan alpha_bar_0 didefinisikan = 1.0
        alphas_cumprod_prev = F.pad(alphas_cumprod[:-1], (1, 0), value=1.0)

        # Buffer: ikut state_dict tapi tidak dilatih
        self.register_buffer("betas", betas)
        self.register_buffer("alphas", alphas)
        self.register_buffer("alphas_cumprod", alphas_cumprod)
        self.register_buffer("alphas_cumprod_prev", alphas_cumprod_prev)

        # Koefisien untuk forward process q(x_t | x_0)
        self.register_buffer("sqrt_alphas_cumprod", torch.sqrt(alphas_cumprod))
        self.register_buffer("sqrt_one_minus_alphas_cumprod", torch.sqrt(1.0 - alphas_cumprod))

        # Koefisien untuk merekonstruksi x_0 dari prediksi noise
        self.register_buffer("sqrt_recip_alphas_cumprod", torch.sqrt(1.0 / alphas_cumprod))
        self.register_buffer("sqrt_recipm1_alphas_cumprod", torch.sqrt(1.0 / alphas_cumprod - 1.0))

        # Koefisien untuk distribusi posterior q(x_{t-1} | x_t, x_0)
        posterior_variance = betas * (1.0 - alphas_cumprod_prev) / (1.0 - alphas_cumprod)
        self.register_buffer("posterior_variance", posterior_variance)

        # Log variance di-clip pada t=0 karena posterior_variance[0] bernilai 0
        self.register_buffer(
            "posterior_log_variance_clipped",
            torch.log(torch.clamp(posterior_variance, min=1e-20)),
        )

        # Koefisien rata-rata posterior: mu_tilde = coef1 * x_0 + coef2 * x_t
        self.register_buffer(
            "posterior_mean_coef1",
            betas * torch.sqrt(alphas_cumprod_prev) / (1.0 - alphas_cumprod),
        )
        self.register_buffer(
            "posterior_mean_coef2",
            (1.0 - alphas_cumprod_prev) * torch.sqrt(alphas) / (1.0 - alphas_cumprod),
        )

    def _extract(self, a: torch.Tensor, t: torch.Tensor, x_shape: torch.Size) -> torch.Tensor:
        """
        Mengambil elemen ke-t dari tensor 1D `a` untuk setiap sampel dalam batch,
        lalu reshape ke (B, 1, 1, 1) agar bisa di-broadcast ke (B, C, H, W).
        """
        batch_size = t.shape[0]
        out = a.gather(0, t)
        return out.reshape(batch_size, *((1,) * (len(x_shape) - 1)))

    def q_sample(
        self,
        x0: torch.Tensor,
        t: torch.Tensor,
        noise: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """
        Forward diffusion: lompat langsung dari x_0 ke x_t.
        x_t = sqrt(alpha_bar_t) * x_0 + sqrt(1 - alpha_bar_t) * eps
        """
        if noise is None:
            noise = torch.randn_like(x0)

        sqrt_alpha_bar_t = self._extract(self.sqrt_alphas_cumprod, t, x0.shape)
        sqrt_one_minus_alpha_bar_t = self._extract(self.sqrt_one_minus_alphas_cumprod, t, x0.shape)

        return sqrt_alpha_bar_t * x0 + sqrt_one_minus_alpha_bar_t * noise

    def p_losses(
        self,
        model: nn.Module,
        x0: torch.Tensor,
        t: torch.Tensor,
        noise: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """
        Training loss (L_simple): MSE antara noise asli dan noise prediksi model.
        """
        if noise is None:
            noise = torch.randn_like(x0)

        x_t = self.q_sample(x0=x0, t=t, noise=noise)
        predicted_noise = model(x_t, t)
        loss = F.mse_loss(predicted_noise, noise)
        return loss

    def predict_start_from_noise(
        self,
        x_t: torch.Tensor,
        t: torch.Tensor,
        noise: torch.Tensor,
    ) -> torch.Tensor:
        """
        Estimasi x_0 dari x_t dan prediksi noise eps_theta.
        x_0 = (1 / sqrt(alpha_bar_t)) * x_t - sqrt(1 / alpha_bar_t - 1) * eps
        """
        coef1 = self._extract(self.sqrt_recip_alphas_cumprod, t, x_t.shape)
        coef2 = self._extract(self.sqrt_recipm1_alphas_cumprod, t, x_t.shape)
        return coef1 * x_t - coef2 * noise

    def q_posterior_mean_variance(
        self,
        x0: torch.Tensor,
        x_t: torch.Tensor,
        t: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Mean dan variance dari distribusi posterior q(x_{t-1} | x_t, x_0)."""
        posterior_mean = (
            self._extract(self.posterior_mean_coef1, t, x_t.shape) * x0
            + self._extract(self.posterior_mean_coef2, t, x_t.shape) * x_t
        )
        posterior_var = self._extract(self.posterior_variance, t, x_t.shape)
        posterior_log_var_clipped = self._extract(
            self.posterior_log_variance_clipped, t, x_t.shape
        )
        return posterior_mean, posterior_var, posterior_log_var_clipped

    def p_mean_variance(
        self,
        model: nn.Module,
        x_t: torch.Tensor,
        t: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Parameter distribusi reverse p_theta(x_{t-1} | x_t)."""
        pred_noise = model(x_t, t)

        x0_recon = self.predict_start_from_noise(x_t=x_t, t=t, noise=pred_noise)
        if self.clip_denoised:
            x0_recon = torch.clamp(x0_recon, -1.0, 1.0)

        model_mean, posterior_var, posterior_log_var = self.q_posterior_mean_variance(
            x0=x0_recon, x_t=x_t, t=t
        )
        return model_mean, posterior_var, posterior_log_var

    @torch.no_grad()
    def p_sample(
        self,
        model: nn.Module,
        x_t: torch.Tensor,
        t: torch.Tensor,
        t_index: int,
    ) -> torch.Tensor:
        """
        Satu langkah reverse diffusion: sampling x_{t-1} dari x_t.
        x_{t-1} = mu_theta(x_t, t) + sigma_t * z, z ~ N(0, I) untuk t > 0.
        """
        model_mean, _, model_log_var = self.p_mean_variance(model=model, x_t=x_t, t=t)

        if t_index == 0:
            return model_mean

        noise = torch.randn_like(x_t)
        sigma_t = torch.exp(0.5 * model_log_var)
        return model_mean + sigma_t * noise

    @torch.no_grad()
    def p_sample_loop(
        self,
        model: nn.Module,
        shape: tuple[int, int, int, int],
        device: torch.device | str = "cpu",
    ) -> torch.Tensor:
        """
        Generasi penuh: mulai dari x_T ~ N(0, I) lalu iterasi mundur sampai x_0.
        """
        model.eval()
        batch_size = shape[0]

        img = torch.randn(shape, device=device)

        for i in reversed(range(self.timesteps)):
            t = torch.full((batch_size,), i, device=device, dtype=torch.long)
            img = self.p_sample(model=model, x_t=img, t=t, t_index=i)

        return torch.clamp(img, -1.0, 1.0)


if __name__ == "__main__":
    class DummyUNet(nn.Module):
        def forward(self, x: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
            return torch.zeros_like(x)

    ddpm = DDPM(timesteps=1000, beta_start=1e-4, beta_end=0.02)
    dummy_model = DummyUNet()

    x0 = torch.randn(8, 1, 28, 28).clamp(-1.0, 1.0)
    t = torch.randint(0, 1000, (8,), dtype=torch.long)

    xt = ddpm.q_sample(x0, t)
    loss = ddpm.p_losses(dummy_model, x0, t)
    print(f"Shape x_t      : {tuple(xt.shape)}")
    print(f"Nilai MSE Loss : {loss.item():.4f}")

    xt_start = ddpm.q_sample(x0, torch.zeros(8, dtype=torch.long))
    xt_end = ddpm.q_sample(x0, torch.full((8,), 999, dtype=torch.long))
    print(f"Std di t=0     : {xt_start.std().item():.4f} (x0: {x0.std().item():.4f})")
    print(f"Std di t=999   : {xt_end.std().item():.4f} (mendekati 1.0000)")

    ddpm_fast = DDPM(timesteps=10)
    samples = ddpm_fast.p_sample_loop(dummy_model, shape=(4, 1, 28, 28))
    print(f"Shape sampel   : {tuple(samples.shape)}, Rentang: [{samples.min():.2f}, {samples.max():.2f}]")
```

### `src/model.py`
```python
"""U-Net untuk DDPM Fashion-MNIST.

Arsitektur mengikuti U-Net DDPM (Ho et al. 2020) yang diperkecil untuk 28x28
dan CPU. Kode awal dari sesi vibe coding dengan Gemini, lalu diverifikasi.

Resolusi: 28x28 -> 14x14 -> 7x7 (bottleneck + attention) -> 14x14 -> 28x28
"""

import math
import sys
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F

# Bootstrap: `python src/model.py` menaruh src/ di sys.path[0], bukan root proyek,
# sehingga `from src.config import ...` gagal tanpa baris ini.
_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))


class SinusoidalPositionEmbeddings(nn.Module):
    """Mengubah skalar timestep t menjadi vektor sinusoidal kontinu."""

    def __init__(self, dim: int):
        super().__init__()
        self.dim = dim

    def forward(self, time: torch.Tensor) -> torch.Tensor:
        device = time.device
        half_dim = self.dim // 2
        embeddings = math.log(10000) / (half_dim - 1)
        embeddings = torch.exp(torch.arange(half_dim, device=device) * -embeddings)
        embeddings = time[:, None].float() * embeddings[None, :]
        embeddings = torch.cat((embeddings.sin(), embeddings.cos()), dim=-1)
        return embeddings


class ResBlock(nn.Module):
    """Blok Residual DDPM: GroupNorm, SiLU, injeksi Time Embedding."""

    def __init__(self, in_ch: int, out_ch: int, time_emb_dim: int, groups: int = 8):
        super().__init__()
        self.norm1 = nn.GroupNorm(groups, in_ch)
        self.act1 = nn.SiLU()
        self.conv1 = nn.Conv2d(in_ch, out_ch, kernel_size=3, padding=1)

        # Proyeksi time embedding agar dimensinya cocok dengan out_ch
        self.time_mlp = nn.Sequential(
            nn.SiLU(),
            nn.Linear(time_emb_dim, out_ch)
        )

        self.norm2 = nn.GroupNorm(groups, out_ch)
        self.act2 = nn.SiLU()
        self.conv2 = nn.Conv2d(out_ch, out_ch, kernel_size=3, padding=1)

        # Shortcut jika jumlah channel berubah
        self.residual_conv = (
            nn.Conv2d(in_ch, out_ch, kernel_size=1) if in_ch != out_ch else nn.Identity()
        )

    def forward(self, x: torch.Tensor, t_emb: torch.Tensor) -> torch.Tensor:
        h = self.conv1(self.act1(self.norm1(x)))

        # Tambahkan informasi timestep ke feature map (broadcasting spasial)
        time_emb = self.time_mlp(t_emb)[:, :, None, None]
        h = h + time_emb

        h = self.conv2(self.act2(self.norm2(h)))
        return h + self.residual_conv(x)


class SelfAttention2d(nn.Module):
    """Multi-Head Self-Attention untuk resolusi rendah (7x7)."""

    def __init__(self, channels: int, num_heads: int = 4, groups: int = 8):
        super().__init__()
        self.num_heads = num_heads
        self.norm = nn.GroupNorm(groups, channels)
        self.qkv = nn.Conv2d(channels, channels * 3, kernel_size=1)
        self.proj = nn.Conv2d(channels, channels, kernel_size=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, C, H, W = x.shape
        h = self.norm(x)
        qkv = self.qkv(h)

        # Pisahkan Q, K, V untuk multi-head attention
        q, k, v = qkv.chunk(3, dim=1)
        head_dim = C // self.num_heads

        q = q.view(B, self.num_heads, head_dim, H * W).transpose(2, 3)
        k = k.view(B, self.num_heads, head_dim, H * W)
        v = v.view(B, self.num_heads, head_dim, H * W).transpose(2, 3)

        # Scaled Dot-Product Attention
        scale = head_dim ** -0.5
        attn = torch.matmul(q, k) * scale
        attn = F.softmax(attn, dim=-1)

        out = torch.matmul(attn, v)
        out = out.transpose(2, 3).contiguous().view(B, C, H, W)
        return x + self.proj(out)


class Downsample(nn.Module):
    """Downsampling 2x dengan strided convolution."""

    def __init__(self, channels: int):
        super().__init__()
        self.conv = nn.Conv2d(channels, channels, kernel_size=3, stride=2, padding=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.conv(x)


class Upsample(nn.Module):
    """Upsampling 2x: nearest-neighbor + conv (hindari checkerboard artifacts)."""

    def __init__(self, channels: int):
        super().__init__()
        self.conv = nn.Conv2d(channels, channels, kernel_size=3, padding=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = F.interpolate(x, scale_factor=2, mode="nearest")
        return self.conv(x)


class DDPMUNet(nn.Module):
    """
    U-Net untuk Fashion-MNIST (1x28x28).
    Resolusi: 28x28 -> 14x14 -> 7x7 (bottleneck + attention) -> 14x14 -> 28x28
    """

    def __init__(self, in_channels: int = 1, base_ch: int = 48, time_emb_dim: int = 192):
        super().__init__()

        # 1. Time Embedding Network
        self.time_mlp = nn.Sequential(
            SinusoidalPositionEmbeddings(base_ch),
            nn.Linear(base_ch, time_emb_dim),
            nn.SiLU(),
            nn.Linear(time_emb_dim, time_emb_dim),
        )

        # Proyeksi awal: (B, 1, 28, 28) -> (B, 48, 28, 28)
        self.init_conv = nn.Conv2d(in_channels, base_ch, kernel_size=3, padding=1)

        # 2. Encoder
        # Level 1: 28x28
        self.down1_res1 = ResBlock(base_ch, base_ch, time_emb_dim)
        self.down1_res2 = ResBlock(base_ch, base_ch, time_emb_dim)
        self.downsample1 = Downsample(base_ch)  # -> 14x14

        # Level 2: 14x14
        self.down2_res1 = ResBlock(base_ch, base_ch * 2, time_emb_dim)
        self.down2_res2 = ResBlock(base_ch * 2, base_ch * 2, time_emb_dim)
        self.downsample2 = Downsample(base_ch * 2)  # -> 7x7

        # 3. Bottleneck: 7x7
        self.mid_res1 = ResBlock(base_ch * 2, base_ch * 2, time_emb_dim)
        self.mid_attn = SelfAttention2d(base_ch * 2, num_heads=4)
        self.mid_res2 = ResBlock(base_ch * 2, base_ch * 2, time_emb_dim)

        # 4. Decoder dengan skip connections
        # 7x7 -> 14x14
        self.upsample1 = Upsample(base_ch * 2)
        self.up2_res1 = ResBlock(base_ch * 4, base_ch * 2, time_emb_dim)
        self.up2_res2 = ResBlock(base_ch * 2, base_ch * 2, time_emb_dim)

        # 14x14 -> 28x28
        self.upsample2 = Upsample(base_ch * 2)
        self.up1_res1 = ResBlock(base_ch * 3, base_ch, time_emb_dim)
        self.up1_res2 = ResBlock(base_ch, base_ch, time_emb_dim)

        # 5. Output Projection
        self.final_norm = nn.GroupNorm(8, base_ch)
        self.final_act = nn.SiLU()
        self.final_conv = nn.Conv2d(base_ch, in_channels, kernel_size=3, padding=1)

    def forward(self, x: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        t_emb = self.time_mlp(t)

        x0 = self.init_conv(x)

        # Encoder
        d1 = self.down1_res1(x0, t_emb)
        d1 = self.down1_res2(d1, t_emb)          # skip (28x28, 48 ch)
        p1 = self.downsample1(d1)                # (14x14, 48 ch)

        d2 = self.down2_res1(p1, t_emb)
        d2 = self.down2_res2(d2, t_emb)          # skip (14x14, 96 ch)
        p2 = self.downsample2(d2)                # (7x7, 96 ch)

        # Bottleneck
        m = self.mid_res1(p2, t_emb)
        m = self.mid_attn(m)
        m = self.mid_res2(m, t_emb)              # (7x7, 96 ch)

        # Decoder
        u1 = self.upsample1(m)                   # (14x14, 96 ch)
        u1 = torch.cat([u1, d2], dim=1)          # (14x14, 192 ch)
        u1 = self.up2_res1(u1, t_emb)
        u1 = self.up2_res2(u1, t_emb)            # (14x14, 96 ch)

        u2 = self.upsample2(u1)                  # (28x28, 96 ch)
        u2 = torch.cat([u2, d1], dim=1)          # (28x28, 144 ch)
        u2 = self.up1_res1(u2, t_emb)
        u2 = self.up1_res2(u2, t_emb)            # (28x28, 48 ch)

        out = self.final_conv(self.final_act(self.final_norm(u2)))
        return out


if __name__ == "__main__":
    # Pakai Config() supaya angka parameter yang dicetak sama dengan yang dipakai
    # training (base_ch=32, time_emb_dim=128), bukan nilai demo yang berbeda.
    from src.config import Config
    cfg = Config()
    model = DDPMUNet(
        in_channels=cfg.channels,
        base_ch=cfg.unet_base,
        time_emb_dim=cfg.time_embed_dim,
    )
    total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Total Trainable Parameters: {total_params:,}")

    dummy_x = torch.randn(4, 1, 28, 28)
    dummy_t = torch.randint(0, cfg.timesteps, (4,))
    pred_noise = model(dummy_x, dummy_t)
    print(f"Input shape : {dummy_x.shape}")
    print(f"Output shape: {pred_noise.shape}")
```

### `src/plot_loss.py`
```python
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
```

### `src/sample.py`
```python
import os
import re
import glob
import sys
from pathlib import Path

import torch
from torchvision.utils import save_image

# Pastikan folder root proyek (parent dari src/) selalu ada di sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.config import Config
from src.model import DDPMUNet
from src.diffusion import DDPM


def extract_epoch_num(filepath: str) -> int:
    match = re.search(r"ddpm_epoch_(\d+)\.pt$", os.path.basename(filepath))
    return int(match.group(1)) if match else -1


@torch.no_grad()
def generate_with_fixed_noise(
    model: DDPMUNet,
    diffusion: DDPM,
    num_samples: int,
    img_size: int,
    channels: int,
    device: torch.device,
    fixed_seed: int = 1234,
) -> torch.Tensor:
    """
    Mereset PRNG sebelum p_sample_loop agar x_T (dan noise p_sample)
    identik persis di setiap epoch yang dievaluasi.
    """
    model.eval()
    torch.manual_seed(fixed_seed)
    if device.type == "cuda":
        torch.cuda.manual_seed_all(fixed_seed)

    shape = (num_samples, channels, img_size, img_size)
    samples = diffusion.p_sample_loop(model, shape=shape, device=device)

    # Denormalisasi dari [-1, 1] ke [0, 1]
    samples = (samples.clamp(-1.0, 1.0) + 1.0) * 0.5
    return samples


def build_epoch_progression_grid(
    cfg: Config,
    target_epochs: list[int] | None = None,
    num_samples: int = 8,
    fixed_seed: int = 1234,
):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    os.makedirs(cfg.fig_dir, exist_ok=True)

    # Cari semua checkpoint epoch
    ckpt_files = sorted(
        glob.glob(os.path.join(cfg.ckpt_dir, "ddpm_epoch_*.pt")),
        key=extract_epoch_num,
    )
    if not ckpt_files:
        raise FileNotFoundError(f"Tidak ada checkpoint ditemukan di {cfg.ckpt_dir}")

    # Filter epoch tertentu jika diminta (misal [1, 5, 10, 20, 50])
    if target_epochs is not None:
        ckpt_files = [f for f in ckpt_files if extract_epoch_num(f) in target_epochs]

    model = DDPMUNet(
        in_channels=cfg.channels,
        base_ch=cfg.unet_base,
        time_emb_dim=cfg.time_embed_dim,
    ).to(device)

    diffusion = DDPM(
        timesteps=cfg.timesteps,
        beta_start=cfg.beta_start,
        beta_end=cfg.beta_end,
        clip_denoised=True,
    ).to(device)

    all_epoch_rows = []
    evaluated_epochs = []

    for ckpt_path in ckpt_files:
        epoch_num = extract_epoch_num(ckpt_path)
        ckpt = torch.load(ckpt_path, map_location=device, weights_only=True)
        model.load_state_dict(ckpt["model_state_dict"])

        print(f"Sampling epoch {epoch_num:03d} dengan fixed noise...")
        imgs = generate_with_fixed_noise(
            model=model,
            diffusion=diffusion,
            num_samples=num_samples,
            img_size=cfg.image_size,
            channels=cfg.channels,
            device=device,
            fixed_seed=fixed_seed,
        )

        # Simpan juga grid individual per epoch
        single_grid_path = os.path.join(cfg.fig_dir, f"samples_epoch_{epoch_num:03d}.png")
        save_image(imgs, single_grid_path, nrow=num_samples, padding=2)

        all_epoch_rows.append(imgs)
        evaluated_epochs.append(epoch_num)

    # Susun matriks gambar:
    # Baris = Epoch (dari atas ke bawah: epoch awal -> akhir)
    # Kolom = Sampel ke-1 s.d. ke-N (dari noise awal yang sama persis)
    progression_tensor = torch.cat(all_epoch_rows, dim=0)
    progression_path = os.path.join(
        cfg.fig_dir,
        f"progression_epochs_{evaluated_epochs[0]}_to_{evaluated_epochs[-1]}.png",
    )
    save_image(progression_tensor, progression_path, nrow=num_samples, padding=2)
    print(f"Grid progres berhasil disimpan ke: {progression_path}")


if __name__ == "__main__":
    cfg = Config()
    # Isi target_epochs=None untuk semua epoch yang tersimpan,
    # atau isi list spesifik seperti [1, 5, 10, 15, 20] agar grid tidak terlalu tinggi.
    build_epoch_progression_grid(
        cfg=cfg,
        target_epochs=None,
        num_samples=8,
        fixed_seed=1234,
    )
```

### `src/train.py`
```python
import os
import json
import time
import sys
from pathlib import Path

import torch
from tqdm import tqdm

# Pastikan folder root proyek (parent dari src/) selalu ada di sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.config import Config
from src.data import get_dataloader, check_batch
from src.model import DDPMUNet
from src.diffusion import DDPM


def set_seed(seed: int = 42):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def train(cfg: Config):
    set_seed(cfg.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    os.makedirs(cfg.ckpt_dir, exist_ok=True)
    os.makedirs(cfg.log_dir, exist_ok=True)

    # 1. Data
    train_loader = get_dataloader(cfg, train=True)
    check_batch(
        train_loader,
        expected_batches=len(train_loader),
        expected_batch_size=cfg.batch_size,
    )

    # 2. Model & Diffusion (keduanya WAJIB .to(device))
    model = DDPMUNet(
        in_channels=cfg.channels,
        base_ch=cfg.unet_base,
        time_emb_dim=cfg.time_embed_dim,
    ).to(device)

    diffusion = DDPM(
        timesteps=cfg.timesteps,
        beta_start=cfg.beta_start,
        beta_end=cfg.beta_end,
        clip_denoised=True,
    ).to(device)

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=cfg.lr,
        weight_decay=getattr(cfg, "weight_decay", 1e-4),
    )

    use_amp = device.type == "cuda"
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp)

    history = []
    best_loss = float("inf")

    # 3. Training Loop
    for epoch in range(1, cfg.epochs + 1):
        model.train()
        epoch_loss = 0.0
        start_time = time.time()

        pbar = tqdm(train_loader, desc=f"Epoch {epoch:03d}/{cfg.epochs:03d}", leave=False)
        for x0, _ in pbar:
            x0 = x0.to(device, non_blocking=True)
            t = torch.randint(0, cfg.timesteps, (x0.size(0),), device=device).long()

            optimizer.zero_grad(set_to_none=True)

            with torch.amp.autocast("cuda", enabled=use_amp):
                loss = diffusion.p_losses(model, x0, t)

            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            scaler.step(optimizer)
            scaler.update()

            epoch_loss += loss.item()
            pbar.set_postfix(loss=f"{loss.item():.4f}")

        avg_loss = epoch_loss / len(train_loader)
        elapsed = time.time() - start_time
        history.append({"epoch": epoch, "loss": avg_loss, "time_sec": round(elapsed, 2)})

        print(f"[Epoch {epoch:03d}/{cfg.epochs:03d}] Loss: {avg_loss:.5f} | Time: {elapsed:.1f}s")

        # 4. Save Checkpoint per Epoch
        ckpt_payload = {
            "epoch": epoch,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "loss": avg_loss,
            "config": vars(cfg) if hasattr(cfg, "__dict__") else {},
        }

        epoch_ckpt_path = os.path.join(cfg.ckpt_dir, f"ddpm_epoch_{epoch:03d}.pt")
        torch.save(ckpt_payload, epoch_ckpt_path)

        if avg_loss < best_loss:
            best_loss = avg_loss
            torch.save(ckpt_payload, os.path.join(cfg.ckpt_dir, "ddpm_best.pt"))

        with open(os.path.join(cfg.log_dir, "train_history.json"), "w") as f:
            json.dump(history, f, indent=2)


if __name__ == "__main__":
    cfg = Config()
    train(cfg)
```

### `src/evaluate.py`
```python
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
```

### `src/ablation.py`
```python
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
```

### `src/figures.py`
```python
import sys
from pathlib import Path

# Bootstrap sys.path di level modul agar `python src/figures.py` mendukung import absolut `src.*`
PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import time
import matplotlib

matplotlib.use("Agg")  # Backend non-interaktif agar aman di CPU/Windows tanpa GUI
import matplotlib.pyplot as plt
import torch

from src.config import Config
from src.data import get_dataset
from src.evaluate import load_checkpoint_and_models
from src.sample import generate_with_fixed_noise

FASHION_MNIST_CLASSES: list[str] = [
    "T-shirt/top",
    "Trouser",
    "Pullover",
    "Dress",
    "Coat",
    "Sandal",
    "Shirt",
    "Sneaker",
    "Bag",
    "Ankle boot",
]


def _select_stratified_real_samples(
    cfg: Config, n_samples: int = 6, seed: int = 42
) -> tuple[torch.Tensor, list[int]]:
    """Ambil sampel data asli terstratifikasi per kelas secara deterministik."""
    test_dataset = get_dataset(cfg, train=False)
    generator = torch.Generator().manual_seed(seed)
    shuffled_indices = torch.randperm(len(test_dataset), generator=generator).tolist()

    selected_by_class: dict[int, torch.Tensor] = {}

    for idx in shuffled_indices:
        img, label = test_dataset[idx]
        label_int = int(label)
        if label_int in selected_by_class:
            continue
        # Denormalisasi dari [-1, 1] ke [0, 1] sesuai transform Normalize(0.5, 0.5)
        selected_by_class[label_int] = (img * 0.5 + 0.5).clamp(0.0, 1.0)
        if len(selected_by_class) >= n_samples:
            break

    ordered_pairs = sorted(selected_by_class.items(), key=lambda item: item[0])

    real_labels = [lbl for lbl, _ in ordered_pairs[:n_samples]]
    real_images = torch.stack([img for _, img in ordered_pairs[:n_samples]], dim=0)
    return real_images, real_labels


def build_comparison_grid(
    real: torch.Tensor,
    fake: torch.Tensor,
    real_labels: list[int] | torch.Tensor,
    class_names: list[str],
    save_path: Path | str,
    n_cols: int = 6,
) -> Path:
    """Bangun figur PNG 2 baris x n_cols kolom (baris asli berlabel, baris sampling tanpa label)."""
    if n_cols < 1 or n_cols > 12:
        raise ValueError(f"n_cols harus berada di rentang 1..12, diterima: {n_cols}")
    if real.shape[0] < n_cols or fake.shape[0] < n_cols:
        raise ValueError(
            f"Jumlah sampel tidak cukup untuk n_cols={n_cols}: "
            f"real={real.shape[0]}, fake={fake.shape[0]}"
        )

    save_path = Path(save_path)
    save_path.parent.mkdir(parents=True, exist_ok=True)

    real_np = real[:n_cols].detach().cpu().clamp(0.0, 1.0).numpy()
    fake_np = fake[:n_cols].detach().cpu().clamp(0.0, 1.0).numpy()
    labels_list = [int(lbl) for lbl in real_labels[:n_cols]]

    fig, axes = plt.subplots(
        nrows=2,
        ncols=n_cols,
        figsize=(n_cols * 2.0, 4.9),
        gridspec_kw={"hspace": 0.48, "wspace": 0.08},
    )
    if n_cols == 1:
        axes = axes.reshape(2, 1)

    for col in range(n_cols):
        # Baris atas: data asli beserta label kelas
        ax_real = axes[0, col]
        ax_real.imshow(real_np[col, 0], cmap="gray", vmin=0.0, vmax=1.0)
        ax_real.set_xticks([])
        ax_real.set_yticks([])
        ax_real.set_xlabel(class_names[labels_list[col]], fontsize=9, labelpad=4)
        for spine in ax_real.spines.values():
            spine.set_edgecolor("#CCCCCC")
            spine.set_linewidth(0.7)

        # Baris bawah: hasil sampling tanpa label kelas
        ax_fake = axes[1, col]
        ax_fake.imshow(fake_np[col, 0], cmap="gray", vmin=0.0, vmax=1.0)
        ax_fake.set_xticks([])
        ax_fake.set_yticks([])
        for spine in ax_fake.spines.values():
            spine.set_edgecolor("#CCCCCC")
            spine.set_linewidth(0.7)

    # Hitung posisi aktual tiap baris agar judul baris rata kiri presisi dan tidak terpotong
    fig.canvas.draw()
    left_x = axes[0, 0].get_position().x0
    top_row_y = axes[0, 0].get_position().y1 + 0.025
    bottom_row_y = axes[1, 0].get_position().y1 + 0.025

    fig.text(
        left_x,
        top_row_y,
        "Data asli (test set)",
        fontsize=11,
        fontweight="bold",
        ha="left",
        va="bottom",
    )
    fig.text(
        left_x,
        bottom_row_y,
        "Hasil sampling (T=1000, seed 1234)",
        fontsize=11,
        fontweight="bold",
        ha="left",
        va="bottom",
    )

    fig.savefig(save_path, dpi=200, bbox_inches="tight", pad_inches=0.15)
    plt.close(fig)
    return save_path


def main() -> None:
    start_time = time.perf_counter()

    device = torch.device("cpu")
    n_cols = 6
    sampling_seed = 1234

    default_cfg = Config()
    ckpt_path = Path(default_cfg.ckpt_dir) / "ddpm_best.pt"
    cfg, model, diffusion, _ = load_checkpoint_and_models(ckpt_path, device)

    real_images, real_labels = _select_stratified_real_samples(
        cfg=cfg,
        n_samples=n_cols,
        seed=cfg.seed,
    )

    fake_images = generate_with_fixed_noise(
        model=model,
        diffusion=diffusion,
        num_samples=n_cols,
        img_size=cfg.image_size,
        channels=cfg.channels,
        device=device,
        fixed_seed=sampling_seed,
    )

    output_path = Path(cfg.fig_dir) / "real_vs_generated.png"
    saved_path = build_comparison_grid(
        real=real_images,
        fake=fake_images,
        real_labels=real_labels,
        class_names=FASHION_MNIST_CLASSES,
        save_path=output_path,
        n_cols=n_cols,
    )

    elapsed = time.perf_counter() - start_time
    print(f"Figur disimpan   : {saved_path.as_posix()}")
    print(f"Shape tensor real: {tuple(real_images.shape)}")
    print(f"Shape tensor fake: {tuple(fake_images.shape)}")
    print(f"Waktu eksekusi   : {elapsed:.2f} detik ({elapsed / 60.0:.2f} menit)")


if __name__ == "__main__":
    main()
```

### `src/figures_ablation.py`
```python
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
    main()```

### `src/verify_seed_n128.py`

```python
"""Verifikasi determinisme sampling terhadap ukuran batch (Sesi #9, susulan).

Pertanyaan yang dijawab: apakah 6 sampel pada baris linear
`report/figures/ablation_schedule_grid.png` sama persis dengan 6 sampel
pertama bila sampling dijalankan dengan batch N=128?

Hasil yang diperoleh: TIDAK sama. Noise awal (x_T) identik, tetapi forward
pass U-Net dengan batch berbeda menghasilkan selisih numerik sekitar 1e-6,
dan selisih itu teramplifikasi oleh 1000 langkah reverse diffusion sampai
gambar akhirnya berbeda secara visual. Grid-nya sendiri reproducible
bit-exact (sha256 PNG sama).

Skrip ini menulis ringkasan ke `report/verify_seed_n128.json` dan tensor ke
`report/verify_seed_batches.pt` (tidak di-commit).
"""

from pathlib import Path
import hashlib
import json
import sys
import time

# Bootstrap sys.path agar import `from src...` berjalan saat skrip dijalankan langsung
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import matplotlib

matplotlib.use("Agg")

import numpy as np
import torch
from PIL import Image

from src.figures_ablation import (
    COSINE_CKPT_DIR,
    COSINE_HISTORY_PATH,
    FIXED_SEED,
    LINEAR_CKPT_DIR,
    OUTPUT_PATH,
    _find_checkpoint_file,
    _sample_cosine_checkpoint,
    _sample_linear_checkpoint,
    build_ablation_grid,
)

OUT_JSON = PROJECT_ROOT / "report" / "verify_seed_n128.json"
TMP_PNG = PROJECT_ROOT / "report" / "figures" / "_grid_repro_tmp.png"
BATCH_N = 128


def main() -> None:
    res: dict = {}
    device = torch.device("cpu")

    lin_ckpt = _find_checkpoint_file(LINEAR_CKPT_DIR)
    cos_ckpt = _find_checkpoint_file(COSINE_CKPT_DIR)
    print("checkpoint linear:", lin_ckpt, flush=True)
    print("checkpoint cosine:", cos_ckpt, flush=True)

    t0 = time.perf_counter()
    s6 = _sample_linear_checkpoint(lin_ckpt, device, 6, FIXED_SEED)
    t_n6 = time.perf_counter() - t0
    print(f"N=6 linear selesai: {t_n6:.2f} s", flush=True)

    t0 = time.perf_counter()
    s128 = _sample_linear_checkpoint(lin_ckpt, device, BATCH_N, FIXED_SEED)
    t_n128 = time.perf_counter() - t0
    print(f"N={BATCH_N} linear selesai: {t_n128:.2f} s", flush=True)

    first6 = s128[:6]
    diff = (s6 - first6).abs()
    a8 = (s6.clamp(0, 1) * 255).round().to(torch.uint8)
    b8 = (first6.clamp(0, 1) * 255).round().to(torch.uint8)

    res["linear_ckpt"] = str(lin_ckpt)
    res["seed"] = FIXED_SEED
    res["batch_compared"] = BATCH_N
    res["time_n6_s"] = round(t_n6, 2)
    res["time_n128_s"] = round(t_n128, 2)
    res["float_equal"] = bool(torch.equal(s6, first6))
    res["max_abs_diff"] = float(diff.max())
    res["mean_abs_diff"] = float(diff.mean())
    res["uint8_equal"] = bool(torch.equal(a8, b8))
    res["n_pixels_8bit_diff"] = int((a8 != b8).sum().item())
    res["n_pixels_total"] = int(a8.numel())
    print(json.dumps(res, indent=2), flush=True)

    t0 = time.perf_counter()
    s6cos = _sample_cosine_checkpoint(cos_ckpt, COSINE_HISTORY_PATH, device, 6, FIXED_SEED)
    res["time_n6_cosine_s"] = round(time.perf_counter() - t0, 2)
    print(f"N=6 cosine selesai: {res['time_n6_cosine_s']:.2f} s", flush=True)

    build_ablation_grid(s6, s6cos, TMP_PNG)
    old = np.asarray(Image.open(OUTPUT_PATH).convert("RGB")).astype(np.int16)
    new = np.asarray(Image.open(TMP_PNG).convert("RGB")).astype(np.int16)
    res["grid_shape_old"] = list(old.shape)
    res["grid_shape_new"] = list(new.shape)
    if old.shape == new.shape:
        gd = np.abs(old - new)
        res["grid_identical"] = bool(gd.max() == 0)
        res["grid_max_channel_diff"] = int(gd.max())
        res["grid_pixels_diff"] = int((gd.sum(axis=2) > 0).sum())
    else:
        res["grid_identical"] = False
        res["grid_shape_mismatch"] = True
    res["sha256_existing_png"] = hashlib.sha256(OUTPUT_PATH.read_bytes()).hexdigest()
    res["sha256_reproduced_png"] = hashlib.sha256(TMP_PNG.read_bytes()).hexdigest()
    print(json.dumps(res, indent=2), flush=True)

    torch.save(
        {"s6": s6, "s128_first6": first6, "s6_cosine": s6cos},
        PROJECT_ROOT / "report" / "verify_seed_batches.pt",
    )
    OUT_JSON.write_text(json.dumps(res, indent=2), encoding="utf-8")
    TMP_PNG.unlink(missing_ok=True)
    print("DONE", flush=True)


if __name__ == "__main__":
    main()
```
