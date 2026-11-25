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

