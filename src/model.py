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
