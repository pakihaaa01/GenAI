"""Fase 1 — Data pipeline Fashion-MNIST untuk DDPM.

Poin penting DDPM (dan semua model difusi):
  gambar harus dinormalisasi ke rentang [-1, 1], BUKAN [0, 1].
  Alasan: proses forward diffusion menambahkan noise Gaussian dengan rata-rata 0,
  sehingga data yang terpusat di 0 membuat skala noise dan skala data sebanding.
  Kalau data di [0,1], model akan "melihat" noise yang jauh lebih besar dari sinyal.

Jalankan modul ini langsung untuk memverifikasi pipeline:
    python src/data.py
Keluarannya: ringkasan shape/rentang + grid gambar contoh di report/figures/batch_contoh.png
"""

from pathlib import Path

import torch
from torch.utils.data import DataLoader, Subset
from torchvision import datasets, transforms

from config import Config

# Root proyek = satu level di atas folder src/.
# Semua path output dihitung dari sini supaya modul bisa dijalankan dari cwd mana pun
# (python src/data.py maupun cd src && python data.py memberi hasil sama).
PROJECT_ROOT = Path(__file__).resolve().parent.parent


def build_transform(cfg: Config) -> transforms.Compose:
    """Transform minimal: tensor + normalisasi ke [-1, 1].

    FashionMNIST sudah 28x28 grayscale dan terpusat, jadi tidak perlu resize/crop.
    RandomHorizontalFlip sengaja TIDAK dipakai: membalik gambar sepatu/sandal
    menghasilkan objek yang tidak realistis, dan DDPM tidak butuh augmentasi
    sebanyak classifier karena setiap sampel sudah menghasilkan ribuan target noise.
    """
    return transforms.Compose([
        transforms.ToTensor(),                      # [0,1], shape (1,28,28)
        transforms.Normalize(mean=[0.5], std=[0.5])  # -> [-1,1]
    ])


def get_dataset(cfg: Config, train: bool = True):
    """Ambil dataset, dengan subset deterministik untuk train."""
    ds = datasets.FashionMNIST(
        root=str(PROJECT_ROOT / cfg.data_root),
        train=train,
        download=True,
        transform=build_transform(cfg),
    )
    if train and cfg.subset_size and cfg.subset_size < len(ds):
        # Subset deterministik: ambil N pertama setelah shuffle ber-seed tetap.
        # Tujuan: waktu training turun ~4x tanpa mengubah distribusi kelas.
        g = torch.Generator().manual_seed(cfg.seed)
        idx = torch.randperm(len(ds), generator=g)[: cfg.subset_size].tolist()
        ds = Subset(ds, idx)
    return ds


def get_dataloader(cfg: Config, train: bool = True, shuffle: bool = True) -> DataLoader:
    ds = get_dataset(cfg, train=train)
    return DataLoader(
        ds,
        batch_size=cfg.batch_size,
        shuffle=shuffle,
        num_workers=cfg.num_workers,
        pin_memory=False,      # CPU-only, pin_memory tidak memberi manfaat
        drop_last=True,        # hindari batch terakhir berukuran 1 (merusak BatchNorm/statistik)
    )


def check_batch(cfg: Config) -> dict:
    """Verifikasi bentuk & rentang satu batch. Dipakai sebagai unit test manual."""
    dl = get_dataloader(cfg, train=True)
    x, y = next(iter(dl))
    info = {
        "n_batch": len(dl),
        "x_shape": tuple(x.shape),
        "x_dtype": str(x.dtype),
        "x_min": float(x.min()),
        "x_max": float(x.max()),
        "x_mean": float(x.mean()),
        "x_std": float(x.std()),
        "label_shape": tuple(y.shape),
        "label_unique": sorted(y.unique().tolist()),
        "expected_shape": (cfg.batch_size, cfg.channels, cfg.image_size, cfg.image_size),
    }
    info["shape_ok"] = info["x_shape"] == info["expected_shape"]
    info["range_ok"] = (info["x_min"] >= -1.01) and (info["x_max"] <= 1.01)
    return info


def save_grid(cfg: Config, path: str | Path | None = None, n: int = 32) -> Path:
    """Simpan grid gambar asli (denormalisasi ke [0,1]) sebagai bukti pipeline benar."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    if path is None:
        path = PROJECT_ROOT / cfg.fig_dir / "batch_contoh.png"

    dl = get_dataloader(cfg, train=True)
    x, y = next(iter(dl))
    x = (x[:n] * 0.5 + 0.5).clamp(0, 1)          # denormalisasi balik ke [0,1]

    cols = 8
    rows = (min(n, x.size(0)) + cols - 1) // cols
    fig, axes = plt.subplots(rows, cols, figsize=(cols * 1.3, rows * 1.3))
    for i, ax in enumerate(axes.flat):
        ax.axis("off")
        if i < x.size(0):
            ax.imshow(x[i, 0].numpy(), cmap="gray")
            ax.set_title(str(int(y[i])), fontsize=7)
    fig.tight_layout()

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=120)
    plt.close(fig)
    return path


if __name__ == "__main__":
    cfg = Config()
    info = check_batch(cfg)

    print("=== Verifikasi Data Pipeline ===")
    for k, v in info.items():
        print(f"  {k:16s} = {v}")

    ok = info["shape_ok"] and info["range_ok"]
    print(f"\nShape sesuai  : {info['shape_ok']}  (harus {info['expected_shape']})")
    print(f"Rentang [-1,1]: {info['range_ok']}")
    print(f"\nSTATUS: {'LULUS' if ok else 'GAGAL — periksa transform'}")

    p = save_grid(cfg)
    print(f"Grid contoh disimpan: {p}")
