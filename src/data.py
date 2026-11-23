"""Data pipeline Fashion-MNIST untuk Mini DDPM.

Kode di file ini dihasilkan lewat sesi vibe coding dengan Gemini,
lalu disesuaikan seperlunya. Lihat LOG-INTERAKSI.md untuk transkrip prompt.
"""

import torch
from torch.utils.data import Dataset, Subset, DataLoader
from torchvision import datasets, transforms
from pathlib import Path

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
    import sys
    from pathlib import Path

    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
    from src.config import Config

    cfg = Config()
    train_loader = get_dataloader(cfg, train=True)
    check_batch(train_loader, expected_batches=117, expected_batch_size=128)
