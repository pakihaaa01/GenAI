from __future__ import annotations

import sys
import time
from pathlib import Path
from typing import Any

# Bootstrap root proyek agar `from src.xxx import yyy` selalu berhasil
# saat script dijalankan langsung (`python src/figures.py`).
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
import numpy as np
import torch
from torch.utils.data import DataLoader

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


def set_deterministic_seed(seed: int) -> None:
    """Mengunci seed Python, NumPy, dan PyTorch untuk hasil yang reproducible."""
    import random

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def resolve_device() -> torch.device:
    """Selalu menggunakan CPU agar konsisten dan aman di lingkungan CPU Windows."""
    return torch.device("cpu")


def load_fashion_mnist_dataset(data_dir: Path) -> Any:
    """Memuat dataset Fashion-MNIST menggunakan helper proyek jika ada, atau torchvision."""
    try:
        from src.config import Config  # type: ignore
        from src.data import get_dataset  # type: ignore

        try:
            return get_dataset(Config(), train=False)
        except TypeError:
            return get_dataset(train=False)
    except Exception:
        from torchvision import datasets, transforms

        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.5,), (0.5,)),
        ])
        return datasets.FashionMNIST(
            root=str(data_dir),
            train=False,
            download=True,
            transform=transform,
        )


def extract_targets(dataset: Any) -> np.ndarray:
    """Mengambil array label kelas (0..9) secara aman dari dataset."""
    if hasattr(dataset, "targets"):
        raw = dataset.targets
        if isinstance(raw, torch.Tensor):
            return raw.detach().cpu().numpy()
        return np.asarray(raw)
    if hasattr(dataset, "labels"):
        raw = dataset.labels
        if isinstance(raw, torch.Tensor):
            return raw.detach().cpu().numpy()
        return np.asarray(raw)

    labels: list[int] = []
    for i in range(len(dataset)):
        _, y = dataset[i]
        labels.append(int(y.item() if isinstance(y, torch.Tensor) else y))
    return np.asarray(labels, dtype=np.int64)


def stratified_sample_per_class(
    dataset: Any,
    samples_per_class: int,
    seed: int,
    num_classes: int = 10,
) -> dict[int, list[np.ndarray]]:
    """
    Mengambil tepat `samples_per_class` gambar asli per kelas (stratified sampling)
    dan menormalkan citra ke rentang [0.0, 1.0] bentuk (28, 28).
    """
    rng = np.random.default_rng(seed)
    targets = extract_targets(dataset)
    stratified_images: dict[int, list[np.ndarray]] = {}

    for cls_idx in range(num_classes):
        cls_indices = np.where(targets == cls_idx)[0]
        if len(cls_indices) < samples_per_class:
            raise ValueError(
                f"Kelas {cls_idx} hanya memiliki {len(cls_indices)} sampel, "
                f"lebih sedikit dari yang diminta ({samples_per_class})."
            )
        chosen_indices = rng.choice(cls_indices, size=samples_per_class, replace=False)
        imgs: list[np.ndarray] = []
        for idx in chosen_indices:
            img, _ = dataset[int(idx)]
            if isinstance(img, torch.Tensor):
                arr = img.detach().cpu().squeeze().numpy().astype(np.float32)
            else:
                arr = np.asarray(img, dtype=np.float32).squeeze()

            # Jika tensor berada di rentang [-1, 1], petakan ke [0, 1].
            if arr.min() < -0.05:
                arr = (arr + 1.0) * 0.5
            arr = np.clip(arr, 0.0, 1.0)
            imgs.append(arr)
        stratified_images[cls_idx] = imgs

    return stratified_images


def build_and_load_ddpm(checkpoint_path: Path, device: torch.device) -> torch.nn.Module:
    """Membangun model DDPM dari modul `src` dan memuat bobot dari `checkpoints/ddpm_best.pt`."""
    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Checkpoint tidak ditemukan: {checkpoint_path}")

    ckpt = torch.load(checkpoint_path, map_location=device)

    from src.config import Config  # type: ignore
    from src.diffusion import DDPM  # type: ignore
    from src.model import DDPMUNet  # type: ignore

    cfg = Config()
    in_channels = getattr(cfg, "channels", getattr(cfg, "in_channels", 1))
    base_ch = getattr(cfg, "base_ch", getattr(cfg, "unet_base", 32))
    time_emb_dim = getattr(cfg, "time_emb_dim", getattr(cfg, "time_embed_dim", 128))
    timesteps = getattr(cfg, "timesteps", getattr(cfg, "T", 1000))
    beta_start = getattr(cfg, "beta_start", 1e-4)
    beta_end = getattr(cfg, "beta_end", 0.02)

    unet = DDPMUNet(
        in_channels=in_channels,
        base_ch=base_ch,
        time_emb_dim=time_emb_dim,
    ).to(device)

    # Di repo ini DDPM adalah modul difusi terpisah: U-Net dititipkan sebagai argumen
    # saat sampling (diffusion.p_sample_loop(unet, shape=..., device=...)), bukan
    # disimpan sebagai atribut.
    diffusion = DDPM(
        timesteps=timesteps,
        beta_start=beta_start,
        beta_end=beta_end,
    ).to(device)

    if isinstance(ckpt, dict):
        if "model_state_dict" in ckpt:
            state_dict = ckpt["model_state_dict"]
        elif "state_dict" in ckpt:
            state_dict = ckpt["state_dict"]
        elif "model" in ckpt and isinstance(ckpt["model"], dict):
            state_dict = ckpt["model"]
        else:
            state_dict = ckpt
    else:
        raise ValueError("Format checkpoint tidak dikenali.")

    try:
        unet.load_state_dict(state_dict, strict=True)
    except RuntimeError:
        # Fallback jika state_dict dibungkus lagi (mis. {"unet": ...}).
        if "unet" in state_dict and isinstance(state_dict["unet"], dict):
            unet.load_state_dict(state_dict["unet"], strict=True)
        elif "model" in state_dict and isinstance(state_dict["model"], dict):
            unet.load_state_dict(state_dict["model"], strict=True)
        else:
            raise

    unet.eval()
    return unet, diffusion


@torch.no_grad()
def sample_ddpm_batch(
    unet: torch.nn.Module,
    diffusion: torch.nn.Module,
    num_samples: int,
    device: torch.device,
    img_size: int = 28,
    channels: int = 1,
) -> list[np.ndarray]:
    """Menghasilkan `num_samples` citra dari model DDPM dan mengembalikannya dalam rentang [0, 1]."""
    samples: torch.Tensor | None = None

    if hasattr(diffusion, "p_sample_loop") and callable(diffusion.p_sample_loop):
        shape = (num_samples, channels, img_size, img_size)
        samples = diffusion.p_sample_loop(unet, shape=shape, device=device)
    elif hasattr(diffusion, "sample") and callable(diffusion.sample):
        samples = diffusion.sample(batch_size=num_samples, device=device)
    else:
        raise AttributeError("Model DDPM tidak memiliki method `sample` maupun `p_sample_loop`.")

    if isinstance(samples, tuple):
        samples = samples[0]

    arr_batch = samples.detach().cpu().numpy().astype(np.float32)
    out: list[np.ndarray] = []
    for i in range(num_samples):
        img = arr_batch[i].squeeze()
        if img.min() < -0.05:
            img = (img + 1.0) * 0.5
        out.append(np.clip(img, 0.0, 1.0))
    return out


def plot_real_vs_generated_per_class(
    real_by_class: dict[int, list[np.ndarray]],
    gen_samples: list[np.ndarray],
    samples_per_class: int,
    seed: int,
    save_path: Path,
) -> None:
    """
    Membuat figur perbandingan gambar asli (stratified per kelas) vs hasil sampling
    model DDPM dengan tata letak jelas, nama kelas di bawah tiap kolom, serta
    keterbacaan tinggi saat dicetak hitam-putih.
    """
    num_classes = len(FASHION_MNIST_CLASSES)
    total_needed = num_classes * samples_per_class
    if len(gen_samples) < total_needed:
        raise ValueError(f"Butuh {total_needed} sampel generated, tersedia {len(gen_samples)}.")

    # Setiap kelas memiliki `samples_per_class` pasangan (Real di atas, Gen di bawah).
    # Baris total = samples_per_class * 2.
    total_rows = samples_per_class * 2

    plt.rcParams["font.family"] = "sans-serif"
    plt.rcParams["axes.edgecolor"] = "#222222"
    plt.rcParams["axes.linewidth"] = 0.8

    fig, axes = plt.subplots(
        nrows=total_rows,
        ncols=num_classes,
        figsize=(15.5, 2.2 * total_rows + 1.4),
        gridspec_kw={"wspace": 0.14, "hspace": 0.22},
    )
    fig.patch.set_facecolor("white")

    if total_rows == 1:
        axes = np.expand_dims(axes, axis=0)

    gen_cursor = 0
    for cls_idx, class_name in enumerate(FASHION_MNIST_CLASSES):
        for s_idx in range(samples_per_class):
            r_real = s_idx * 2
            r_gen = s_idx * 2 + 1

            ax_real = axes[r_real, cls_idx]
            ax_gen = axes[r_gen, cls_idx]

            real_img = real_by_class[cls_idx][s_idx]
            gen_img = gen_samples[gen_cursor]
            gen_cursor += 1

            # Panel REAL (bingkai solid hitam)
            ax_real.imshow(real_img, cmap="gray", vmin=0.0, vmax=1.0, interpolation="nearest")
            ax_real.set_xticks([])
            ax_real.set_yticks([])
            for spine in ax_real.spines.values():
                spine.set_visible(True)
                spine.set_color("#111111")
                spine.set_linewidth(1.1)

            # Panel GEN (bingkai garis putus-putus abu gelap agar beda jelas di cetak B&W)
            ax_gen.imshow(gen_img, cmap="gray", vmin=0.0, vmax=1.0, interpolation="nearest")
            ax_gen.set_xticks([])
            ax_gen.set_yticks([])
            for spine in ax_gen.spines.values():
                spine.set_visible(True)
                spine.set_color("#333333")
                spine.set_linestyle("--")
                spine.set_linewidth(1.1)

            # Label baris di kolom paling kiri (jelas untuk cetak hitam-putih)
            if cls_idx == 0:
                ax_real.set_ylabel(
                    f"REAL #{s_idx + 1}",
                    rotation=0,
                    ha="right",
                    va="center",
                    fontsize=8.5,
                    fontweight="bold",
                    color="#111111",
                    labelpad=8,
                )
                ax_gen.set_ylabel(
                    f"GEN #{s_idx + 1}",
                    rotation=0,
                    ha="right",
                    va="center",
                    fontsize=8.5,
                    fontweight="bold",
                    color="#333333",
                    labelpad=8,
                )

            # Nama kelas ditempatkan di bawah tiap kolom (pada baris terbawah)
            if r_gen == total_rows - 1:
                ax_gen.set_xlabel(
                    class_name,
                    fontsize=9,
                    fontweight="bold",
                    color="#111111",
                    labelpad=6,
                )

    # Judul utama & legenda tipe baris yang kontras untuk cetak hitam-putih
    fig.suptitle(
        "Citra Asli Terstratifikasi per Kelas vs. Sampel DDPM Unconditional — Fashion-MNIST",
        fontsize=12.5,
        fontweight="bold",
        color="#111111",
        y=0.985,
    )

    legend_text = (
        f"Checkpoint: checkpoints/ddpm_best.pt  |  Seed: {seed}  |  "
        "REAL: bingkai solid, citra dataset terstratifikasi  |  "
        "GEN: bingkai putus-putus (--), sampel DDPM unconditional "
        "(kolom bukan kelas bersyarat)"
    )
    fig.text(
        0.5,
        0.958,
        legend_text,
        ha="center",
        va="center",
        fontsize=8.5,
        color="#222222",
        bbox=dict(
            boxstyle="round,pad=0.35",
            facecolor="#F2F2F2",
            edgecolor="#444444",
            linewidth=0.8,
        ),
    )

    save_path.parent.mkdir(parents=True, exist_ok=True)
    plt.subplots_adjust(left=0.08, right=0.99, top=0.93, bottom=0.06)
    fig.savefig(save_path, dpi=220, facecolor="white")
    plt.close(fig)


def main() -> None:
    t_start = time.perf_counter()

    seed = 42
    samples_per_class = 2  # 2 real + 2 gen per kelas = 20 real & 20 gen total
    num_classes = len(FASHION_MNIST_CLASSES)
    total_gen_needed = num_classes * samples_per_class

    data_dir = PROJECT_ROOT / "data"
    checkpoint_path = PROJECT_ROOT / "checkpoints" / "ddpm_best.pt"
    output_path = PROJECT_ROOT / "report" / "figures" / "real_vs_generated.png"

    device = resolve_device()
    set_deterministic_seed(seed)

    t_data_start = time.perf_counter()
    dataset = load_fashion_mnist_dataset(data_dir)
    real_by_class = stratified_sample_per_class(
        dataset=dataset,
        samples_per_class=samples_per_class,
        seed=seed,
        num_classes=num_classes,
    )
    t_data_elapsed = time.perf_counter() - t_data_start

    t_ckpt_start = time.perf_counter()
    unet, diffusion = build_and_load_ddpm(checkpoint_path=checkpoint_path, device=device)
    t_ckpt_elapsed = time.perf_counter() - t_ckpt_start

    # Kunci ulang seed tepat sebelum sampling agar urutan noise awal selalu identik.
    set_deterministic_seed(seed)
    t_sample_start = time.perf_counter()
    gen_samples = sample_ddpm_batch(
        unet=unet,
        diffusion=diffusion,
        num_samples=total_gen_needed,
        device=device,
        img_size=28,
        channels=1,
    )
    t_sample_elapsed = time.perf_counter() - t_sample_start

    t_plot_start = time.perf_counter()
    plot_real_vs_generated_per_class(
        real_by_class=real_by_class,
        gen_samples=gen_samples,
        samples_per_class=samples_per_class,
        seed=seed,
        save_path=output_path,
    )
    t_plot_elapsed = time.perf_counter() - t_plot_start

    total_elapsed = time.perf_counter() - t_start

    print("=" * 60)
    print("RINGKASAN EKSEKUSI — src/figures.py")
    print("=" * 60)
    print(f"Perangkat        : {device}")
    print(f"Checkpoint       : {checkpoint_path.relative_to(PROJECT_ROOT)}")
    print(f"Output Figur     : {output_path.relative_to(PROJECT_ROOT)}")
    print(f"Seed             : {seed}")
    print(f"Sampel / Kelas   : {samples_per_class} real + {samples_per_class} generated")
    print("-" * 60)
    print(f"Waktu Load & Stratified Data : {t_data_elapsed:.2f} s")
    print(f"Waktu Load Checkpoint Model  : {t_ckpt_elapsed:.2f} s")
    print(f"Waktu Sampling DDPM (T=1000) : {t_sample_elapsed:.2f} s")
    print(f"Waktu Render & Simpan Figur : {t_plot_elapsed:.2f} s")
    print(f"Total Waktu Eksekusi         : {total_elapsed:.2f} s")
    print("=" * 60)


if __name__ == "__main__":
    main()
