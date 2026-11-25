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
