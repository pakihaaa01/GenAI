# Sesi #8 (ulang) — `src/figures.py`

- **Tautan share:** https://share.gemini.google/cKH8zn52Mdc6
- **Dibuat:** 27/11/2026 00.19 WIB · **Diperiksa:** 27/11/2026 00.27 WIB (halaman ter-render, 21.953 karakter)
- **Operator:** Mia Rosiana Santoso · **Model:** Gemini
- **Status:** tautan **aktif**. Percakapan ini adalah permintaan ulang karena tautan asli `NhHSVgoMabeC` mengembalikan "Link doesn't exist" saat diperiksa 26/11/2026.
- **Keputusan:** kode pada percakapan ini **tidak dipakai** di repo. Kode yang berjalan tetap hasil percakapan asli (transkrip: `sesi-8.md`). Alasan ada di bagian akhir berkas ini.

## Prompt yang dikirim (verbatim)

```
halo, aku lanjut proyek Mini DDPM from scratch buat Fashion-MNIST, 28x28 grayscale 1 channel. training 25 epoch udah selesai: loss 0.16831 turun ke 0.04297, model 793.953 parameter (base_ch=32, time_emb_dim=128), T=1000, beta linear 1e-4 sampai 0.02. jalan di CPU Windows, python 3.13, torch 2.15.0+cpu.
sekarang aku mau nambah satu figur buat bab hasil laporan. idenya: gambar asli dari dataset dan gambar hasil sampling model ditampilkan berdampingan per kelas, biar pembaca bisa lihat sendiri seberapa mirip hasil model tanpa harus percaya angka loss doang. file baru src/figures.py, outputnya report/figures/real_vs_generated.png.
biar ga tumpang tindih, ini figur yang udah ada di report/figures/ beserta isinya, jangan dibuat ulang:
- loss_curve.png: kurva training loss per epoch, 25 titik
- progression_epochs_1_to_25.png: kumpulan grid sampel dari beberapa checkpoint, epoch awal sampai akhir, buat nunjukin progres kualitas
- samples_epoch_001.png sampai samples_epoch_025.png: satu grid sampel per epoch, pakai noise awal yang sama biar perubahannya kelihatan
- evaluation_metrics.png: tiga panel, yaitu grid 4x4 sampel, histogram intensitas piksel, dan histogram gradien sobel, dipakai buat bandingin distribusi asli vs generated
- ablation_noise_schedule.png: dua kurva loss, schedule linear vs cosine
- ablation_schedule_grid.png: 6 sampel linear di baris atas dan 6 sampel cosine di baris bawah, plus label MMD tiap varian
jadi yang belum ada itu figur "asli vs generated per kelas" yang simpel dan langsung kelihatan mata. yang aku minta:
1. gambar asli jangan diambil acak, tapi stratifikasi: ambil sejumlah gambar dari tiap 10 kelas. jelasin kenapa stratifikasi lebih tepat buat figur perbandingan daripada random sampling
2. hasil sampling pakai checkpoint checkpoints/ddpm_best.pt dengan seed tetap. jelasin kenapa seed harus dikunci
3. susunannya jelas: mana baris asli mana baris generated, ada judul, nama kelas di bawah tiap kolom, dan tetap kebaca kalau dicetak hitam putih
4. kode harus bisa dijalanin ulang di CPU tanpa GPU, jangan hardcode path absolut, tampilkan ringkasan waktu eksekusi di akhir
5. kasih kode lengkap plus alasan desainnya, bukan potongan. aku harus bisa jelasin tiap pilihan ini di laporan
```

## Jawaban Gemini (teks halaman share, apa adanya)

Alasan Desain Tiap Pilihan (Buat Bahan Penjelasan di Laporan)

Stratified Sampling vs. Random Sampling untuk Gambar Asli

Kenapa stratifikasi lebih tepat: Pada random sampling murni dari dataset, jumlah sampel tiap kelas bisa tidak seimbang (ada kelas yang muncul 3 kali, ada yang 0 kali), sehingga kolom perbandingan per kelas jadi bolong atau tidak adil. Dengan stratified sampling (mengambil tepat 
𝑘
k sampel per kelas untuk ke-10 kelas Fashion-MNIST), setiap kelas punya representasi setara dan sejajar tepat di atas sampel hasil generasi kelas yang sama.

Cara kerja: Kita kumpulkan indeks per label 0..9 dari dataset, acak urutan indeks di dalam tiap kelas menggunakan seed tetap, lalu ambil 
𝑛
per_class
n
per_class
	​

 pertama per kelas.

Mengunci Seed Saat Sampling dari checkpoints/ddpm_best.pt

Kenapa seed harus dikunci: Proses reverse diffusion (p_sample_loop / sampling dari 
𝑥
𝑇
∼
𝑁
(
0
,
𝐼
)
x
T
	​

∼N(0,I)) bersifat stokastik. Tanpa mengunci seed (random, np.random, torch.manual_seed), setiap kali src/figures.py dijalankan ulang, gambar generated yang muncul akan berubah—sehingga narasi di bab hasil laporan bisa tidak sinkron dengan gambar yang terpasang.

Reproducibility: Seed yang dikunci membuat figur bisa direproduksi persis sama di mesin/CPU yang sama maupun saat pengujian ulang.

Tata Letak Jelas & Ramah Cetak Hitam-Putih (Grayscale / B&W)

Susunan baris & kolom: Ada 10 kolom (1 kolom = 1 kelas Fashion-MNIST). Untuk tiap kelas, ditampilkan gambar Real dan gambar Generated secara berdampingan/berpasangan agar mata pembaca bisa langsung membandingkan bentuk siluet, tekstur, dan kontrasnya. Misalnya, dalam 1 blok per kelas (atau 2 baris utama: baris Real dan baris Generated, atau pasangan kiri-kanan/atas-bawah per kelas—di bawah ini kita susun 20 baris atau 10 blok pasangan / 2 baris per sampel yang sangat jelas: tiap kelas punya baris Real dan baris Generated tepat di bawahnya, lengkap dengan label teks tegas REAL dan GEN, garis pemisah antar kelas, serta nama kelas di bagian bawah tiap kolom).

Keterbacaan saat dicetak hitam-putih: Tidak mengandalkan warna saja untuk membedakan Real vs Generated. Kita pakai label teks kapital (REAL vs GEN), garis pemisah tegas (linewidth), dan latar belakang/bingkai kontras tinggi (#FFFFFF dan #EAEAEA / border hitam solid) supaya tetap terbaca jelas meski dicetak monokrom/hitam-putih di laporan.

Kompatibilitas CPU Windows, Path Relatif, & Ringkasan Waktu

Semua operasi dipaksa ke torch.device("cpu") dengan map_location="cpu" dan num_workers=0 agar aman di Windows + Python 3.13 + torch 2.15.0+cpu.

Root proyek dihitung dinamis lewat Path(__file__).resolve().parent.parent, lalu ditambahkan ke sys.path sebelum mengimpor modul dari src/—jadi bebas dari hardcoded path absolut dan bisa dijalankan dari direktori mana saja (python src/figures.py).

Waktu muat checkpoint/data, waktu sampling/rendering, dan total waktu eksekusi dicatat dengan time.perf_counter() lalu dicetak di akhir.

Kode Lengkap src/figures.py

Catatan kompatibilitas modul src/: kode ini mengimpor Config, get_dataloader (atau fallback get_dataset), dan DDPM / DDPMUNet sesuai struktur standar proyek Mini DDPM kamu, dengan deteksi atribut yang fleksibel agar aman bila ada sedikit perbedaan nama method/field antar modul.

Python
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
        from src.data import get_dataset  # type: ignore

        try:
            return get_dataset(train=False)
        except TypeError:
            return get_dataset()
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
    base_ch = getattr(cfg, "base_ch", 32)
    time_emb_dim = getattr(cfg, "time_emb_dim", 128)
    timesteps = getattr(cfg, "timesteps", getattr(cfg, "T", 1000))
    beta_start = getattr(cfg, "beta_start", 1e-4)
    beta_end = getattr(cfg, "beta_end", 0.02)

    unet = DDPMUNet(
        in_channels=in_channels,
        out_channels=in_channels,
        base_ch=base_ch,
        time_emb_dim=time_emb_dim,
    )

    try:
        model = DDPM(
            model=unet,
            timesteps=timesteps,
            beta_start=beta_start,
            beta_end=beta_end,
            device=device,
        )
    except TypeError:
        model = DDPM(unet, timesteps=timesteps, beta_start=beta_start, beta_end=beta_end).to(device)

    model = model.to(device)

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
        model.load_state_dict(state_dict, strict=True)
    except RuntimeError:
        # Fallback jika state_dict milik UNet saja dan berada di atribut `model` atau `unet`.
        if hasattr(model, "model") and isinstance(model.model, torch.nn.Module):
            model.model.load_state_dict(state_dict, strict=True)
        elif hasattr(model, "unet") and isinstance(model.unet, torch.nn.Module):
            model.unet.load_state_dict(state_dict, strict=True)
        else:
            raise

    model.eval()
    return model


@torch.no_grad()
def sample_ddpm_batch(
    model: torch.nn.Module,
    num_samples: int,
    device: torch.device,
    img_size: int = 28,
    channels: int = 1,
) -> list[np.ndarray]:
    """Menghasilkan `num_samples` citra dari model DDPM dan mengembalikannya dalam rentang [0, 1]."""
    samples: torch.Tensor | None = None

    if hasattr(model, "sample") and callable(model.sample):
        try:
            samples = model.sample(batch_size=num_samples, device=device)
        except TypeError:
            samples = model.sample(num_samples)
    elif hasattr(model, "p_sample_loop") and callable(model.p_sample_loop):
        shape = (num_samples, channels, img_size, img_size)
        samples = model.p_sample_loop(shape=shape, device=device)
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
        "Perbandingan Citra Asli (Stratified) vs. Hasil Sampling DDPM per Kelas — Fashion-MNIST",
        fontsize=12.5,
        fontweight="bold",
        color="#111111",
        y=0.985,
    )

    legend_text = (
        f"Checkpoint: checkpoints/ddpm_best.pt  |  Seed: {seed}  |  "
        "REAL: garis bingkai solid + citra dataset terstratifikasi  |  "
        "GEN: garis bingkai putus-putus (--) + sampel model DDPM"
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
    model = build_and_load_ddpm(checkpoint_path=checkpoint_path, device=device)
    t_ckpt_elapsed = time.perf_counter() - t_ckpt_start

    # Kunci ulang seed tepat sebelum sampling agar urutan noise awal selalu identik.
    set_deterministic_seed(seed)
    t_sample_start = time.perf_counter()
    gen_samples = sample_ddpm_batch(
        model=model,
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

Catatan Praktis Sebelum Dijalankan

Kecepatan sampling di CPU: Karena model menggunakan T=1000 langkah reverse diffusion pada CPU, proses sample_ddpm_batch untuk 20 sampel akan memakan waktu lebih lama dibanding sekadar plotting—itulah alasan ringkasan waktu per tahap dicetak di akhir agar kamu bisa mencatat durasinya dengan akurat di bagian eksperimen/LAMPIRAN.

Jumlah sampel per kelas: Di main(), samples_per_class = 2 memberi 2 baris REAL dan 2 baris GEN per kelas (cukup padat untuk membandingkan variasi antar sampel tanpa membuat figur terlalu tinggi saat dimasukkan ke Word/LaTeX). Jika ingin 1 sampel per kelas agar lebih ringkas, cukup ubah samples_per_class = 1.

## Apa yang terjadi saat kode dijalankan

Kode disalin apa adanya ke `src/figures.py` (441 baris), lalu dijalankan di `.venv`:

```
Traceback (most recent call last):
  File "C:\Users\rosiana\Documents\GenAI\src\figures.py", line 441, in <module>
    main()
    ~~~~^^
  File "C:\Users\rosiana\Documents\GenAI\src\figures.py", line 396, in main
    model = build_and_load_ddpm(checkpoint_path=checkpoint_path, device=device)
  File "C:\Users\rosiana\Documents\GenAI\src\figures.py", line 151, in build_and_load_ddpm
    unet = DDPMUNet(
        in_channels=in_channels,
    ...<2 lines>...
        time_emb_dim=time_emb_dim,
    )
TypeError: DDPMUNet.__init__() got an unexpected keyword argument 'out_channels'. Did you mean 'in_channels'?
```

Tiga penyesuaian API setelah itu (tanpa mengubah struktur/komentar kode):

1. `out_channels` dihapus dari pemanggilan `DDPMUNet` — kelas di `src/model.py` hanya menerima `in_channels`, `base_ch`, `time_emb_dim`.
2. `DDPM(model=unet, ...)` diganti `DDPM(timesteps, beta_start, beta_end)`; objek U-Net dititipkan saat sampling: `diffusion.p_sample_loop(unet, shape=shape, device=device)`.
3. Nama field config: `base_ch`/`time_emb_dim` tidak ada di `src/config.py`; yang ada `unet_base`/`time_embed_dim`, jadi ditambah fallback `getattr`.

Hasil run setelah penyesuaian: exit 0, data 4,23 s · checkpoint 0,23 s · sampling (T=1000) 192,93 s · render 2,79 s · total 200,23 s; figur 4 baris × 10 kolom (20 asli + 20 generated), 3410×2244 px, 198.205 byte.

## Kenapa tidak dipakai

Prompt yang dikirim identik, tetapi jawaban berbeda dari sesi asli: sesi asli menghasilkan figur 2×6 (6 kelas unik) dengan durasi 69,03 detik, dan angka itulah yang sudah tertulis di `LOG-INTERAKSI.md`, `LAPORAN-PROYEK1.md`, `LAMPIRAN-COMMIT.md`, serta `sesi-8.md`. Mengganti kode berarti harus menulis ulang seluruh angka dan tabel di dokumen yang sudah diselaraskan dengan anggota lain.

Karena itu kode+figur baru di-commit (`bd87886`) lalu **di-revert** (`b9618c9`); `src/figures.py` dan `report/figures/real_vs_generated.png` kembali byte-identik dengan versi yang dilaporkan (PNG 62.189 byte, sha256 `bfef9b94d1666b832b3e2f8f8914312c4fada208b998039d9982f1571cfad126`). Berkas ini disimpan sebagai bukti percakapan pengganti, bukan sebagai sumber kode.
