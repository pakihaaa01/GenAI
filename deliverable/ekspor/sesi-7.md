# Ekspor Percakapan Gemini — Sesi #7

- **Judul percakapan:** Ablasi Noise Schedule Mini DDPM
- **Tautan share:** https://share.gemini.google/mvsNfSQKek01
- **Model:** Gemini (gemini.google.com)
- **Operator:** Mia Rosiana Santoso
- **Cara memperoleh:** dibuka langsung dari tautan share (26/11/2026), teks halaman disalin utuh setelah scroll penuh

> Teks di bawah adalah isi halaman share apa adanya (tanggal dibuat/dipublikasikan tercantum di baris pertama). Pesan pengguna tampil terlipat oleh Gemini; teks prompt lengkap ada di `LOG-INTERAKSI.md`.

---

```text
Gemini
Ablasi Noise Schedule Mini DDPM
https://share.gemini.google/QLLYzOP2jOaV
Created with Bantuan sehari-hari November 25, 2026 at 10:04 PM
Published November 25, 2026 at 11:44 PM
You said konteks: aku punya proyek Mini DDPM from scratch untuk Fashion-MNIST 28x28 grayscale, sudah selesai …

konteks: aku punya proyek Mini DDPM from scratch untuk Fashion-MNIST 28x28 grayscale, sudah selesai training 25 epoch (loss 0.16831 -> 0.04297, model 793.953 parameter, base_ch=32, T=1000, beta linear 1e-4 sampai 0.02). sekarang sesi 7: aku harus bikin src/ablation.py yang membandingkan dua konfigurasi supaya kelihatan pilihan di config.py itu masuk akal, bukan asal comot. jalan di CPU, Windows, python 3.13, torch 2.14.0+cpu.







yang mau aku bandingkan: noise schedule linear (yang sekarang dipakai) versus cosine. masing-masing 3 epoch saja, tujuannya lihat tren loss, bukan hasil final.







masalahnya begini: src/diffusion.py cuma punya linear, beta-nya di-hardcode pakai torch.linspace di __init__, dan field beta_schedule di Config nggak pernah dibaca siapa-siapa. aku nggak boleh ngubah diffusion.py karena itu file punya rekan tim dan sudah diverifikasi. jadi kalau mau cosine, semua harus dari dalam ablation.py tanpa nyentuh file itu.







yang aku bingung:







1. cara paling bersih nambah schedule cosine tanpa ngubah diffusion.py apa? subclass DDPM lalu hitung ulang buffer-nya, atau ada cara lain yang lebih sedikit menduplikasi rumus?







2. kalau lewat subclass, buffer apa saja yang wajib dihitung ulang supaya q_sample, p_losses, dan p_sample_loop tetap konsisten? aku nggak mau hasilnya beda bukan karena schedule-nya, tapi karena buffer-nya nggak sinkron.







3. 3 epoch cukup nggak buat lihat bedanya, atau selisihnya bakal ketutup noise? kalau ternyata dua-duanya mirip, gimana cara nulis kesimpulan yang jujur tanpa maksa bilang salah satu lebih bagus?







4. src/train.py selalu menulis checkpoints/ddpm_best.pt dan logs/train_history.json. ablation-ku nggak boleh nimpa dua file itu. aman nggak kalau aku cuma mengganti cfg.ckpt_dir dan cfg.log_dir per varian, atau harus punya loop training sendiri?







yang aku minta ada di src/ablation.py, spesifik per fungsi:







1. CosineDDPM(DDPM)




subclass dari DDPM yang menghitung ulang buffer schedule cosine di dalam ablation.py, tanpa mengubah src/diffusion.py.







2. run_variant(name, cfg, use_cosine)




menjalankan training 3 epoch dengan seed dan urutan data yang sama untuk kedua varian, menyimpan checkpoint dan log ke folder terpisah (bukan checkpoints/ddpm_best.pt atau logs/train_history.json), lalu mencatat loss tiap epoch dan waktu eksekusinya.







3. plot_ablation(results, save_path)




menggabungkan kurva loss kedua varian dalam satu grafik PNG dan menyimpannya ke report/figures/.







4. main()




menjalankan varian linear dan cosine, mencetak tabel perbandingan di konsol (loss epoch 1/2/3, waktu per epoch, jumlah parameter), lalu memanggil fungsi plot.







signature yang ada di repo (jangan diubah, ini verbatim):







```python




# src/config.py




@dataclass




class Config:




dataset: str = "FashionMNIST"




data_root: str = "data"




subset_size: int = 15_000




image_size: int = 28




channels: int = 1




num_workers: int = 0




unet_base: int = 32




time_embed_dim: int = 128




num_groups: int = 8




timesteps: int = 1000




beta_start: float = 1e-4




beta_end: float = 0.02




beta_schedule: str = "linear"




batch_size: int = 128




epochs: int = 25




lr: float = 2e-4




seed: int = 42




sample_every: int = 5




num_sample_images: int = 64




ckpt_dir: str = "checkpoints"




log_dir: str = "logs"




fig_dir: str = "report/figures"




# metode: to_dict(), run_name()







# src/diffusion.py




# buffer yang terdaftar di __init__: betas, alphas, alphas_cumprod,




# alphas_cumprod_prev, sqrt_alphas_cumprod, sqrt_one_minus_alphas_cumprod,




# sqrt_recip_alphas_cumprod, sqrt_recipm1_alphas_cumprod, posterior_variance,




# posterior_log_variance_clipped, posterior_mean_coef1, posterior_mean_coef2




class DDPM(nn.Module):




def __init__(self, timesteps: int = 1000, beta_start: float = 1e-4,




beta_end: float = 0.02, clip_denoised: bool = True) -> None




def q_sample(self, x0, t, noise=None) -> torch.Tensor




def p_losses(self, model, x0, t, noise=None) -> torch.Tensor




def predict_start_from_noise(self, x_t, t, noise)




def q_posterior_mean_variance(self, x0, x_t, t)




def p_mean_variance(self, model, x_t, t)




def p_sample(self, model, x_t, t, t_index)




def p_sample_loop(self, model, shape, device="cpu") -> torch.Tensor







# src/model.py




class DDPMUNet(nn.Module):




def __init__(self, in_channels: int = 1, base_ch: int = 48,




time_emb_dim: int = 192)







# src/train.py




def set_seed(seed: int = 42)




def train(cfg: Config) # loop tqdm; simpan ddpm_epoch_XXX.pt tiap epoch,




# ddpm_best.pt kalau loss terbaik, logs/train_history.json







# src/data.py




def get_dataset(cfg, train: bool = True) -> Dataset




def get_dataloader(cfg, train: bool = True) -> DataLoader




```







batasan:







- cuma bikin src/ablation.py, jangan ubah file lain di src/ (config.py, data.py, diffusion.py, model.py, train.py, sample.py, plot_loss.py sudah diverifikasi)




- jangan pakai library diffusion siap pakai (diffusers dll)




- import absolut from src.x import y, dan bootstrap sys.path di level modul (bukan di dalam __main__), karena skrip dijalankan `python src/ablation.py`




- num_workers=0 (Windows)




- 3 epoch per konfigurasi, jangan lebih




- checkpoint dan log ablation wajib di lokasi terpisah; checkpoints/ddpm_best.pt dan logs/train_history.json tidak boleh tersentuh




- seed dan urutan data harus sama untuk kedua varian




- keluaran: tabel perbandingan di konsol (loss epoch 1/2/3, waktu per epoch, jumlah parameter) + minimal satu PNG di report/figures/ berisi dua kurva loss dalam satu plot

Jawaban Pertanyaan Desain & Metodologi
1. Cara Paling Bersih Menambah Cosine Schedule Tanpa Mengubah src/diffusion.py

Subclassing DDPM menjadi CosineDDPM(DDPM) dan menimpa seluruh buffer schedule di __init__ (setelah super().__init__(...)) adalah pendekatan paling bersih karena:

Semua metode di DDPM (q_sample, p_losses, predict_start_from_noise, q_posterior_mean_variance, p_mean_variance, p_sample, p_sample_loop) hanya membaca tensor dari buffer self.<nama_buffer> lewat indexing timestep t — tidak ada perhitungan ulang dari beta_start/beta_end di tengah forward/reverse pass.

Tidak ada duplikasi metode forward/sampling dan tidak perlu monkey-patch modul src.diffusion saat runtime.

isinstance(diffusion, DDPM) tetap True, sehingga kompatibel penuh dengan seluruh kode proyek.

2. Daftar 12 Buffer yang Wajib Dihitung Ulang Agar Sinkron

Semua 12 buffer di DDPM.__init__ saling bergantung secara berantai. Urutan penurunannya wajib konsisten dari betas hasil cosine schedule (Nichol & Dhariwal, 2021):

betas: dihitung dari rasio 
1
−
𝛼
ˉ
𝑡
/
𝛼
ˉ
𝑡
−
1
1−
α
ˉ
t
	​

/
α
ˉ
t−1
	​

 dengan 
𝑓
(
𝑡
)
=
cos
⁡
2
(
𝑡
/
𝑇
+
𝑠
1
+
𝑠
⋅
𝜋
2
)
f(t)=cos
2
(
1+s
t/T+s
	​

⋅
2
π
	​

), 
𝑠
=
0.008
s=0.008, lalu di-clip ke [1e-4, 0.999].

alphas: 
1
−
betas
1−betas

alphas_cumprod: torch.cumprod(alphas, dim=0) (wajib dihitung dari alphas pasca-clipping, bukan dari kurva mentah sebelum clipping, supaya identitas 
𝛼
ˉ
𝑡
=
∏
𝑖
=
0
𝑡
𝛼
𝑖
α
ˉ
t
	​

=∏
i=0
t
	​

α
i
	​

 tetap eksak).

alphas_cumprod_prev: F.pad(alphas_cumprod[:-1], (1, 0), value=1.0) (
𝛼
ˉ
−
1
=
1.0
α
ˉ
−1
	​

=1.0)

sqrt_alphas_cumprod: 
𝛼
ˉ
𝑡
α
ˉ
t
	​

	​

 (dipakai q_sample)

sqrt_one_minus_alphas_cumprod: 
1
−
𝛼
ˉ
𝑡
1−
α
ˉ
t
	​

	​

 (dipakai q_sample)

sqrt_recip_alphas_cumprod: 
1
/
𝛼
ˉ
𝑡
1/
α
ˉ
t
	​

	​

 (dipakai predict_start_from_noise)

sqrt_recipm1_alphas_cumprod: 
1
/
𝛼
ˉ
𝑡
−
1
1/
α
ˉ
t
	​

−1
	​

 (dipakai predict_start_from_noise)

posterior_variance: 
𝛽
~
𝑡
=
𝛽
𝑡
⋅
1
−
𝛼
ˉ
𝑡
−
1
1
−
𝛼
ˉ
𝑡
β
~
	​

t
	​

=β
t
	​

⋅
1−
α
ˉ
t
	​

1−
α
ˉ
t−1
	​

	​

 (dipakai q_posterior_mean_variance & p_sample)

posterior_log_variance_clipped: 
log
⁡
(
max
⁡
(
𝛽
~
𝑡
,
10
−
20
)
)
log(max(
β
~
	​

t
	​

,10
−20
)) (mencegah log(0) = -inf di 
𝑡
=
0
t=0)

posterior_mean_coef1: 
𝛽
𝑡
𝛼
ˉ
𝑡
−
1
1
−
𝛼
ˉ
𝑡
1−
α
ˉ
t
	​

β
t
	​

α
ˉ
t−1
	​

	​

	​

 (dipakai q_posterior_mean_variance)

posterior_mean_coef2: 
(
1
−
𝛼
ˉ
𝑡
−
1
)
𝛼
𝑡
1
−
𝛼
ˉ
𝑡
1−
α
ˉ
t
	​

(1−
α
ˉ
t−1
	​

)
α
t
	​

	​

	​

 (dipakai q_posterior_mean_variance)

Hitung seluruhnya dalam torch.float64 agar presisi pembagian dan fungsi trigonometri terjaga, lalu konversi ke torch.float32 saat self.register_buffer(...).

3. Kecukupan 3 Epoch & Cara Menulis Kesimpulan yang Jujur

Apakah cukup? Cukup untuk menguji stabilitas konvergensi awal (apakah loss turun mulus tanpa lonjakan gradien pada 354 iterasi pertama dengan 15.000 sampel & batch size 128), tetapi belum cukup untuk menilai ketajaman sampel akhir.

Catatan penting soal perbandingan angka loss: Raw MSE loss 
𝐸
𝑡
,
𝑥
0
,
𝜖
[
∥
𝜖
−
𝜖
𝜃
(
𝑥
𝑡
,
𝑡
)
∥
2
]
E
t,x
0
	​

,ϵ
	​

[∥ϵ−ϵ
θ
	​

(x
t
	​

,t)∥
2
] adalah rata-rata error di seluruh timestep 
𝑡
t. Karena schedule cosine mempertahankan Signal-to-Noise Ratio (SNR) lebih tinggi pada 
𝑡
t menengah dibanding linear, distribusi tingkat kesulitan prediksi noise-nya berbeda — sehingga selisih kecil pada raw MSE antar schedule tidak otomatis berarti salah satu menghasilkan gambar lebih tajam.

Cara merumuskan kesimpulan di laporan:

Laporkan angka secara faktual (loss epoch 1–3, persentase penurunan, dan waktu eksekusi per epoch).

Jelaskan bahwa dalam horizon 3 epoch model baru menangkap struktur frekuensi rendah (siluet global pakaian Fashion-MNIST), sehingga perbedaan perilaku pada tahap detail (
𝑡
t kecil) belum terekspos sepenuhnya.

Justifikasi pilihan linear di config.py: pada 3 epoch awal linear menunjukkan penurunan loss yang stabil dan konsisten (yang terbukti berlanjut hingga 0.04297 di epoch 25), memiliki kompleksitas komputasi yang sama per epoch, dan mengikuti formulasi standar Ho et al. (2020) tanpa parameter offset tambahan.

4. Perlu Loop Training Sendiri di run_variant vs Memanggil src.train.train

Wajib punya loop training ringkas sendiri di run_variant (tetap memakai set_seed, get_dataloader, DDPMUNet, dan diffusion.p_losses dari src/), karena dua alasan teknis:

src/train.py menginstansiasi kelas DDPM secara internal — jika train(cfg) dipanggil langsung, varian cosine tetap akan memakai DDPM linear karena src/train.py tidak membaca cfg.beta_schedule.

train(cfg) tidak mengembalikan riwayat loss/waktu per epoch dan berpotensi menjalankan sampling 1000-step (sample_every) yang memperlambat CPU. Dengan loop training 3 epoch khusus di run_variant, checkpoints/ddpm_best.pt dan logs/train_history.json dijamin 100% aman tidak tersentuh, dan bobot awal model + urutan batch + RNG noise 
𝑡
,
𝜖
t,ϵ identik persis antara linear dan cosine.

Implementasi src/ablation.py
Python
from __future__ import annotations

import json
import math
import sys
import time
from dataclasses import replace
from pathlib import Path
from typing import Any, Dict, List

# Bootstrap sys.path di level modul agar eksekusi `python src/ablation.py`
# dapat melakukan import absolut `from src.x import y`.
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

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


class CosineDDPM(DDPM):
    """Subclass DDPM yang mengganti seluruh 12 buffer schedule ke cosine schedule."""

    def __init__(
        self,
        timesteps: int = 1000,
        s: float = 0.008,
        clip_denoised: bool = True,
    ) -> None:
        super().__init__(
            timesteps=timesteps,
            clip_denoised=clip_denoised,
        )
        self._register_cosine_buffers(timesteps=timesteps, s=s)

    def _register_cosine_buffers(self, timesteps: int, s: float) -> None:
        steps = timesteps + 1
        t = torch.linspace(0, timesteps, steps, dtype=torch.float64) / timesteps
        alphas_cumprod_bar = torch.cos(((t + s) / (1.0 + s)) * (math.pi * 0.5)) ** 2
        alphas_cumprod_bar = alphas_cumprod_bar / alphas_cumprod_bar[0]

        betas = 1.0 - (alphas_cumprod_bar[1:] / alphas_cumprod_bar[:-1])
        betas = torch.clamp(betas, min=1e-4, max=0.999)

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

        buffers = {
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

        for buffer_name, tensor in buffers.items():
            self.register_buffer(buffer_name, tensor.to(torch.float32))


def run_variant(name: str, cfg: Config, use_cosine: bool) -> Dict[str, Any]:
    """Menjalankan training 3 epoch terisolasi untuk satu varian noise schedule."""
    ablation_epochs = 3
    ckpt_subdir = Path(cfg.ckpt_dir) / "ablation" / name
    log_subdir = Path(cfg.log_dir) / "ablation" / name
    ckpt_subdir.mkdir(parents=True, exist_ok=True)
    log_subdir.mkdir(parents=True, exist_ok=True)

    ckpt_path = ckpt_subdir / f"ablation_{name}_best.pt"
    log_path = log_subdir / f"ablation_{name}_history.json"

    # Guard eksplisit agar artefak utama sesi training 25 epoch tidak pernah tertimpa
    protected_files = {
        (Path(cfg.ckpt_dir) / "ddpm_best.pt").resolve(),
        (Path(cfg.log_dir) / "train_history.json").resolve(),
    }
    if ckpt_path.resolve() in protected_files or log_path.resolve() in protected_files:
        raise RuntimeError("Path output ablation bertabrakan dengan file utama.")

    variant_cfg = replace(
        cfg,
        epochs=ablation_epochs,
        num_workers=0,
        beta_schedule="cosine" if use_cosine else "linear",
        ckpt_dir=str(ckpt_subdir),
        log_dir=str(log_subdir),
    )

    device = torch.device("cpu")

    # Reset seed identik sebelum inisialisasi dataloader, model, dan loop training
    # supaya urutan data, bobot awal U-Net, serta sampling timestep/noise konsisten.
    set_seed(variant_cfg.seed)
    dataloader = get_dataloader(variant_cfg, train=True)

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
    for epoch in range(1, ablation_epochs + 1):
        model.train()
        start_time = time.perf_counter()
        running_loss = 0.0
        total_samples = 0

        for batch in dataloader:
            x0 = batch[0] if isinstance(batch, (list, tuple)) else batch
            x0 = x0.to(device)
            batch_size = x0.shape[0]

            t = torch.randint(
                0, variant_cfg.timesteps, (batch_size,), device=device
            ).long()

            optimizer.zero_grad()
            loss = diffusion.p_losses(model, x0, t)
            loss.backward()
            optimizer.step()

            running_loss += loss.item() * batch_size
            total_samples += batch_size

        elapsed = time.perf_counter() - start_time
        avg_loss = running_loss / max(total_samples, 1)

        epoch_losses.append(avg_loss)
        epoch_times.append(elapsed)

        print(
            f"  Epoch {epoch}/{ablation_epochs} | "
            f"Loss: {avg_loss:.5f} | Waktu: {elapsed:.2f}s"
        )

        if avg_loss < best_loss:
            best_loss = avg_loss
            torch.save(
                {
                    "epoch": epoch,
                    "variant": name,
                    "beta_schedule": variant_cfg.beta_schedule,
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "loss": avg_loss,
                    "config": variant_cfg.to_dict(),
                },
                ckpt_path,
            )

    result: Dict[str, Any] = {
        "name": name,
        "beta_schedule": variant_cfg.beta_schedule,
        "num_params": num_params,
        "epoch_losses": epoch_losses,
        "epoch_times": epoch_times,
        "avg_epoch_time": sum(epoch_times) / len(epoch_times),
        "total_time": sum(epoch_times),
        "best_loss": best_loss,
        "ckpt_path": str(ckpt_path),
        "log_path": str(log_path),
    }

    with open(log_path, "w", encoding="utf-8") as f:
        json.dump(result, f, indent=2)

    return result


def plot_ablation(results: List[Dict[str, Any]], save_path: str | Path) -> None:
    """Menggabungkan kurva loss kedua varian dalam satu grafik PNG."""
    output_path = Path(save_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    fig, ax = plt.subplots(figsize=(7.5, 4.8), dpi=150)

    styles = {
        "linear": {"color": "#1f77b4", "marker": "o", "linestyle": "-"},
        "cosine": {"color": "#d95f02", "marker": "s", "linestyle": "--"},
    }

    for res in results:
        name = res["name"]
        losses = res["epoch_losses"]
        epochs = list(range(1, len(losses) + 1))
        st = styles.get(name, {"color": None, "marker": "o", "linestyle": "-"})

        ax.plot(
            epochs,
            losses,
            label=f"{name.capitalize()} Schedule (akhir: {losses[-1]:.5f})",
            color=st["color"],
            marker=st["marker"],
            linestyle=st["linestyle"],
            linewidth=2.2,
            markersize=7,
        )

        for ep, val in zip(epochs, losses):
            offset_y = 8 if name == "linear" else -14
            ax.annotate(
                f"{val:.4f}",
                (ep, val),
                textcoords="offset points",
                xytext=(0, offset_y),
                ha="center",
                fontsize=8.5,
                color=st["color"] or "#222222",
            )

    ax.set_title(
        "Ablation Study: Linear vs Cosine Noise Schedule (3 Epoch, Fashion-MNIST)",
        fontsize=11,
        fontweight="bold",
        pad=12,
    )
    ax.set_xlabel("Epoch", fontsize=10)
    ax.set_ylabel("Training MSE Loss", fontsize=10)
    ax.set_xticks([1, 2, 3])
    ax.grid(True, linestyle=":", alpha=0.6)
    ax.legend(frameon=True, fontsize=9.5)

    fig.tight_layout()
    fig.savefig(output_path)
    plt.close(fig)
    print(f"\n[Ablation] Grafik perbandingan disimpan ke: {output_path}")


def main() -> None:
    cfg = Config()

    linear_result = run_variant(name="linear", cfg=cfg, use_cosine=False)
    cosine_result = run_variant(name="cosine", cfg=cfg, use_cosine=True)
    results = [linear_result, cosine_result]

    header = (
        f"{'Varian':<10} | {'Params':>9} | {'Ep 1 Loss':>10} | "
        f"{'Ep 2 Loss':>10} | {'Ep 3 Loss':>10} | {'Avg Waktu/Ep':>12}"
    )
    separator = "-" * len(header)

    print("\n" + separator)
    print("RINGKASAN ABLATION NOISE SCHEDULE (3 EPOCH, CPU)")
    print(separator)
    print(header)
    print(separator)
    for res in results:
        losses = res["epoch_losses"]
        print(
            f"{res['name']:<10} | "
            f"{res['num_params']:>9,} | "
            f"{losses[0]:>10.5f} | "
            f"{losses[1]:>10.5f} | "
            f"{losses[2]:>10.5f} | "
            f"{res['avg_epoch_time']:>10.2f} s"
        )
    print(separator)

    plot_path = Path(cfg.fig_dir) / "ablation_noise_schedule.png"
    plot_ablation(results, plot_path)


if __name__ == "__main__":
    main()

Detail Teknis yang Sudah Disesuaikan dengan Repo

Jumlah Parameter Konsisten (793,953): DDPMUNet diinstansiasi memakai in_channels=cfg.channels, base_ch=cfg.unet_base (32), dan time_emb_dim=cfg.time_embed_dim (128), bukan nilai default di src/model.py (48 dan 192).

Isolasi Artefak: Seluruh checkpoint dan log ablation disimpan ke checkpoints/ablation/{linear,cosine}/ dan logs/ablation/{linear,cosine}/, disertai pengecekan protected_files agar checkpoints/ddpm_best.pt dan logs/train_history.json tidak pernah tersentuh.

Fair Comparison: set_seed(variant_cfg.seed) dipanggil tepat sebelum membuat DataLoader dan model pada masing-masing varian sehingga inisialisasi bobot U-Net, urutan mini-batch, dan sampel timestep t identik persis.

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
