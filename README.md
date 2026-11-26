# Mini DDPM pada Fashion-MNIST: Proyek Vibe Coding Generative AI from Scratch

Repositori ini memuat hasil Proyek 1 mata kuliah CAK4KBB3. Isinya sebuah implementasi
*Denoising Diffusion Probabilistic Model* (DDPM) berskala kecil yang dibangun dari nol
melalui metode *vibe coding*, yaitu penulisan kode secara bertahap lewat percakapan
dengan model bahasa besar, bukan dengan menyalin blok kode dari tutorial.

| | |
|---|---|
| Arsitektur | DDPM (Ho et al., 2020) dengan U-Net minimalis |
| Dataset | Fashion-MNIST, 28×28 piksel, grayscale, subset 15.000 citra |
| Perangkat | CPU saja (tanpa CUDA); diuji pada Ryzen 7 8845HS / 16 GB RAM |
| Anggota | 2 orang, dengan penggabungan empat job desk |

## Status pengerjaan

| Fase | Lingkup | Status |
|---|---|---|
| 0 | Fondasi repositori, log, pembagian tugas | selesai |
| 1 | Pipeline data (`data.py`): loader, normalisasi, `check_batch` | selesai, 117 batch, `STATUS: LULUS` |
| 2 | Arsitektur (`diffusion.py`, `model.py`): noise schedule dan U-Net | selesai, U-Net 793.953 parameter (`base_ch=32`) |
| 3 | Training loop dan pencatatan metrik (`train.py`) | selesai, 25 epoch, loss 0,168 turun ke 0,043 |
| 4 | Sampling dan grid progres antar-epoch (`sample.py`) | selesai, grid 25 epoch |
| 5 | Evaluasi kuantitatif (`evaluate.py`): MMD dan Wasserstein-1 dengan lantai Real vs Real | selesai, MMD 0,038898 (5,4× floor) |
| 6 | Ablasi noise schedule (`ablation.py`): linear vs cosine | selesai, 2 epoch per varian, artefak terisolasi di `checkpoints/ablation/` |
| 7 | Figur dan kompilasi log serta laporan akhir | selesai, `figures.py`, `figures_ablation.py`, `LOG-INTERAKSI.md` terisi 9 sesi, laporan di `deliverable/` |
| 8 | Verifikasi determinisme sampling (batch N=6 vs N=128) | selesai, grid reproducible bit-exact; batch besar tidak identik (selisih maksimum 0,99) |

Seluruh modul di dalam `src/` dibangun lewat sesi percakapan dengan Gemini. Kronologi
lengkapnya, termasuk prompt yang dikirim, error yang benar-benar muncul di konsol, dan
cara perbaikannya, tercatat di `LOG-INTERAKSI.md`. Rencana kerja per sesi ada di
`RENCANA.md`, sedangkan aturan kolaborasi git ada di `TEAM-GIT.md`.

## Struktur repositori

```
MiniDDPM-FashionMNIST/
├── src/
│   ├── __init__.py       penanda package
│   ├── config.py         seluruh hyperparameter terkumpul di satu tempat
│   ├── data.py           Fase 1, dataloader dan normalisasi
│   ├── diffusion.py      Fase 2, noise schedule, proses forward dan reverse
│   ├── model.py          Fase 2, U-Net dan timestep embedding
│   ├── train.py          Fase 3, training loop dan logging metrik
│   ├── sample.py         Fase 4, sampling dari checkpoint
│   ├── plot_loss.py      Fase 4, kurva loss dari train_history.json
│   ├── evaluate.py       Fase 5, metrik MMD dan Wasserstein-1 + lantai Real vs Real
│   ├── ablation.py       Fase 6, ablasi noise schedule linear vs cosine
│   ├── figures.py        Fase 7, grid data asli vs hasil sampling
│   ├── figures_ablation.py  Fase 7, grid sampel linear vs cosine
│   └── verify_seed_n128.py   verifikasi: batch N=6 vs N=128 tidak bit-exact
├── colab/
│   └── jalankan_di_colab.ipynb   notebook eksekusi pada GPU T4
├── logs/                 metrik training per eksperimen (JSON)
│   └── ablation/         riwayat loss dua varian noise schedule
├── report/figures/       kurva loss, grid hasil sampling, figur evaluasi dan ablasi
├── checkpoints/          bobot model (.pt), tidak diikutsertakan ke git
│   └── ablation/         checkpoint linear dan cosine (terpisah dari model utama)
├── data/                 Fashion-MNIST terunduh, tidak diikutsertakan ke git
├── notebooks/            notebook eksperimen bebas
├── LOG-INTERAKSI.md      rekam jejak percakapan dengan LLM
├── TEAM-GIT.md           aturan kolaborasi dan pembagian commit
├── RENCANA.md            rencana kerja per sesi
├── BRIEF-MIA.md          panduan kerja untuk anggota kedua
└── README.md
```

## Pembagian tugas

Panduan proyek menyebut empat peran, sedangkan kelompok ini beranggotakan dua orang.
Peran tersebut digabung sebagai berikut.

| Anggota | Peran gabungan | Tanggung jawab |
|---|---|---|
| A | Prompt Engineer & LLM Navigator, Training Loop & Hyperparameter Tuning | Menyusun strategi prompt dan urutan pembangunan modul, memandu LLM untuk `model.py`, `diffusion.py`, dan `train.py`, menjalankan eksperimen hyperparameter, mencatat metrik |
| B | Data Pipeline & Preprocessing, Lead Documentarian & Error Handler | Memverifikasi pipeline data (menjalankan `data.py` sampai `STATUS: LULUS`, memeriksa shape dan rentang batch), memandu LLM untuk `evaluate.py`, `ablation.py`, `figures.py`, `figures_ablation.py`, mengompilasi `LOG-INTERAKSI.md` sesi 1–9, menangkap error lalu merumuskan prompt perbaikan |

Catatan pembagian: judul peran *Data Pipeline* ada di anggota B, tetapi kode `src/data.py` dihasilkan pada sesi 1–2 dengan operator A. Yang dapat diverifikasi dari repo adalah operator tiap sesi dan penulis tiap commit (lihat `LAMPIRAN-COMMIT.md`), dan laporan mencatatnya apa adanya.

Penggabungan ini beralasan. Peran yang memandu arsitektur membutuhkan umpan balik cepat
dari hasil training, sehingga wajar disatukan dengan pemilik training loop. Di sisi lain,
pekerjaan pipeline data dan pendokumentasian error bermuara pada artefak yang sama: setiap
error bentuk tensor atau distribusi data langsung menjadi entri di log.

## Konfigurasi

U-Net yang dipakai berukuran 793.953 parameter (`base_ch=32`), menerima input 28×28×1.
Benchmark pada CPU memberi angka berikut.

| Batch | Waktu per step | Epoch pada 15.000 citra |
|---|---|---|
| 64 | 0,394 s | sekitar 1,5 menit |
| 128 | 0,640 s | sekitar 1,3 menit |

Konfigurasi yang akhirnya dipakai:

```yaml
dataset      : FashionMNIST
subset       : 15000          # dari 60000 data latih
image_size   : 28
channels     : 1
batch_size   : 128
epochs       : 20-25          # sekitar 30 menit total di CPU
lr           : 2e-4           # Adam
timesteps    : 1000
beta_schedule: linear (1e-4 sampai 0.02)
unet_base    : 32
time_embed   : 128            # sinusoidal, dihitung manual
seed         : 42
```

Hasil training penuh selama 25 epoch pada CPU:

| Epoch | Loss |
|---|---|
| 1 | 0,16831 |
| 5 | 0,05708 |
| 10 | 0,04809 |
| 17 | 0,04332 |
| 24 (terbaik) | 0,04297 |
| 25 | 0,04307 |

Loss turun 74,4 persen dan mulai mendatar di sekitar 0,043. Satu epoch memakan waktu
80 sampai 100 detik, sehingga totalnya sekitar 35 menit. Grafiknya tersedia di
`report/figures/loss_curve.png` dan `report/figures/progression_epochs_1_to_25.png`.

## Aturan pengerjaan

Enam aturan berikut bukan formalitas administratif, melainkan bagian yang dinilai.

1. Satu sesi LLM menghasilkan satu entri di `LOG-INTERAKSI.md`, dan entri itu diisi
   langsung setelah sesi berakhir. Rekonstruksi di akhir proyek akan terlihat.
2. Setiap entri memuat tautan percakapan yang masih dapat dibuka.
3. Setiap prompt bersifat spesifik dan disertai alasan pemilihannya. Prompt generik
   seperti "buatkan model diffusion" menurunkan nilai orisinalitas.
4. Error disalin utuh dari konsol, dikonsultasikan ke LLM, lalu dianalisis. Error yang
   disembunyikan atau dipalsukan lebih merugikan daripada error yang jujur dicatat.
5. Commit dibuat kecil dan sering dengan pesan deskriptif, bukan satu commit besar di
   akhir pengerjaan.
6. Penyalinan blok kode utuh dari GitHub, blog, atau tutorial tidak dilakukan.
   Dokumentasi resmi PyTorch hanya dipakai sebagai rujukan fungsi dasar.

## Cara menjalankan

```bash
# 1. environment
python -m pip install -r requirements.txt

# 2. Fase 1, verifikasi pipeline data
python src/data.py

# 3. Fase 2, verifikasi diffusion dan model
python src/diffusion.py
python src/model.py

# 4. Fase 3, training (hyperparameter dibaca dari src/config.py)
python src/train.py

# 5. Fase 4, sampling dan kurva loss
python src/sample.py
python src/plot_loss.py

# 6. Fase 5, evaluasi kuantitatif (butuh checkpoints/ddpm_best.pt, sekitar 53 menit di CPU)
python src/evaluate.py

# 7. Fase 6, ablasi noise schedule (sekitar 30 menit per varian di CPU)
python src/ablation.py

# 8. Fase 7, figur laporan
python src/figures.py            # -> report/figures/real_vs_generated.png
python src/figures_ablation.py   # -> report/figures/ablation_schedule_grid.png (sekitar 208 detik)

# 9. Verifikasi determinisme sampling terhadap ukuran batch (sekitar 25 menit di CPU)
python src/verify_seed_n128.py   # -> report/verify_seed_n128.json
```

Pada Google Colab dengan GPU T4, buka `colab/jalankan_di_colab.ipynb` lalu jalankan sel
secara berurutan.

## Kesesuaian dengan rubrik

| Kriteria | Bobot | Bagaimana proyek ini menjawabnya |
|---|---|---|
| Kualitas dan orisinalitas vibe coding | 30% | Prompt disusun bertahap per modul, masing-masing disertai alasan arsitektural tertulis |
| Log Interaksi dan dokumentasi LLM | 25% | `LOG-INTERAKSI.md` disusun kronologis dari inisiasi sampai debugging, dengan tautan percakapan |
| Fungsionalitas model dan pelatihan | 25% | Training 25 epoch selesai tanpa error; metrik tersimpan di `logs/train_history.json`, ditambah evaluasi kuantitatif (`evaluate.py`) dan ablasi noise schedule (`ablation.py`) yang keduanya dapat dijalankan ulang dari CPU |
| Kerjasama tim dan manajemen integrasi | 20% | Riwayat commit per anggota, job desk tertulis, dan pembagian modul yang jelas |
