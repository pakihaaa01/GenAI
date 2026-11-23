# Mini DDPM — Fashion-MNIST (Proyek 1: Vibe Coding Generative AI from Scratch)

Proyek kelompok mata kuliah **CAK4KBB3** — membangun *Mini Diffusion Model* (DDPM) dari nol
menggunakan metode **Vibe Coding** (kolaborasi aktif dengan LLM, tanpa copy-paste tutorial).

- **Arsitektur:** Denoising Diffusion Probabilistic Model (DDPM, Ho et al. 2020) — UNet minimalis
- **Dataset:** Fashion-MNIST (28×28, grayscale), subset 15.000 gambar
- **Hardware:** CPU-only (tanpa CUDA). Sudah di-benchmark pada Ryzen 7 8845HS / 16 GB RAM
- **Anggota:** 2 orang (penggabungan 4 job desk — lihat bagian Pembagian Tugas)

---

## Status Proyek

| Fase | Isi | Status |
|---|---|---|
| 0 | Fondasi: repo, log, pembagian tugas | ✅ selesai |
| 1 | Data pipeline (`data.py`: loader, normalisasi, `check_batch`) | ✅ selesai — 117 batch, `STATUS: LULUS` |
| 2 | Arsitektur (`diffusion.py` + `model.py`: schedule, UNet) | ✅ selesai — UNet 1.782.289 param |
| 3 | Training loop + logging metrik (`train.py`) | ✅ selesai — smoke test lulus |
| 4 | Sampling & grid progres (`sample.py`) | ✅ selesai — grid progres antar-epoch |
| 5 | Kompilasi Log Interaksi + laporan akhir | 🔨 `LOG-INTERAKSI.md` 5 sesi terisi |

> **Catatan:** seluruh modul di `src/` dibangun lewat sesi vibe coding dengan Gemini.
> Kronologi lengkap — prompt, error asli, dan perbaikannya — ada di `LOG-INTERAKSI.md`.
> Rencana kerja ada di `RENCANA.md`, panduan kolaborasi git di `TEAM-GIT.md`.

---

## Struktur Folder

```
MiniDDPM-FashionMNIST/
├── src/                  # kode sumber (dibangun lewat vibe coding)
│   ├── __init__.py       # penanda package
│   ├── config.py         # semua hyperparameter di satu tempat
│   ├── data.py           # Fase 1 — dataloader + normalisasi
│   ├── diffusion.py      # Fase 2 — noise schedule, forward/reverse process
│   ├── model.py          # Fase 2 — UNet + timestep embedding
│   ├── train.py          # Fase 3 — training loop, custom loss, logging
│   └── sample.py         # Fase 4 — sampling dari checkpoint
├── colab/
│   └── jalankan_di_colab.ipynb   # notebook eksekusi di Colab GPU T4
├── logs/                 # metrik training (JSON) per eksperimen
├── report/figures/       # grafik loss, grid gambar hasil sampling
├── checkpoints/          # bobot model tersimpan (.pt)  [git-ignored]
├── data/                 # Fashion-MNIST terunduh            [git-ignored]
├── notebooks/            # notebook eksperimen bebas
├── LOG-INTERAKSI.md      # ⭐ rekam jejak chat LLM (25% nilai)
├── TEAM-GIT.md           # aturan kolaborasi & pembagian commit
├── RENCANA.md            # rencana kerja per sesi
└── README.md
```

---

## Pembagian Tugas (2 anggota)

Panduan asli mengasumsikan 4 anggota. Karena kelompok ini 2 orang, job desk digabung:

| Anggota | Job desk gabungan | Tanggung jawab konkret |
|---|---|---|
| **A** | Anggota 1 (Prompt Engineer & LLM Navigator) + Anggota 3 (Training Loop & Hyperparameter Tuning) | Menyusun system prompt & strategi arsitektur, memandu LLM untuk `model.py` + `diffusion.py` + `train.py`, menjalankan eksperimen LR/batch size/epoch, mencatat metrik |
| **B** | Anggota 2 (Data Pipeline & Preprocessing) + Anggota 4 (Lead Documentarian & Error Handler) | Menyiapkan dataset & `data.py`, normalisasi ke [-1,1], visualisasi batch, mengompilasi `LOG-INTERAKSI.md`, menangkap error → prompt perbaikan, menyusun laporan akhir |

**Justifikasi penggabungan (tulis di laporan):** pemandu prompt arsitektur butuh umpan balik cepat
dari hasil training, sehingga digabung dengan training loop; data pipeline dan dokumentasi error
saling terkait karena seluruh error bentuk tensor/data bermuara di log yang sama.

> Isi nama asli anggota di sini sebelum submit:
> - Anggota A: `[nama]`
> - Anggota B: `[nama]`

---

## Konfigurasi Final (hasil benchmark nyata di CPU)

Model UNet 1,48 M parameter, 28×28×1:

| batch | waktu/step | epoch @15k gambar |
|---|---|---|
| 64 | 0,394 s | 1,5 menit |
| **128** | 0,640 s | **1,3 menit** |

Setting awal yang dipakai:

```yaml
dataset      : FashionMNIST
subset       : 15000          # dari 60000 train
image_size   : 28
channels     : 1
batch_size   : 128
epochs       : 20-25          # ± 30 menit total di CPU
lr           : 2e-4           # Adam
timesteps    : 1000           # T
beta_schedule: linear (1e-4 → 0.02)
unet_base    : 32
time_embed   : nn.Embedding(1000, 128)   # bukan sinusoidal manual
seed         : 42
```

---

## Aturan Kerja (wajib, menentukan nilai)

1. **Satu sesi LLM = satu entri** di `LOG-INTERAKSI.md`. Diisi **langsung** setelah sesi, bukan direkonstruksi di akhir.
2. Setiap entri wajib punya **link sesi chat** yang masih aktif (share link / ekspor MD/PDF).
3. Setiap prompt harus **spesifik** dan disertai **"Alasan Khusus"** yang jujur. Prompt generik ("buatkan model diffusion") menurunkan nilai orisinalitas (30%).
4. Error → salin **utuh** dari konsol → tanya LLM → catat analisis & pelajaran.
5. Commit kecil dan sering, pesan commit deskriptif, jangan satu commit raksasa di akhir (20% nilai dari jejak kolaborasi).
6. Dilarang copy-paste blok kode utuh dari GitHub/blog. Dokumentasi resmi PyTorch hanya untuk referensi fungsi dasar.

---

## Cara Menjalankan

```bash
# 1. environment
python -m pip install -r requirements.txt

# 2. Fase 1 — cek data pipeline
python src/data.py

# 3. Fase 2 — cek diffusion & model
python src/diffusion.py
python src/model.py

# 4. Fase 3 — training (semua hyperparameter dari src/config.py)
python src/train.py

# 5. Fase 4 — sampling + grid progres antar-epoch
python src/sample.py
```

Di Colab (GPU T4): buka `colab/jalankan_di_colab.ipynb`, jalankan sel berurutan.

---

## Rubrik (total 100)

| Kriteria | Bobot | Cara proyek ini menjawab |
|---|---|---|
| Kualitas & Orisinalitas Vibe Coding | 30% | Prompt bertahap per-modul, tiap prompt punya alasan arsitektural tertulis |
| Log Interaksi & Dokumentasi LLM | 25% | `LOG-INTERAKSI.md` kronologis: inisiasi → error → optimasi, lengkap dengan link |
| Fungsionalitas Model & Pelatihan | 25% | Training jalan tanpa error, `logs/metrics.csv` → grafik loss, grid sampel per epoch |
| Kerjasama Tim & Manajemen Integrasi | 20% | Git commit per anggota, job desk tertulis, latihan presentasi silang |
