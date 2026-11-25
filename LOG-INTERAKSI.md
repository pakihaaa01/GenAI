# LOG INTERAKSI LLM — MINI DDPM FASHION-MNIST

> **Aturan pengisian:** satu sesi dengan LLM = satu entri. Diisi langsung setelah sesi selesai.
> Setiap entri wajib punya tautan sesi chat yang masih aktif.
> Jangan menulis ulang di akhir proyek — kronologi asli yang dinilai.

**Anggota:**
- Faqih Abdullah — Prompt Engineer & LLM Navigator + Training Loop & Hyperparameter Tuning
- `[Nama B]` — Data Pipeline & Preprocessing + Lead Documentarian & Error Handler

**Model:** Mini DDPM (Denoising Diffusion Probabilistic Model) — Fashion-MNIST 28×28, dibangun dari nol tanpa library diffusion siap pakai.
**Akun LLM yang dipakai:** Google Gemini

**Daftar isi sesi:**

| # | Tanggal | Target Modul | Operator | Status |
|---|---|---|---|---|
| 1 | 23/11/2026 | `src/data.py` — `get_dataset` | Faqih | ✅ LULUS (1 error) |
| 2 | 23/11/2026 | `src/data.py` — `get_dataloader` + `check_batch` | Faqih | ✅ LULUS |
| 3 | 23/11/2026 | `src/diffusion.py` — kelas `DDPM` | Faqih | ✅ LULUS |
| 4 | 23/11/2026 | `src/model.py` — U-Net DDPM | Faqih | ✅ LULUS |
| 5 | 23/11/2026 | `src/train.py` + `src/sample.py` | Faqih | ✅ LULUS (4 error) |

---

## Sesi #1 — 23/11/2026

- **Target Modul:** Data pipeline — fungsi `get_dataset`
- **Operator:** Faqih Abdullah
- **Tautan Sesi Chat:** https://share.gemini.google/dlPvjwM0Z1KQ — satu percakapan dipakai untuk Sesi #1–#3 (prompt ke-1)
- **Model LLM:** Gemini

### 1. TAHAP INISIASI (Meminta Kode Awal)

**Prompt Kelompok:**

> "halo, saya bikin proyek Mini DDPM dari nol pakai PyTorch buat Fashion-MNIST. saya butuh fungsi `get_dataset` dulu.
>
> yang saya bingung:
> 1. kenapa gambar harus dinormalisasi ke rentang [-1, 1] sih? kan biasanya CNN klasifikasi cukup [0, 1] atau pakai mean/std ImageNet. apa bedanya buat diffusion
> 2. saya cuma mau pakai 15.000 gambar dari 60.000 biar training di CPU ga lama. gimana cara ambil subset yang deterministik, jadi kalau saya run ulang indeksnya sama persis
> 3. perlu augmentasi ga? saya lihat banyak implementasi diffusion ga pakai augmentasi sama sekali. kalau saya pakai RandomHorizontalFlip, itu membantu atau malah bikin model belajar distribusi yang salah
>
> tolong tulis fungsinya. saya pakai torchvision.datasets.FashionMNIST."

**Alasan Khusus:**
Normalisasi ke [-1, 1] dipilih karena forward diffusion menambahkan noise Gaussian N(0, I) — agar skala sinyal dan skala noise sebanding, data harus dipusatkan di nol. Kalau data di [0, 1], mean-nya 0,5 sehingga di t kecil noise menenggelamkan sinyal secara asimetris. Subset 15.000 dipilih karena target eksekusi awal di CPU (Ryzen 7 8845HS, tanpa CUDA) — 60.000 gambar memakan ±5× lebih lama per epoch tanpa perbaikan kualitas yang proporsional.

**Output Model:**
Gemini menjelaskan tiga hal: (1) normalisasi [-1, 1] menyamakan skala data dengan noise N(0, I) sehingga `q(x_t|x_0)` tetap seimbang di semua t; (2) subset deterministik dilakukan dengan `torch.Generator().manual_seed(seed)` lalu `torch.randperm`, bukan `random.sample` yang bergantung state global; (3) augmentasi **tidak** dipakai — diffusion model mempelajari distribusi data, dan flip horizontal pada Fashion-MNIST menciptakan distribusi yang tidak ada di data uji (sepatu kiri jadi sepatu kanan).

Kode yang diberikan: `get_dataset(cfg, train=True)` dengan `transforms.Compose([ToTensor(), Normalize((0.5,), (0.5,))])`, lalu `Subset` bila `train=True` dan `subset_size < len(dataset)`.

**Hasil Uji:**
Dijalankan lewat `check_batch` — shape `(128, 1, 28, 28)`, min `-1.0`, max `1.0`, mean `-0.4584`, std `0.6909`, 10 label unik, 117 batch. Sesuai harapan (15.000 / 128 = 117,1875 → 117 karena `drop_last=True`).

### 2. TAHAP PENANGANAN EROR / DEBUGGING

**Pesan Eror dari Notebook:**

```
Traceback (most recent call last):
  File "C:\Users\ASUS\Documents\MiniDDPM-FashionMNIST\src\data.py", line 27, in get_dataset
    root=cfg.data_dir,
AttributeError: 'Config' object has no attribute 'data_dir'
```

**Prompt Perbaikan ke LLM:**

> "error: `AttributeError: 'Config' object has no attribute 'data_dir'`. ternyata config saya field-nya `data_root`, bukan `data_dir`. salah saya ga kasih lihat config-nya.
>
> tapi ada masalah lain yang lebih penting: waktu saya jalanin, datasetnya ke-download dua kali. sekali ke `data/` di root proyek, sekali lagi ke `src/data/`. ternyata karena `root="data"` itu path relatif, jadi tergantung dari mana saya jalankan python-nya. gimana cara bikin root-nya selalu nempel ke root proyek, ga peduli cwd-nya apa"

**Analisis & Solusi Kelompok:**
Dua bug sekaligus. Pertama, mismatch nama field (`data_dir` vs `data_root`) — murni karena konteks config tidak diberikan di prompt awal. Kedua, yang lebih serius: path relatif `"data"` di-resolve terhadap *current working directory*, bukan lokasi file. Karena skrip dijalankan sebagai `python src/data.py` dari root, `"data"` jatuh di root (benar), tapi begitu dijalankan dari dalam `src/`, dataset terunduh lagi ke `src/data/` — 21 MB terbuang dan dataset jadi tidak konsisten.

Solusi dari Gemini: fungsi `_resolve_data_root(cfg)` yang mengubah path relatif jadi absolut terhadap `PROJECT_ROOT` (didefinisikan di `config.py` sebagai `Path(__file__).resolve().parent.parent`), plus `root.mkdir(parents=True, exist_ok=True)` agar folder dibuat sebelum torchvision menulis ke sana.

**Hasil Uji:**
`python src/data.py` → exit 0. Sanity check cetak `STATUS: LULUS`. Folder `src/data/` dihapus manual.

---

## Sesi #2 — 23/11/2026

- **Target Modul:** Data pipeline — `get_dataloader` + `check_batch`
- **Operator:** Faqih Abdullah
- **Tautan Sesi Chat:** https://share.gemini.google/dlPvjwM0Z1KQ — percakapan yang sama dengan Sesi #1 (prompt ke-2 dan ke-3)
- **Model LLM:** Gemini

### 1. TAHAP INISIASI (Meminta Kode Awal)

**Prompt Kelompok:**

> "sekarang saya butuh `get_dataloader` dan fungsi `check_batch` buat sanity check. yang saya bingung:
>
> 1. `num_workers` — saya di Windows. setahu saya num_workers > 0 di Windows suka hang atau BrokenPipeError. bener ga? kalau iya, apa efeknya ke kecepatan training
> 2. `drop_last` — kalau saya set True, 15.000 / 128 = 117,1875 jadi 117 batch, ada 24 gambar yang dibuang tiap epoch. buat diffusion ini masalah ga sih, kan tiap epoch datanya di-shuffle ulang jadi yang dibuang beda-beda
> 3. `shuffle` — di diffusion kan targetnya noise acak, bukan label. jadi apa masih perlu shuffle? apa justru ga perlu karena tiap sampel independen
>
> terus tolong tulis `check_batch` yang ngecek: jumlah batch, shape satu batch, min/max/mean/std piksel, dan label unik. harusnya 117 batch, shape (128,1,28,28), rentang [-1,1], 10 label. print 'STATUS: LULUS' kalau semua lolos, kalau ada yang aneh print 'STATUS: GAGAL' plus daftar masalahnya."

**Alasan Khusus:**
`num_workers=0` dipilih bukan karena malas, tapi karena DataLoader dengan worker proses di Windows memakai `spawn` (bukan `fork`), sehingga setiap worker harus meng-import ulang modul — ini yang memicu hang dan `BrokenPipeError` bila ada kesalahan di level modul. Untuk dataset 21 MB yang sudah di-cache, overhead spawn lebih besar dari keuntungan paralelisasi. `drop_last=True` dipilih karena batch berukuran 24 di akhir epoch membuat statistik loss per-step tidak sebanding dengan batch 128 — grafik loss jadi bergerigi tanpa alasan.

**Output Model:**
`get_dataloader(cfg, train=True)` membungkus `get_dataset` dengan `generator` ber-seed agar urutan batch reproducible, `shuffle=train`, `drop_last=train`, `num_workers` dari config, `pin_memory=False` (training penuh di CPU). `check_batch(loader, expected_batches=117, expected_batch_size=128)` mengumpulkan semua anomali ke list `issues` lalu mencetak status di akhir — bukan `assert` yang berhenti di error pertama.

**Hasil Uji:**
117 batch, shape `(128, 1, 28, 28)`, min `-1.0`, max `1.0`, mean `-0.4584`, std `0.6909`, label `[0..9]`, `STATUS: LULUS`.

### 2. TAHAP PENANGANAN EROR / DEBUGGING

Tidak ada error pada sesi ini — kode langsung jalan.

---

## Sesi #3 — 23/11/2026

- **Target Modul:** `src/diffusion.py` — kelas `DDPM`
- **Operator:** Faqih Abdullah
- **Tautan Sesi Chat:** https://share.gemini.google/dlPvjwM0Z1KQ — percakapan yang sama (prompt ke-4)
- **Model LLM:** Gemini

### 1. TAHAP INISIASI (Meminta Kode Awal)

**Prompt Kelompok:**

> "sekarang bagian paling inti: `src/diffusion.py`. saya mau kelas DDPM dari nol, jangan pakai library diffusion siap pakai. yang saya butuh dan pengen dijelasin alasannya:
>
> 1. noise schedule linear dari 1e-4 sampai 0.02 dengan T=1000. kenapa angka itu? apa jadinya kalau saya pakai 1e-2 sampai 0.2, atau T=100
> 2. `q_sample` — saya lihat ada trik yang bisa lompat langsung dari x_0 ke x_t tanpa iterasi. itu gimana caranya dan kenapa boleh
> 3. `p_losses` — kenapa targetnya noise epsilon, bukan x_0 atau x_{t-1}? apa bedanya secara matematis
> 4. `p_sample` dan `p_sample_loop` — dari mana asal sigma_t? saya lihat ada yang pakai beta_t, ada yang pakai posterior variance. mana yang bener
>
> saya butuh bisa dijelasin di laporan, jadi tolong kasih penjelasan matematis singkat di tiap bagian."

**Alasan Khusus:**
Schedule linear β: 1e-4 → 0.02 adalah schedule dari paper Ho et al. (2020). Dipilih karena dengan T=1000, nilai ᾱ_T = ∏(1-β_t) ≈ 4e-5, artinya di t=T sinyal asli praktis hilang total dan x_T benar-benar mendekati N(0, I) — prasyarat agar sampling bisa dimulai dari noise murni. Kalau β_end terlalu kecil (mis. 0.005), ᾱ_T masih besar dan x_T belum jadi noise murni, sehingga sampling menghasilkan gambar berkabut. Kalau terlalu besar, training tidak stabil di t kecil. T=1000 dipilih karena itu titik di mana schedule linear mencapai ᾱ_T yang cukup kecil; dengan T=100 angka yang sama tidak cukup.

**Output Model:**
Kelas `DDPM(nn.Module)` dengan semua koefisien di-`register_buffer` (ikut `state_dict`, pindah device otomatis, tapi tidak dilatih):

- `betas = torch.linspace(1e-4, 0.02, 1000)`
- `alphas = 1 - betas`, `alphas_cumprod = cumprod(alphas)`
- `alphas_cumprod_prev` di-`F.pad` dengan 1.0 di depan agar ᾱ_0 = 1
- `posterior_variance = betas * (1 - ᾱ_{t-1}) / (1 - ᾱ_t)`

`q_sample` memakai reparametrization trick: karena x_t = √ᾱ_t·x_0 + √(1-ᾱ_t)·ε dengan ε~N(0,I), dan komposisi dua Gaussian tetap Gaussian, distribusi x_t | x_0 bisa dihitung langsung tanpa menyampel t langkah satu per satu. Ini yang membuat training bisa paralel — satu batch bisa memakai t berbeda-beda sekaligus.

`p_losses` memakai target ε (bukan x_0) karena prediksi x_0 punya bobot yang tidak seragam terhadap t: di t besar, x_t hampir seluruhnya noise, sehingga meminta model memprediksi x_0 berarti meminta ia menebak gambar dari noise — gradiennya besar dan tidak stabil. Target ε membuat loss setara di semua t, yang secara matematis adalah penyederhanaan dari ELBO (L_simple).

`p_sample` memakai posterior variance σ_t² = β_t(1-ᾱ_{t-1})/(1-ᾱ_t), bukan β_t mentah — karena yang disampel adalah q(x_{t-1}|x_t, x_0), bukan forward process.

**Hasil Uji:**
`python src/diffusion.py` → exit 0. Sanity check internal:
```
Shape x_t      : (8, 1, 28, 28)
Nilai MSE Loss : 0.9922
Std di t=0     : 0.9992 (x0: 1.0011)
Std di t=999   : 1.0000
Shape sampel   : (4, 1, 28, 28), Rentang: [-1.00, 1.00]
```
Std di t=0 hampir sama dengan std x_0 (sinyal utuh) dan std di t=999 tepat 1.0 (noise murni) — konfirmasi schedule bekerja benar di kedua ujung.

### 2. TAHAP PENANGANAN EROR / DEBUGGING

Tidak ada error. Kode jalan pada percobaan pertama.

---

## Sesi #4 — 23/11/2026

- **Target Modul:** `src/model.py` — arsitektur U-Net DDPM
- **Operator:** Faqih Abdullah
- **Tautan Sesi Chat:** https://share.gemini.google/x6krWUrccnLp
- **Model LLM:** Gemini

### 1. TAHAP INISIASI (Meminta Kode Awal)

**Prompt Kelompok:**

> "sekarang model U-Net-nya. konteks: gambar 28x28 grayscale 1 channel, target sekitar 1.5 juta parameter biar ringan.
>
> yang saya bingung:
> 1. kenapa harus U-Net sih, ga bisa pakai CNN biasa kayak yang buat klasifikasi? saya ngerti U-Net buat segmentasi tapi ga paham kenapa diffusion butuh bentuk itu
> 2. gimana cara masukin informasi timestep t ke dalam jaringan? saya lihat ada yang pakai sinusoidal position encoding kayak transformer, ada yang pakai nn.Embedding biasa. bedanya apa dan mana yang cocok buat kasus saya
> 3. di 28x28, kalau saya downsampling 3 kali jadi 28-14-7, terus 7 mau di-pooling lagi jadi 3.5 kan ga bisa. gimana ngatasinnya
> 4. self-attention di resolusi rendah: UNet DDPM asli ada itu. buat 28x28 sekecil ini perlu ga? atau buang-buang waktu di CPU
> 5. GroupNorm vs BatchNorm, DDPM selalu pakai GroupNorm. kenapa? pengaruh batch kecil gimana
>
> tolong kasih kodenya yang efisien buat CPU tapi jangan disederhanakan sampai ga mirip U-Net aslinya."

**Alasan Khusus:**
U-Net dipilih karena tugasnya bukan klasifikasi (gambar → label) melainkan **regresi per piksel** (gambar ber-noise → noise). Yang dibutuhkan adalah output beresolusi penuh yang sejajar dengan input. Skip connection menyambungkan feature map resolusi tinggi dari encoder langsung ke decoder, sehingga detail spasial halus (tepi siluet baju, garis sepatu) tidak hilang saat melewati bottleneck. Tanpa skip connection, decoder hanya punya informasi terkompresi dari 7×7 dan hasilnya blur.

Downsampling berhenti di 7×7 (dua kali, bukan tiga) karena 28 → 14 → 7, dan 7 → 3 tidak bisa dibagi dua tanpa padding. Menambah padding ke 32×32 akan mengubah domain data dan memboroskan komputasi untuk piksel yang tidak informatif.

**Output Model:**
`DDPMUNet(in_channels=1, base_ch=32, time_emb_dim=128)` — sesuai `Config()` yang dipakai training — total **793.953** parameter. Komponen:

- `SinusoidalPositionEmbeddings` — encoding posisi sinusoidal (bukan `nn.Embedding`) karena t adalah **nilai kontinu** 0..999 yang punya makna ordinal; `nn.Embedding` memperlakukannya sebagai 1000 kategori diskret tanpa hubungan kedekatan antar t.
- `ResBlock` — GroupNorm → SiLU → Conv2d → (+ time embedding, di-broadcast secara spasial) → GroupNorm → SiLU → Conv2d → residual add.
- `SelfAttention2d` — 4 head, hanya di bottleneck 7×7. Di resolusi 28×28, matriks atensi 784×784 per head memakan memori dan waktu CPU tanpa manfaat — pada resolusi tinggi, konvolusi sudah menangkap konteks lokal dengan cukup baik.
- `Downsample`/`Upsample` — strided conv untuk turun, nearest-neighbor + conv untuk naik (menghindari checkerboard artifact dari `ConvTranspose2d`).
- GroupNorm dipilih karena tidak bergantung pada statistik lintas-batch: dengan batch kecil, BatchNorm menghasilkan statistik berisik, dan saat sampling (batch 16) statistik running mean bisa sangat berbeda dari saat training.

**Hasil Uji:**
```
Total Trainable Parameters: 793,953
Input shape : torch.Size([4, 1, 28, 28])
Output shape: torch.Size([4, 1, 28, 28])
```
Shape input = shape output, sesuai syarat model prediksi noise.

> Catatan revisi: jawaban Gemini menyebut `base_ch=48, time_emb_dim=192` (1.782.289 param), tetapi `src/config.py` memakai `unet_base=32, time_embed_dim=128`. Blok `__main__` model.py semula mencetak angka demo yang berbeda dari model yang benar-benar dilatih. Setelah diseragamkan ke `Config()`, angka sebenarnya = **793.953**. Model tetap berjalan dan training 25 epoch selesai dengan arsitektur ini.

### 2. TAHAP PENANGANAN EROR / DEBUGGING

Tidak ada error pada `model.py` itu sendiri. Namun pada integrasi ke `train.py` muncul `ImportError: cannot import name 'UNet'` — akar masalahnya Gemini mengasumsikan nama kelas `UNet` padahal di `model.py` namanya `DDPMUNet`. Error ini dicatat di Sesi #5.

---

## Sesi #5 — 23/11/2026

- **Target Modul:** `src/train.py` + `src/sample.py`
- **Operator:** Faqih Abdullah
- **Tautan Sesi Chat:**
  - Permintaan awal ("Implementasi Training Loop dan Sampling DDPM"): https://share.gemini.google/EnVN44jYNafs
  - Error #1 `ImportError: cannot import name 'UNet'` ("Perbaikan Kode DDPM FashionMNIST"): https://share.gemini.google/6VbG50aT86E9
  - Error #2 `AttributeError: checkpoint_dir` ("Konfigurasi DDPM FashionMNIST"): https://share.gemini.google/YtTIJQBwQ0gY
  - Error #3 signature lengkap diminta Gemini ("Skrip Pelatihan dan Sampling DDPM"): https://share.gemini.google/miBXTd38leD3
  - Error #4 import path / `No module named 'src'` ("Mengatasi Error Import Python Project"): https://share.gemini.google/GjBM4PxUxflz
- **Model LLM:** Gemini

### 1. TAHAP INISIASI (Meminta Kode Awal)

**Prompt Kelompok:**

> "lanjut, model UNet-nya udah jalan (shape input = output, ±794 ribu parameter). sekarang saya butuh training loop dan sampling.
>
> A. `src/train.py`. yang saya bingung:
> - saya perlu validasi loss ga sih? setahu saya DDPM ga ada validation set yang jelas karena targetnya noise acak, jadi loss-nya ga bisa dibandingin antar-run. bener ga pemahaman saya
> - gimana cara tau training-nya konvergen kalau ga ada akurasi kayak klasifikasi? apa yang harus saya pantau
> - saya mau simpan checkpoint tiap beberapa epoch, tapi di Colab nanti storage-nya bisa hilang. gimana pola yang aman
> - EMA bobot itu perlu ga? saya lihat di implementasi DDPM sering dipakai, tapi nambah kompleksitas
>
> B. `src/sample.py`. cara load checkpoint yang benar, grid gambar pakai matplotlib buat laporan, dan generate sampel di beberapa epoch berbeda biar kelihatan progres belajarnya.
>
> konteks: batch 128, lr 2e-4 Adam, 25 epoch, subset 15000, T=1000, CPU. nanti saya pindah ke Colab pakai GPU T4 jadi jangan hardcode device, bikin auto-detect."

**Alasan Khusus:**
Validasi loss tetap dipakai meski targetnya acak — caranya dengan **mem-fix seed** untuk t dan ε saat evaluasi, sehingga angka val loss deterministik dan bisa dibandingkan antar-epoch. Tanpa itu, val loss berayun hanya karena sampel t berbeda. Ini penting untuk mendeteksi overfitting pada subset 15.000 dengan model 1,78 juta parameter.

Grid progres antar-epoch memakai **initial noise yang identik** (seed di-reset tepat sebelum `p_sample_loop`). Kalau tiap epoch memakai noise awal berbeda, perbedaan hasil tidak bisa diatribusikan ke perbaikan model.

**Output Model:**
`train.py`: AdamW dengan `weight_decay=1e-4`, AMP hanya aktif bila `device.type == "cuda"`, `GradScaler`, `clip_grad_norm_(max_norm=1.0)`, checkpoint tiap epoch ke `cfg.ckpt_dir` + `ddpm_best.pt`, riwayat loss ke `logs/train_history.json`. Diffusion **wajib** `.to(device)` karena buffer-nya `register_buffer`.

`sample.py`: `build_epoch_progression_grid(cfg, target_epochs, num_samples, fixed_seed)` — memindai `ddpm_epoch_*.pt`, memuat tiap checkpoint, sampling dengan seed tetap, menyimpan grid individual + satu grid progres (baris = epoch, kolom = sampel dari noise awal sama) ke `cfg.fig_dir`.

**Hasil Uji:**
Gagal empat kali berturut-turut — lihat bagian debugging. Setelah perbaikan, smoke test lulus:
```
=== SANITY CHECK DATALOADER ===
Jumlah batch      : 117
Shape satu batch  : (128, 1, 28, 28)
Min / Max piksel  : -1.0000 / 1.0000
Mean / Std piksel : -0.4584 / 0.6909
Label unik (10)  : [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]
===============================
STATUS: LULUS
```
Training penuh dijalankan: 117 batch/epoch, ±1,45 it/s → ±80 detik per epoch di CPU, 25 epoch ≈ 33 menit.
Sampling: `Grid progres berhasil disimpan ke: report/figures/progression_epochs_*.png`

### 2. TAHAP PENANGANAN EROR / DEBUGGING

#### Error #1 — `ImportError: cannot import name 'UNet'`

```
Traceback (most recent call last):
  File "C:\Users\ASUS\Documents\MiniDDPM-FashionMNIST\src\train.py", line 12, in <module>
    from model import UNet
ImportError: cannot import name 'UNet' from 'model'
```

**Prompt Perbaikan ke LLM:**

> "aduh kodenya langsung error. ini traceback-nya: [...]. terus setelah saya cek lagi, kayaknya ada beberapa yang ga nyambung sama file saya:
> 1. di `src/model.py` class-nya namanya `DDPMUNet`, bukan `UNet`. dan constructor-nya butuh `(in_channels=1, base_ch=48, time_emb_dim=192)`
> 2. di `src/diffusion.py` class-nya namanya `DDPM`, bukan `GaussianDiffusion`, dan ga ada parameter `device`. method-nya `p_losses(model, x0, t, noise)`, bukan `q_sample(x_start=...)`
> 3. dataset saya FashionMNIST 1 channel 28x28, bukan CIFAR10 3 channel 32x32. dan saya udah punya `src/data.py` dengan `get_dataset`/`get_dataloader` yang udah jalan — kenapa bikin `get_dataloaders` baru lagi?
> 4. `src/config.py` saya isinya dataclass dengan semua hyperparameter, jadi ga perlu argparse
> 5. normalisasi saya `Normalize((0.5,), (0.5,))` buat 1 channel
> 6. di `sample.py` ada `torch.randn((num_samples, 3, img_size, img_size))` — itu 3 channel
>
> tolong perbaiki biar nyambung sama file saya yang udah ada. dan tolong jelasin kenapa kamu tadi bikin ulang semuanya padahal saya udah bilang di prompt sebelumnya kalau data pipeline dan DDPM-nya udah jadi."

**Analisis & Solusi Kelompok:**
Gemini mengakui asumsinya: ia memakai struktur DDPM "standar" (`UNet`, `GaussianDiffusion`, CIFAR-10 3 channel) karena signature file lokal tidak diberikan di prompt awal — hanya disebutkan "sudah jadi". Akibatnya seluruh pipeline ditulis ulang, bukan disambungkan. Pelajaran konkret: **menyebut "sudah jadi" tidak cukup; LLM butuh signature persis.**

**Hasil Uji:** error berikutnya, `AttributeError: 'Config' object has no attribute 'checkpoint_dir'`.

#### Error #2 — `AttributeError: 'Config' object has no attribute 'checkpoint_dir'`

```
Traceback (most recent call last):
  File "C:\Users\ASUS\Documents\MiniDDPM-FashionMNIST\src\train.py", line 27, in train
    os.makedirs(cfg.checkpoint_dir, exist_ok=True)
AttributeError: 'Config' object has no attribute 'checkpoint_dir'
```

**Prompt Perbaikan ke LLM:** isi lengkap `config.py` ditempel verbatim, ditambah daftar eksplisit semua field yang salah nama (`checkpoint_dir`→`ckpt_dir`, `in_channels`→`channels`, `base_ch`→`unet_base`, `time_emb_dim`→`time_embed_dim`, `num_epochs`→`epochs`, `img_size`→`image_size`, `report_dir`→`fig_dir`), plus pertanyaan trade-off `base_ch` 32 vs 48.

**Analisis & Solusi Kelompok:**
Akar masalah sama seperti error #1: LLM menebak nama field. Solusi dari Gemini: `Config` dijadikan *single source of truth* — semua hyperparameter dibaca dari situ, `os.makedirs` dihapus karena `__post_init__` sudah membuat semua folder. Untuk trade-off: `unet_base=32` dipilih (bukan 48) karena FLOPs Conv2d tumbuh kuadratik terhadap jumlah channel — 48/32 = 1,5, sehingga biaya komputasi ±2,25×. Untuk 28×28 grayscale dengan subset 15.000, 32 channel sudah cukup menangkap siluet; 48 baru dipertimbangkan bila hasil 25 epoch masih blur.

Selain itu Gemini menjawab pertanyaan device: `DDPM` mewarisi `nn.Module` dan menyimpan schedule sebagai `register_buffer`, sehingga **`diffusion.to(device)` wajib dipanggil** — kalau hanya `model.to(device)` yang dipindah, buffer tetap di CPU dan muncul `RuntimeError: Expected all tensors to be on the same device`.

**Hasil Uji:** error berikutnya, `ModuleNotFoundError: No module named 'config'`.

#### Error #3 — `ModuleNotFoundError: No module named 'config'`

```
Traceback (most recent call last):
  File "<string>", line 3, in <module>
    from src.train import train
  File "C:\Users\ASUS\Documents\MiniDDPM-FashionMNIST\src\train.py", line 8, in <module>
    from src.data import get_dataloader, check_batch
  File "C:\Users\ASUS\Documents\MiniDDPM-FashionMNIST\src\data.py", line 12, in <module>
    from config import Config, PROJECT_ROOT
ModuleNotFoundError: No module named 'config'
```

**Prompt Perbaikan ke LLM:**

> "error lagi, sekarang soal import path: [...]. jadi masalahnya: `data.py`, `model.py`, `diffusion.py` yang udah ada semuanya pakai import datar (`from config import Config`), karena dulu dijalankan dengan `python src/data.py` dari root. tapi `train.py` dan `sample.py` yang baru kamu kasih pakai `from src.config import Config`. dua gaya ini ga bisa hidup bareng.
>
> 1. mana yang lebih bener buat proyek sekecil ini: import datar atau package style? jelasin konsekuensinya, terutama di Colab yang notebooknya ada di root
> 2. kalau package style, apakah harus nambah `src/__init__.py`? dan apakah `data.py`, `model.py`, `diffusion.py` harus diubah semua
> 3. kalau datar, gimana supaya `python src/train.py` jalan tanpa ModuleNotFoundError, tapi tetap jalan kalau di-import dari notebook Colab di root?
>
> tolong kasih solusi final yang bener-bener jalan di dua skenario."

**Analisis & Solusi Kelompok:**
Ini konflik struktural, bukan typo. Python menambahkan **direktori skrip** ke `sys.path[0]`, bukan root proyek. Jadi `python src/train.py` membuat `src/` ada di path (import datar jalan), tapi `from src.train import train` dari root membuat root yang ada di path (package style jalan). Dua-duanya tidak bisa jalan bersamaan tanpa bootstrap.

Solusi dari Gemini — package style + path bootstrap:
1. Tambah `src/__init__.py` (kosong) agar `src` dikenali eksplisit sebagai package.
2. Seragamkan semua import ke `from src.config import ...` di `data.py`.
3. Tambahkan blok bootstrap di **atas** `train.py` dan `sample.py`, sebelum import `src.*`:
```python
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
```
Dengan ini: `python src/train.py` jalan (bootstrap mendaftarkan root), `python -m src.train` jalan, dan `from src.train import train` dari notebook Colab di root juga jalan tanpa utak-atik.

**Hasil Uji:** import `src.train` dari root berhasil — tapi `python src/data.py` jadi gagal, lihat error #4.

#### Error #4 — `ModuleNotFoundError: No module named 'src'`

```
Traceback (most recent call last):
  File "C:\Users\ASUS\Documents\MiniDDPM-FashionMNIST\src\data.py", line 12, in <module>
    from src.config import Config, PROJECT_ROOT
ModuleNotFoundError: No module named 'src'
```

**Konteks:** setelah `data.py` diseragamkan ke `from src.config import ...`, perintah `python src/data.py` jadi gagal — karena yang masuk `sys.path` adalah `src/`, bukan root, sehingga nama `src` tidak bisa di-resolve.

**Prompt Perbaikan:** dilaporkan ke Gemini bersama error #3 sebagai satu paket pertanyaan ("solusi yang jalan di dua skenario: `python src/data.py` dari root DAN import dari notebook Colab di root"). Jawabannya: bootstrap `sys.path` tidak boleh hanya ada di `train.py`/`sample.py` — setiap modul yang bisa dijalankan langsung sebagai skrip (`__main__`) juga butuh bootstrap yang sama.

**Analisis & Solusi Kelompok:**
Bootstrap dipasang di **atas** `data.py`, sebelum import `src.config`. Blok `sys.path.insert` yang tadinya hanya ada di dalam `if __name__ == "__main__":` dipindah ke level modul — karena import `src.config` terjadi di level modul, jadi bootstrap harus sudah jalan sebelum baris itu dieksekusi. Kalau bootstrap ditaruh di dalam `__main__`, import di atasnya tetap gagal.

**Hasil Uji:** `python src/data.py` → exit 0, `STATUS: LULUS`. `python src/model.py` → 793.953 param. `python src/diffusion.py` → exit 0. `python -c "import src.train"` → jalan. Ketiga jalur (skrip langsung, `-m`, dan import dari root) konsisten.

#### Catatan dependensi — `ModuleNotFoundError: No module named 'tqdm'`

`train.py` mengimpor `tqdm`; paket belum terpasang di environment. Tidak dikonsultasikan ke LLM (bukan masalah kode) — diselesaikan dengan memasang `tqdm` dan menambahkannya ke `requirements.txt`. Dicatat untuk kelengkapan kronologi.

---

## Sesi #6 — 25/11/2026

- **Target Modul:** `src/evaluate.py`
- **Operator:** Mia Rosiana Santoso
- **Tautan Sesi Chat:** https://share.gemini.google/QMM0U3W3ZDe0
- **Model LLM:** Gemini

### 1. TAHAP INISIASI (Meminta Kode Awal)

**Prompt Kelompok:**

> "konteks: aku punya proyek Mini DDPM from scratch untuk Fashion-MNIST 28x28 grayscale, sudah selesai training 25 epoch (loss 0.16831 -> 0.04297, model 793.953 parameter, base_ch=32). sekarang aku mau bikin src/evaluate.py yang ngasih ANGKA, bukan cuma liat gambar. jalan di CPU, Windows, python 3.13, torch 2.14.0+cpu.
>
> yang aku bingung:
>
> 1. FID katanya standar buat evaluasi diffusion. tapi InceptionV3 maunya input 299x299 RGB, punya aku 28x28 grayscale. kalau aku resize ke 299 terus channel-nya direplikasi 3x, FID-nya masih bermakna atau malah menyesatkan? aku takut angkanya kelihatan ilmiah tapi sebenarnya ngukur hal yang salah.
>
> 2. kalau FID nggak cocok, metrik apa yang lebih jujur buat kasus ini? aku kepikiran histogram intensitas piksel antara data asli dan hasil sampling, tapi itu cukup ilmiah nggak sih buat laporan? atau ada metrik lain yang lebih defensible dan masih bisa dihitung di CPU tanpa download model besar?
>
> 3. dari angka yang keluar nanti, gimana cara nyimpulin modelnya bagus atau belum? aku butuh ambang atau pembanding, bukan cuma "angkanya kecil".
>
> signature yang ada di repo (jangan diubah, ini verbatim):
>
> ```
> # src/diffusion.py
> class DDPM(nn.Module):
>     def __init__(self, timesteps: int = 1000, beta_start: float = 1e-4,
>                  beta_end: float = 0.02, clip_denoised: bool = True) -> None
>     def q_sample(self, x0: torch.Tensor, t: torch.Tensor,
>                  noise: torch.Tensor | None = None) -> torch.Tensor
>     def p_losses(self, model: nn.Module, x0: torch.Tensor, t: torch.Tensor,
>                  noise: torch.Tensor | None = None) -> torch.Tensor
>     def p_sample_loop(self, model: nn.Module, shape: tuple[int, int, int, int],
>                       device: torch.device | str = "cpu") -> torch.Tensor
>
> # src/model.py
> class DDPMUNet(nn.Module):
>     def __init__(self, in_channels: int = 1, base_ch: int = 48,
>                  time_emb_dim: int = 192)
>
> # src/sample.py
> def generate_with_fixed_noise(model: DDPMUNet, diffusion: DDPM, num_samples: int,
>                               img_size: int, channels: int, device: torch.device,
>                               fixed_seed: int = 1234) -> torch.Tensor
>     # return sudah didenormalisasi ke [0, 1]
>
> # src/data.py
> def get_dataset(cfg, train: bool = True) -> Dataset
> def get_dataloader(cfg, train: bool = True)
>
> # isi checkpoint (src/train.py): {"epoch", "model_state_dict",
> #   "optimizer_state_dict", "loss", "config"}  -> checkpoints/ddpm_best.pt
>
> # field Config yang ada: data_root, subset_size, image_size=28, channels=1,
> # num_workers=0, unet_base=32, time_embed_dim=128, num_groups=8, timesteps=1000,
> # beta_start=1e-4, beta_end=0.02, beta_schedule="linear", batch_size=128,
> # epochs=25, lr=2e-4, seed=42, sample_every=5, num_sample_images=64,
> # ckpt_dir, log_dir, fig_dir="report/figures"
> ```
>
> batasan:
>
> - cuma bikin src/evaluate.py, jangan ubah file lain (config.py, data.py, diffusion.py, model.py, train.py, sample.py, plot_loss.py sudah diverifikasi)
> - jangan pakai library diffusion siap pakai (diffusers dll)
> - import absolut from src.x import y, dan bootstrap sys.path di level modul (bukan di dalam `__main__`), karena skrip dijalankan `python src/evaluate.py`
> - num_workers=0 (Windows)
> - keluaran: cetak minimal satu angka metrik ke konsol, simpan minimal satu gambar ke report/figures/"

**Alasan Khusus:**
FID secara teknis bisa dihitung (resize 28→299, replikasi channel), tapi yang diukur jadi artefak interpolasi bikubik dan korelasi antar-channel buatan, bukan kualitas sampel — angka ilmiah yang salah sasaran. Karena itu prompt tidak meminta "implementasikan FID", tapi meminta penilaian apakah FID layak, lalu alternatif yang bisa dipertanggungjawabkan di CPU tanpa bobot besar.

Histogram intensitas piksel saja ditolak sebagai satu-satunya metrik karena hanya membaca distribusi marginal — gambar acak bisa punya histogram mirip tapi struktur spasialnya rusak. Tambahan yang diminta: metrik yang menangkap struktur (MMD pada ruang piksel, Wasserstein-1 pada magnitudo gradien Sobel).

Agar angkanya bisa disimpulkan tanpa ambang universal, prompt meminta pembanding internal: jarak **Real vs Generated** dilaporkan bersama jarak **Real vs Real** (dua paruhan test set) sebagai lantai (floor) praktis. Tanpa floor itu, angka 0,03 atau 0,10 tidak bisa ditafsirkan.

**Output Model:**
`src/evaluate.py` — 252 baris, import absolut `from src...` dengan bootstrap `sys.path` di level modul. Isi: `_extract_real_images` (test set, dinormalkan ke [0,1]), `compute_rbf_mmd` (RBF multi-sigma 0,5/1/2/5/10 pada piksel ter-flatten), `_hist_wasserstein_1d` (jarak CDF 32 bin), `_sobel_gradient_magnitude` (kernel 3×3, `conv2d` groups=channel), `load_checkpoint_and_models`, `save_evaluation_figure`, `evaluate`.

Default `num_eval_samples=256` → 256 gambar test dibelah dua (128 Real-A vs 128 Real-B sebagai referensi) dan 128 gambar generated dengan `fixed_seed=1234`. Keluaran: 6 angka di konsol + `report/figures/evaluation_metrics.png` (grid 4×4 sampel, histogram intensitas, histogram gradien Sobel).

**Verifikasi checkpoint (dijalankan sebelum evaluasi):**

```
file            : C:\Users\rosiana\Documents\GenAI\checkpoints\ddpm_best.pt
size            : 9637216 bytes
tipe objek      : dict
kunci top-level : ['config', 'epoch', 'loss', 'model_state_dict', 'optimizer_state_dict']
epoch           : 24
loss            : 0.04297231029496234
jumlah tensor   : 130
load_state_dict : strict=True LOLOS
parameter model : 793,953
semua bobot finite (tidak NaN/Inf): True
forward dummy   : input (1, 1, 28, 28) -> output (1, 1, 28, 28)
STATUS: CHECKPOINT VALID
```

Cocok dengan catatan Sesi #1–#5: epoch terbaik 24, loss 0,04297, 793.953 parameter.

**Hasil Uji:**

```
=== Mini DDPM Evaluation Summary (CPU) ===
Checkpoint      : C:\Users\rosiana\Documents\GenAI\checkpoints\ddpm_best.pt (epoch=24, loss=0.04297231029496234)
Eval Samples    : 256 per set (image_size=28, channels=1)
RBF-MMD (Real vs Gen)   : 0.038898  [Real Split Ref: 0.007159]
Pixel W1 (Real vs Gen)  : 0.101793  [Real Split Ref: 0.008048]
Sobel Grad W1 (R vs G)  : 0.042321  [Real Split Ref: 0.004876]
Saved figure    : C:\Users\rosiana\Documents\GenAI\report\figures\evaluation_metrics.png
```

Exit code 0. Waktu jalan ±53 menit di CPU (1.000 timestep × batch 128, ±2,7–3,0 detik per forward pass). Gambar tersimpan 93.849 byte.

### 2. TAHAP PENANGANAN EROR / DEBUGGING

#### Kegagalan #1 — `FileNotFoundError: checkpoints/ddpm_best.pt`

```
Traceback (most recent call last):
  File "C:\Users\rosiana\Documents\GenAI\src\evaluate.py", line 252, in <module>
    evaluate()
  File "C:\Users\rosiana\Documents\GenAI\src\evaluate.py", line 198, in evaluate
    cfg, model, diffusion, ckpt = load_checkpoint_and_models(ckpt_path, device)
  File "C:\Users\rosiana\Documents\GenAI\src\evaluate.py", line 104, in load_checkpoint_and_models
    ckpt = torch.load(ckpt_path, map_location=device)
  File "C:\Users\rosiana\AppData\Local\Programs\Python\Python313\Lib\site-packages\torch\serialization.py", line 1565, in load
    with _open_file_like(f, "rb") as opened_file:
  File "C:\Users\rosiana\AppData\Local\Programs\Python\Python313\Lib\site-packages\torch\serialization.py", line 795, in _open_file_like
    return _open_file(name_or_buffer, mode)
  File "C:\Users\rosiana\AppData\Local\Programs\Python\Python313\Lib\site-packages\torch\serialization.py", line 776, in __init__
    super().__init__(open(name, mode))  # noqa: SIM115
FileNotFoundError: [Errno 2] No such file or directory: 'C:\\Users\rosiana\\Documents\\GenAI\\checkpoints\\ddpm_best.pt'
```

**Analisis & Solusi Kelompok:**
Ini bukan error kode dan **tidak dikirim ke Gemini** — `checkpoints/` memang hanya berisi `.gitkeep` saat itu, jadi prasyarat belum terpenuhi. Kode yang sama, tanpa satu baris pun diubah, langsung jalan begitu file checkpoint dari Faqih dipasang. Mencatatnya sebagai "error Gemini" akan menyesatkan; mencetak angka evaluasi tanpa checkpoint akan mengarang.

Satu detail teknis saat memulihkan file: checkpoint yang diterima berbentuk zip berisi direktori `ddpm_best/` (format `_use_new_zipfile_serialization`), sehingga harus disusun ulang menjadi satu file `ddpm_best.pt` sebelum bisa dibaca `torch.load`.

**Hasil Uji:** error di atas hilang setelah checkpoint tersedia; run berikutnya exit 0 dengan angka di bagian Hasil Uji.

#### Error kode pada sesi ini: tidak ada

Kode dari Gemini jalan apa adanya pada percobaan pertama setelah prasyarat lengkap — `load_state_dict(strict=True)` lolos, seluruh bobot finite, dan `evaluate()` selesai tanpa traceback.

### 3. INTERPRETASI ANGKA

| Metrik | Real vs Generated | Real vs Real (floor) | Rasio terhadap floor |
|---|---|---|---|
| RBF-MMD² | 0,038898 | 0,007159 | 5,4× |
| Wasserstein-1 intensitas piksel | 0,101793 | 0,008048 | 12,6× |
| Wasserstein-1 magnitudo gradien Sobel | 0,042321 | 0,004876 | 8,7× |

Pembacaan: dua paruhan data asli hampir tidak bisa dibedakan metrik ini (MMD 0,007; W1 piksel 0,008), jadi floor-nya rendah dan bisa dipakai sebagai pembanding. Distribusi sampel model masih jauh di atas floor — 5–13× tergantung metrik — dengan jarak terbesar pada intensitas piksel.

Panel histogram menjelaskan sumbernya: sampel generated menumpuk di intensitas mendekati 0 (latar hitam) jauh lebih tajam daripada data asli, artinya model cenderung menghasilkan gambar berkontras rendah dengan area gelap berlebih. Gradien Sobel relatif lebih dekat (8,7×) karena kedua distribusi sama-sama didominasi tepi lemah. Ini konsisten dengan tampilan visual grid 4×4: siluet pakaian terbaca, tapi blur dan tepi bergerigi.

Kesimpulan yang bisa ditulis di laporan: model **belum** mencapai kualitas distribusi data asli, dan angka ini jadi baseline kuantitatif — bukan penilaian "sudah bagus" hanya karena loss training 0,043.

---

## Sesi #7 — 25/11/2026

- **Target Modul:** `src/ablation.py`
- **Operator:** Mia Rosiana Santoso
- **Tautan Sesi Chat:** _belum diisi — link share percakapan Gemini menyusul dari operator (dialog share tidak berhasil dibuka lewat otomasi; diisi manual, tidak dikarang)_
- **Model LLM:** Gemini

### 1. TAHAP INISIASI (Meminta Kode Awal)

**Prompt Kelompok:**

> "konteks: aku punya proyek Mini DDPM from scratch untuk Fashion-MNIST 28x28 grayscale, sudah selesai training 25 epoch (loss 0.16831 -> 0.04297, model 793.953 parameter, base_ch=32, T=1000, beta linear 1e-4 sampai 0.02). sekarang sesi 7: aku harus bikin src/ablation.py yang membandingkan dua konfigurasi supaya kelihatan pilihan di config.py itu masuk akal, bukan asal comot. jalan di CPU, Windows, python 3.13, torch 2.14.0+cpu.
>
> yang mau aku bandingkan: noise schedule linear (yang sekarang dipakai) versus cosine. masing-masing 3 epoch saja, tujuannya lihat tren loss, bukan hasil final.
>
> masalahnya begini: src/diffusion.py cuma punya linear, beta-nya di-hardcode pakai torch.linspace di __init__, dan field beta_schedule di Config nggak pernah dibaca siapa-siapa. aku nggak boleh ngubah diffusion.py karena itu file punya rekan tim dan sudah diverifikasi. jadi kalau mau cosine, semua harus dari dalam ablation.py tanpa nyentuh file itu.
>
> yang aku bingung:
>
> 1. cara paling bersih nambah schedule cosine tanpa ngubah diffusion.py apa? subclass DDPM lalu hitung ulang buffer-nya, atau ada cara lain yang lebih sedikit menduplikasi rumus?
>
> 2. kalau lewat subclass, buffer apa saja yang wajib dihitung ulang supaya q_sample, p_losses, dan p_sample_loop tetap konsisten? aku nggak mau hasilnya beda bukan karena schedule-nya, tapi karena buffer-nya nggak sinkron.
>
> 3. 3 epoch cukup nggak buat lihat bedanya, atau selisihnya bakal ketutup noise? kalau ternyata dua-duanya mirip, gimana cara nulis kesimpulan yang jujur tanpa maksa bilang salah satu lebih bagus?
>
> 4. src/train.py selalu menulis checkpoints/ddpm_best.pt dan logs/train_history.json. ablation-ku nggak boleh nimpa dua file itu. aman nggak kalau aku cuma mengganti cfg.ckpt_dir dan cfg.log_dir per varian, atau harus punya loop training sendiri?
>
> yang aku minta ada di src/ablation.py, spesifik per fungsi:
>
> 1. CosineDDPM(DDPM)
>    subclass dari DDPM yang menghitung ulang buffer schedule cosine di dalam ablation.py, tanpa mengubah src/diffusion.py.
>
> 2. run_variant(name, cfg, use_cosine)
>    menjalankan training 3 epoch dengan seed dan urutan data yang sama untuk kedua varian, menyimpan checkpoint dan log ke folder terpisah (bukan checkpoints/ddpm_best.pt atau logs/train_history.json), lalu mencatat loss tiap epoch dan waktu eksekusinya.
>
> 3. plot_ablation(results, save_path)
>    menggabungkan kurva loss kedua varian dalam satu grafik PNG dan menyimpannya ke report/figures/.
>
> 4. main()
>    menjalankan varian linear dan cosine, mencetak tabel perbandingan di konsol (loss epoch 1/2/3, waktu per epoch, jumlah parameter), lalu memanggil fungsi plot.
>
> signature yang ada di repo (jangan diubah, ini verbatim):
>
> ```python
> # src/config.py
> @dataclass
> class Config:
>     dataset: str = "FashionMNIST"
>     data_root: str = "data"
>     subset_size: int = 15_000
>     image_size: int = 28
>     channels: int = 1
>     num_workers: int = 0
>     unet_base: int = 32
>     time_embed_dim: int = 128
>     num_groups: int = 8
>     timesteps: int = 1000
>     beta_start: float = 1e-4
>     beta_end: float = 0.02
>     beta_schedule: str = "linear"
>     batch_size: int = 128
>     epochs: int = 25
>     lr: float = 2e-4
>     seed: int = 42
>     sample_every: int = 5
>     num_sample_images: int = 64
>     ckpt_dir: str = "checkpoints"
>     log_dir: str = "logs"
>     fig_dir: str = "report/figures"
>     # metode: to_dict(), run_name()
>
> # src/diffusion.py
> # buffer yang terdaftar di __init__: betas, alphas, alphas_cumprod,
> # alphas_cumprod_prev, sqrt_alphas_cumprod, sqrt_one_minus_alphas_cumprod,
> # sqrt_recip_alphas_cumprod, sqrt_recipm1_alphas_cumprod, posterior_variance,
> # posterior_log_variance_clipped, posterior_mean_coef1, posterior_mean_coef2
> class DDPM(nn.Module):
>     def __init__(self, timesteps: int = 1000, beta_start: float = 1e-4,
>                  beta_end: float = 0.02, clip_denoised: bool = True) -> None
>     def q_sample(self, x0, t, noise=None) -> torch.Tensor
>     def p_losses(self, model, x0, t, noise=None) -> torch.Tensor
>     def predict_start_from_noise(self, x_t, t, noise)
>     def q_posterior_mean_variance(self, x0, x_t, t)
>     def p_mean_variance(self, model, x_t, t)
>     def p_sample(self, model, x_t, t, t_index)
>     def p_sample_loop(self, model, shape, device="cpu") -> torch.Tensor
>
> # src/model.py
> class DDPMUNet(nn.Module):
>     def __init__(self, in_channels: int = 1, base_ch: int = 48,
>                  time_emb_dim: int = 192)
>
> # src/train.py
> def set_seed(seed: int = 42)
> def train(cfg: Config)   # loop tqdm; simpan ddpm_epoch_XXX.pt tiap epoch,
>                          # ddpm_best.pt kalau loss terbaik, logs/train_history.json
>
> # src/data.py
> def get_dataset(cfg, train: bool = True) -> Dataset
> def get_dataloader(cfg, train: bool = True) -> DataLoader
> ```
>
> batasan:
>
> - cuma bikin src/ablation.py, jangan ubah file lain di src/ (config.py, data.py, diffusion.py, model.py, train.py, sample.py, plot_loss.py sudah diverifikasi)
> - jangan pakai library diffusion siap pakai (diffusers dll)
> - import absolut from src.x import y, dan bootstrap sys.path di level modul (bukan di dalam `__main__`), karena skrip dijalankan `python src/ablation.py`
> - num_workers=0 (Windows)
> - 3 epoch per konfigurasi, jangan lebih
> - checkpoint dan log ablation wajib di lokasi terpisah; checkpoints/ddpm_best.pt dan logs/train_history.json tidak boleh tersentuh
> - seed dan urutan data harus sama untuk kedua varian
> - keluaran: tabel perbandingan di konsol (loss epoch 1/2/3, waktu per epoch, jumlah parameter) + minimal satu PNG di report/figures/ berisi dua kurva loss dalam satu plot"

**Alasan Khusus:**

Ablation ini diminta untuk mempertanggungjawabkan pilihan `beta_schedule = "linear"` di `config.py`. Pemeriksaan repo sebelum menulis prompt menemukan bahwa field itu **tidak dibaca file mana pun** (grep: 1 definisi, 0 pemakaian) — `DDPM.__init__` meng-hardcode `torch.linspace(beta_start, beta_end, timesteps)`. Jadi pilihan schedule di proyek ini sebelumnya tidak punya dasar eksperimen sama sekali; ablation dibuat untuk mengisinya.

Karena `src/diffusion.py` milik rekan tim dan tidak boleh disentuh, cosine harus datang dari subclass di dalam `ablation.py`. Dua belas nama buffer didaftarkan verbatim di prompt supaya jawaban tidak menebak nama buffer yang harus dihitung ulang — risiko yang disebut eksplisit di pertanyaan 2 (hasil beda bukan karena schedule, tapi karena buffer tidak sinkron).

Pertanyaan 4 diajukan lebih dulu daripada meminta kode karena `src/train.py` menulis path statis: kalau `train()` dipanggil ulang dengan `cfg.ckpt_dir`/`cfg.log_dir` yang diganti, `DDPM` standar tetap yang diinstansiasi di dalamnya sehingga varian cosine tidak akan terpakai. Prompt meminta jawaban eksplisit soal ini sebelum menulis loop training sendiri.

**Output Model:**
`src/ablation.py` — 316 baris, import absolut `from src...` dengan bootstrap `ROOT_DIR = Path(__file__).resolve().parents[1]` di level modul. Isi: `CosineDDPM(DDPM)` (hitung `alpha_bar` cosine `f(t)=cos((t/T+s)/(1+s)·π/2)²`, `s=0.008`, di float64 lalu cast float32, `betas = 1 - ᾱ_t/ᾱ_{t-1}` di-clamp `[1e-4, 0.999]`, lalu `register_buffer` menimpa 12 buffer induk; `raise KeyError` bila nama buffer tidak ada di induk), `run_variant(name, cfg, use_cosine)` (`copy.deepcopy(cfg)`, seed dan `DataLoader` sama untuk kedua varian, ckpt ke `checkpoints/ablation/<name>/`, log ke `logs/ablation/<name>_history.json`, payload `{epoch, variant, beta_schedule, model_state_dict, optimizer_state_dict, loss}`), `plot_ablation(results, save_path)` (figsize 8.5×5.0 dpi 150; linear `#1f77b4` bulat solid, cosine `#d95f02` kotak putus-putus), `_print_comparison_table`, `main()`.

**Hasil Uji (run final, 2 epoch per varian):**

```
[Ablation] Menjalankan varian 'linear' (linear)...
  Epoch 1/2 | Loss: 0.18872 | Waktu: 911.23 dtk
  Epoch 2/2 | Loss: 0.08593 | Waktu: 868.07 dtk

[Ablation] Menjalankan varian 'cosine' (cosine)...
  Epoch 1/2 | Loss: 0.25580 | Waktu: 854.90 dtk
  Epoch 2/2 | Loss: 0.13699 | Waktu: 836.77 dtk

-------------------------------------------------------------------------
HASIL ABLATION STUDY NOISE SCHEDULE (3 EPOCH)
-------------------------------------------------------------------------
Varian     |    Params |   Epoch 1 |   Epoch 2 | Avg Time/Ep | Total Time
-------------------------------------------------------------------------
linear     |   793,953 |   0.18872 |   0.08593 |    889.65 s |  1779.29 s
cosine     |   793,953 |   0.25580 |   0.13699 |    845.84 s |  1691.67 s
-------------------------------------------------------------------------

[Ablation] Grafik perbandingan disimpan ke: report\figures\ablation_noise_schedule.png
```

Exit code 0. Total ±58 menit. `logs/ablation/linear_history.json` dan `logs/ablation/cosine_history.json` berisi `num_params: 793953` (identik dengan model utama), `seed: 42`, `timesteps: 1000`, `unet_base: 32`. Gambar 95.627 byte. `checkpoints/ddpm_best.pt` mtime tetap `2026-11-25 18:59:06` — tidak tersentuh; `logs/train_history.json` tidak dibuat ulang.

### 2. TAHAP PENANGANAN EROR / DEBUGGING

#### Error kode pada sesi ini: tidak ada

Kode dari Gemini jalan apa adanya pada percobaan pertama — tidak ada traceback dari `CosineDDPM`, `run_variant`, `plot_ablation`, maupun `main`. Signature yang ditebak di jawaban (`get_dataloader(cfg, train=True)` mengembalikan `(x0, y)`, `p_losses(model, x0, t)`, `DDPMUNet(in_channels, base_ch, time_emb_dim)`) cocok dengan isi repo, jadi tidak ada perbaikan import/argumen yang perlu dikirim balik ke Gemini.

#### Hentikan run pertama (bukan error kode)

Run pertama memakai 3 epoch sesuai prompt. Hasil nyata epoch pertama:

```
[Ablation] Menjalankan varian 'linear' (linear)...
  Epoch 1/3 | Loss: 0.18872 | Waktu: 915.48 dtk
```

915,48 detik per epoch berarti 6 epoch ≈ 92 menit — melewati batas waktu sesi (1 jam). Proses dihentikan manual oleh operator (`py.exe` pid 13744 → `python.exe` pid 3948, `Stop-Process -Force`), **bukan crash** (wrapper `tee` tetap melaporkan exit 0 karena pipa, bukan karena python selesai), dan hasilnya tidak dipalsukan: tidak ada angka epoch 2/3 yang dikarang untuk run ini.

Tindakan operator sebelum run kedua (perubahan kecil, bukan logika model):

1. `ABLATION_EPOCHS = 2` sebagai konstanta modul, dipakai `run_variant` dan `main` — 3 epoch per varian diturunkan jadi 2 karena batas waktu, dicatat di komentar kode.
2. Judul grafik dan header tabel dibuat dinamis mengikuti jumlah epoch (sebelumnya hardcode "Epoch 1/2/3" dan "(3 EPOCH)", sehingga akan `IndexError` di `losses[2]` kalau dijalankan 2 epoch). Perbaikan ini dikerjakan **setelah** run final selesai, jadi yang tercetak di konsol run final masih judul lama `(3 EPOCH)` — dikutip apa adanya di blok output di atas; hasil verifikasi ulang hanya syntax check, tidak ada run penuh ketiga.

Bukti bahwa pemotongan epoch tidak mengubah apa pun selain durasi: loss epoch 1 varian linear pada run kedua **identik** dengan run pertama (0.18872) karena seed dan urutan data sama.

### 3. INTERPRETASI ANGKA

| Varian | Parameter | Loss epoch 1 | Loss epoch 2 | Rata-rata waktu/epoch | Total |
|---|---|---|---|---|---|
| Linear | 793.953 | 0,18872 | 0,08593 | 889,65 s | 1779,29 s |
| Cosine | 793.953 | 0,25580 | 0,13699 | 845,84 s | 1691,67 s |

Pembacaan yang jujur: pada dua epoch ini **linear memberi loss lebih rendah**, dan penurunannya juga lebih dalam (linear −0,1028; cosine −0,1188). Cosine tidak lebih buruk secara laju penurunan relatif, tapi level loss-nya konsisten lebih tinggi di kedua epoch.

Alasannya bukan "cosine jelek", melainkan perbedaan skala target: schedule cosine mempertahankan `ᾱ_t` lebih tinggi di timestep awal, sehingga noise yang ditambahkan pada `x_t` lebih kecil dan MSE prediksi noise-nya bergerak di skala yang berbeda dari linear. Dua epoch juga jauh dari konvergen (model utama butuh 24 epoch untuk mencapai 0,043), jadi **tidak cukup** untuk menyimpulkan schedule mana yang lebih baik — yang bisa ditulis di laporan hanya: pada anggaran 2 epoch dan subset 15.000 gambar, linear lebih rendah loss-nya, dan perbandingan ini belum menjawab kualitas sampel.

Angka waktu: linear 889,65 s/epoch vs cosine 845,84 s/epoch (cosine ±5% lebih cepat). Selisih ini bukan efek schedule, melainkan variasi beban CPU antar-run — kedua varian memakai arsitektur dan jumlah parameter identik (793.953), jadi tidak ada alasan komputasional cosine lebih murah.

Yang bisa diklaim sebagai temuan teknis justru bagian non-angka: `CosineDDPM` berhasil menimpa 12 buffer `DDPM` tanpa menyentuh `src/diffusion.py`, dan artefak ablation benar-benar terisolasi (`checkpoints/ablation/*`, `logs/ablation/*`, mtime `checkpoints/ddpm_best.pt` tidak berubah).

---

## Hasil Akhir Training (Bukti Fungsionalitas)

Training penuh dijalankan setelah kelima sesi selesai, memakai kode apa adanya dari log di atas — tanpa perubahan logika, hanya perbaikan import dan penyeragaman angka parameter.

**Konfigurasi yang benar-benar dipakai** (`src/config.py`): subset 15.000 gambar, batch 128, 25 epoch, LR 2e-4 Adam, T=1000 linear β 1e-4→0.02, UNet `base_ch=32` / `time_emb_dim=128` (**793.953** parameter).

**Hardware:** CPU lokal, AMD Ryzen 7 8845HS (8C/16T), 16 GB RAM, tanpa CUDA.

**Hasil:**

| Epoch | Loss (MSE noise) | Waktu |
|---|---|---|
| 1 | 0.16831 | 80,2 s |
| 2 | 0.07584 | — |
| 5 | 0.05708 | 93,2 s |
| 10 | 0.04809 | 100,2 s |
| 17 | 0.04332 | 86,5 s |
| 24 (terbaik) | **0.04297** | — |
| 25 (akhir) | 0.04307 | 79,6 s |

Loss turun **74,4%** dari epoch 1 ke 25, lalu mendatar di ~0,043 — pola konvergen yang wajar untuk DDPM. Total waktu ±35 menit.

**Artefak yang dihasilkan:**

- `checkpoints/ddpm_best.pt` — checkpoint loss terendah (epoch 24) + `ddpm_epoch_001.pt` … `ddpm_epoch_025.pt`
- `logs/train_history.json` — loss per epoch
- `report/figures/progression_epochs_1_to_25.png` — grid progres sampling dengan **fixed noise** (`torch.manual_seed(1234)` di-reset tiap epoch, jadi x_T identik dan perbandingan antar epoch apple-to-apple)
- `report/figures/samples_epoch_XXX.png` — grid 64 sampel per epoch
- `report/figures/loss_curve.png` — kurva loss

**Interpretasi kualitatif:** pada epoch 1–3 sampel masih berupa gumpalan abu-abu tanpa struktur. Mulai epoch ~8 siluet mulai terbaca (bentuk memanjang untuk celana, blok persegi untuk tas). Epoch 15–25 sudah menghasilkan item Fashion-MNIST yang bisa dikenali — pullover, trouser, sepatu, tas — dengan tepi masih agak lembek, yang wajar untuk 793 ribu parameter dan 25 epoch.

---

## Ringkasan Pola Interaksi

| Sesi | Prompt | Error nyata | Sumber perbaikan |
|---|---|---|---|
| 1 | 1 + 1 lanjutan | 1 (`data_dir` + path relatif) | Gemini |
| 2 | 1 | 0 | — |
| 3 | 1 | 0 | — |
| 4 | 1 | 0 | — |
| 5 | 1 + 4 lanjutan | 4 (`UNet`, `checkpoint_dir`, import path, `src` not found) | Gemini |
| 6 | 1 | 0 error kode (`FileNotFoundError` murni prasyarat: checkpoint belum ada) | — |
| 7 | 1 | 0 error kode (run pertama dihentikan manual: proyeksi 92 menit > batas 1 jam) | operator (epoch 3→2, header dinamis) |

**Pelajaran utama:** LLM tidak bisa menebak isi file yang tidak diberikan. Dua dari empat error di proyek ini berakar pada satu hal yang sama — konteks kode tidak lengkap di prompt. Setelah signature verbatim ditempel, kode langsung nyambung. Kesimpulan operasional: **tempel signature, bukan ringkasan.**
