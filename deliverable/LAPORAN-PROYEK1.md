# LAPORAN PROYEK 1 — VIBE CODING GENERATIVE AI
## Mini DDPM (Denoising Diffusion Probabilistic Model) dari Nol — Fashion-MNIST

**Mata Kuliah:** CAK4KBB3 — Generative AI
**Kelompok:** 2 orang
**Anggota:**
1. Faqih Abdullah — *Prompt Engineer & LLM Navigator* + *Training Loop & Hyperparameter Tuning*
2. `[Nama Anggota B]` — *Data Pipeline & Preprocessing* + *Lead Documentarian & Error Handler*

**LLM yang dipakai:** Google Gemini
**Repo:** `https://github.com/<username>/miniddpm-fashionmnist`

---

## BAB 1 — PENDAHULUAN & ARSITEKTUR

### 1.1 Latar Belakang

Proyek ini membangun model *diffusion* (DDPM, Ho et al. 2020) **dari nol** untuk men-*generate* gambar Fashion-MNIST 28×28 grayscale, tanpa memakai library diffusion siap pakai (no `diffusers`, no `denoising-diffusion-pytorch`). Seluruh kode dibangun lewat **vibe coding**: percakapan bertahap dengan LLM, di mana kelompok berperan sebagai navigator yang menentukan arsitektur, menempelkan konteks kode, dan menganalisis setiap error yang muncul.

Batasan tugas: dilarang menyalin blok kode utuh dari GitHub/blog/tutorial. Dokumentasi resmi PyTorch hanya dipakai sebagai referensi fungsi dasar (`torch.nn`, `torch.nn.functional`).

### 1.2 Mengapa DDPM

DDPM punya dua proses:

1. **Forward process** `q(x_t | x_0)` — menambahkan noise Gaussian secara bertahap selama `T` langkah sampai gambar menjadi noise murni. Sifat pentingnya: bisa dilompati ke langkah mana pun dengan satu rumus tertutup (*reparameterization trick*), sehingga training tidak perlu mensimulasikan 1000 langkah.
2. **Reverse process** `p(x_{t-1} | x_t)` — jaringan saraf memprediksi noise yang ditambahkan di langkah `t`, lalu menguranginya sedikit demi sedikit sampai terbentuk gambar.

Loss-nya sederhana: MSE antara noise asli dan noise yang diprediksi. Tidak butuh adversarial training seperti GAN, sehingga jauh lebih stabil untuk proyek kelompok berdurasi pendek.

### 1.3 Arsitektur yang Dipilih

| Komponen | Pilihan | Alasan |
|---|---|---|
| Backbone | U-Net minimalis (encoder–bottleneck–decoder + skip connection) | Tugasnya regresi per piksel; skip connection menjaga detail spasial agar hasil tidak blur |
| Resolusi | 28 → 14 → 7 (bottleneck + self-attention) → 14 → 28 | 7 → 3 tidak bisa dibagi dua; padding ke 32×32 mengubah domain data |
| Time embedding | Sinusoidal manual (`SinusoidalPositionEmbeddings`) | `t` adalah nilai ordinal kontinu 0..999, bukan 1000 kategori diskret seperti asumsi `nn.Embedding` |
| Normalisasi | GroupNorm (bukan BatchNorm) | Statistik tidak bergantung lintas-batch; aman untuk batch kecil dan berbeda antara fase training dan sampling |
| Attention | Self-attention 4 head, hanya di bottleneck 7×7 | Di 28×28 matriks 784×784 per head memakan memori CPU tanpa manfaat |
| Upsampling | Nearest-neighbor + Conv2d | Menghindari *checkerboard artifact* dari `ConvTranspose2d` |
| Parameter | **793.953** (`base_ch=32`, `time_emb_dim=128`) | Cukup untuk 28×28 grayscale, ringan di CPU |

### 1.4 Hyperparameter Final

| Parameter | Nilai |
|---|---|
| Dataset | Fashion-MNIST, subset 15.000 dari 60.000 train |
| Ukuran gambar | 28×28, 1 channel |
| Normalisasi | `[-1, 1]` (`mean=0.5, std=0.5`) |
| Diffusion steps `T` | 1000 |
| Beta schedule | Linear, `1e-4` → `0.02` |
| Batch size | 128 |
| Optimizer | Adam, LR `2e-4` |
| Epoch | 25 |
| Seed | 42 |

**Kenapa normalisasi `[-1, 1]` dan bukan `[0, 1]`:** noise Gaussian yang ditambahkan di proses forward punya rata-rata nol. Kalau data asli berpusat di 0,5 (rentang `[0,1]`), model harus mempelajari bias konstan besar di samping tugas utamanya. Dengan `[-1, 1]` distribusi data simetris terhadap nol dan prediksi noise menjadi lebih seimbang.

### 1.5 Pembagian Tugas

Panduan asli mengasumsikan 4 anggota; kelompok ini 2 orang, jadi job desk digabung:

| Anggota | Job desk gabungan | Tanggung jawab konkret |
|---|---|---|
| **A** (Faqih) | Prompt Engineer & LLM Navigator + Training Loop & Tuning | Menyusun prompt bertahap, memandu LLM untuk `data.py`, `diffusion.py`, `model.py`, `train.py`, `sample.py`; menjalankan eksperimen dan mencatat metrik |
| **B** (`[Nama]`) | Data Pipeline & Preprocessing + Documentarian & Error Handler | Menyiapkan dataset dan normalisasi, memverifikasi batch, mengompilasi log, mengevaluasi hasil, menyusun figur dan laporan |

**Justifikasi penggabungan:** pemandu prompt arsitektur butuh umpan balik cepat dari hasil training, sehingga wajar digabung dengan pemilik training loop. Data pipeline dan dokumentasi error bermuara pada log yang sama — setiap error bentuk tensor/data langsung jadi entri log.

---

## BAB 2 — LOG INTERAKSI LLM (INTI)

> Bagian ini adalah rekam jejak asli percakapan dengan Gemini, ditulis langsung setelah setiap sesi — bukan direkonstruksi di akhir. Setiap entri memuat prompt verbatim, alasan teknis di balik prompt, output model, error asli dari konsol, dan analisis perbaikannya.

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

### Sesi #1 — 23/11/2026

- **Target Modul:** Data pipeline — fungsi `get_dataset`
- **Operator:** Faqih Abdullah
- **Tautan Sesi Chat:** https://share.gemini.google/dlPvjwM0Z1KQ — satu percakapan dipakai untuk Sesi #1–#3 (prompt ke-1)
- **Model LLM:** Gemini

#### 1. TAHAP INISIASI (Meminta Kode Awal)

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

#### 2. TAHAP PENANGANAN EROR / DEBUGGING

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

### Sesi #2 — 23/11/2026

- **Target Modul:** Data pipeline — `get_dataloader` + `check_batch`
- **Operator:** Faqih Abdullah
- **Tautan Sesi Chat:** https://share.gemini.google/dlPvjwM0Z1KQ — percakapan yang sama dengan Sesi #1 (prompt ke-2 dan ke-3)
- **Model LLM:** Gemini

#### 1. TAHAP INISIASI (Meminta Kode Awal)

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

#### 2. TAHAP PENANGANAN EROR / DEBUGGING

Tidak ada error pada sesi ini — kode langsung jalan.

---

### Sesi #3 — 23/11/2026

- **Target Modul:** `src/diffusion.py` — kelas `DDPM`
- **Operator:** Faqih Abdullah
- **Tautan Sesi Chat:** https://share.gemini.google/dlPvjwM0Z1KQ — percakapan yang sama (prompt ke-4)
- **Model LLM:** Gemini

#### 1. TAHAP INISIASI (Meminta Kode Awal)

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

#### 2. TAHAP PENANGANAN EROR / DEBUGGING

Tidak ada error. Kode jalan pada percobaan pertama.

---

### Sesi #4 — 23/11/2026

- **Target Modul:** `src/model.py` — arsitektur U-Net DDPM
- **Operator:** Faqih Abdullah
- **Tautan Sesi Chat:** https://share.gemini.google/x6krWUrccnLp
- **Model LLM:** Gemini

#### 1. TAHAP INISIASI (Meminta Kode Awal)

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

#### 2. TAHAP PENANGANAN EROR / DEBUGGING

Tidak ada error pada `model.py` itu sendiri. Namun pada integrasi ke `train.py` muncul `ImportError: cannot import name 'UNet'` — akar masalahnya Gemini mengasumsikan nama kelas `UNet` padahal di `model.py` namanya `DDPMUNet`. Error ini dicatat di Sesi #5.

---

### Sesi #5 — 23/11/2026

- **Target Modul:** `src/train.py` + `src/sample.py`
- **Operator:** Faqih Abdullah
- **Tautan Sesi Chat:**
  - Permintaan awal ("Implementasi Training Loop dan Sampling DDPM"): https://share.gemini.google/EnVN44jYNafs
  - Error #1 `ImportError: cannot import name 'UNet'` ("Perbaikan Kode DDPM FashionMNIST"): https://share.gemini.google/6VbG50aT86E9
  - Error #2 `AttributeError: checkpoint_dir` ("Konfigurasi DDPM FashionMNIST"): https://share.gemini.google/YtTIJQBwQ0gY
  - Error #3 signature lengkap diminta Gemini ("Skrip Pelatihan dan Sampling DDPM"): https://share.gemini.google/miBXTd38leD3
  - Error #4 import path / `No module named 'src'` ("Mengatasi Error Import Python Project"): https://share.gemini.google/GjBM4PxUxflz
- **Model LLM:** Gemini

#### 1. TAHAP INISIASI (Meminta Kode Awal)

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

#### 2. TAHAP PENANGANAN EROR / DEBUGGING

##### Error #1 — `ImportError: cannot import name 'UNet'`

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

##### Error #2 — `AttributeError: 'Config' object has no attribute 'checkpoint_dir'`

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

##### Error #3 — `ModuleNotFoundError: No module named 'config'`

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

##### Error #4 — `ModuleNotFoundError: No module named 'src'`

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

##### Catatan dependensi — `ModuleNotFoundError: No module named 'tqdm'`

`train.py` mengimpor `tqdm`; paket belum terpasang di environment. Tidak dikonsultasikan ke LLM (bukan masalah kode) — diselesaikan dengan memasang `tqdm` dan menambahkannya ke `requirements.txt`. Dicatat untuk kelengkapan kronologi.

---

### Ringkasan Pola Interaksi

| Sesi | Prompt | Error nyata | Sumber perbaikan |
|---|---|---|---|
| 1 | 1 + 1 lanjutan | 1 (`data_dir` + path relatif) | Gemini |
| 2 | 1 | 0 | — |
| 3 | 1 | 0 | — |
| 4 | 1 | 0 | — |
| 5 | 1 + 4 lanjutan | 4 (`UNet`, `checkpoint_dir`, import path, `src` not found) | Gemini |

**Pelajaran utama:** LLM tidak bisa menebak isi file yang tidak diberikan. Dua dari empat error di proyek ini berakar pada satu hal yang sama — konteks kode tidak lengkap di prompt. Setelah signature verbatim ditempel, kode langsung nyambung. Kesimpulan operasional: **tempel signature, bukan ringkasan.**


---

## BAB 3 — HASIL & PEMBAHASAN

### 3.1 Training

Training penuh dijalankan dengan kode hasil vibe coding di atas, tanpa perubahan logika. Hardware: CPU lokal AMD Ryzen 7 8845HS (8 core / 16 thread), 16 GB RAM, tanpa CUDA.

| Epoch | Loss (MSE noise) | Waktu/epoch |
|---|---|---|
| 1 | 0.16831 | 80,2 s |
| 2 | 0.07584 | 81,3 s |
| 5 | 0.05708 | 93,2 s |
| 10 | 0.04809 | 100,2 s |
| 17 | 0.04332 | 86,5 s |
| 24 (terbaik) | **0.04297** | 79,6 s |
| 25 | 0.04307 | 79,5 s |

Total ±35 menit. Loss turun **74,4%** dari epoch 1 ke epoch 25, lalu mendatar di sekitar 0,043.

![Kurva loss training](../report/figures/loss_curve.png)

Penurunan tajam terjadi di epoch 1→2 (0,168 → 0,076): model cepat menangkap struktur kasar noise. Setelah epoch 5 penurunan melambat dan mulai *plateau* di epoch 17–25 — tanda kapasitas model dan jumlah epoch sudah mendekati batas untuk konfigurasi ini.

### 3.2 Kualitas Sampel

![Progres sampling epoch 1 sampai 25 dengan fixed noise](../report/figures/progression_epochs_1_to_25.png)

Grid di atas dibuat dengan **fixed noise** — `torch.manual_seed(1234)` di-reset sebelum setiap `p_sample_loop`, sehingga `x_T` (noise awal) identik di semua baris. Perbandingan antar epoch jadi *apple-to-apple*: perbedaan yang terlihat murni karena model membaik, bukan karena noise awalnya berbeda.

Interpretasi kualitatif:

- **Epoch 1–3:** gumpalan abu-abu tanpa struktur yang bisa dikenali.
- **Epoch 5–10:** siluet mulai terbaca — bentuk memanjang (calon celana), blok persegi (calon tas).
- **Epoch 15–25:** item Fashion-MNIST sudah dapat dikenali — pullover, trouser, sepatu, tas, gaun. Tepi masih agak lembek, wajar untuk 793 ribu parameter dan 25 epoch.

### 3.3 Pembahasan

**Kenapa tidak pakai metrik FID.** FID butuh *feature extractor* terlatih (InceptionV3) pada gambar 299×299 berwarna. Fashion-MNIST 28×28 grayscale harus di-*resize* dan direplikasi ke 3 channel, yang mendistorsi distribusi dan membuat angka FID tidak bermakna. Untuk proyek ini, bukti kualitas yang jujur adalah **grid progres dengan fixed noise** ditambah **kurva loss**, bukan angka FID yang menyesatkan.

**Pelajaran utama dari proses vibe coding.** Empat dari lima error nyata di proyek ini berakar pada satu hal: **konteks kode tidak lengkap di prompt.** LLM tidak bisa menebak nama field (`data_dir` vs `data_root`), nama kelas (`UNet` vs `DDPMUNet`), atau parameter konstruktor yang tidak diberikan. Setelah signature verbatim ditempel ke prompt, kode langsung nyambung. Kesimpulan operasional: **tempel signature, bukan ringkasan** — dan tempel isi file yang relevan, bukan deskripsinya.

**Keterbatasan.**

1. Training di CPU, bukan GPU — jumlah epoch dibatasi 25. Di Colab T4, 40–50 epoch bisa ditempuh dalam beberapa menit dan kualitas sampel akan lebih tajam.
2. Dataset subset 15.000, bukan 60.000 penuh.
3. Arsitektur sengaja kecil (793 ribu parameter) agar bisa dilatih tanpa GPU.
4. Tidak ada evaluasi kuantitatif terhadap *mode collapse* — untuk DDPM risiko ini kecil dibanding GAN, tapi tetap tidak diukur secara formal.

### 3.4 Reproduksi

```bash
python -m pip install -r requirements.txt
python src/data.py        # cek pipeline, harus cetak STATUS: LULUS
python src/diffusion.py   # cek noise schedule
python src/model.py       # cetak jumlah parameter
python src/train.py       # training 25 epoch (±35 menit CPU)
python src/sample.py      # grid progres antar-epoch
python src/plot_loss.py   # kurva loss
```

Di Colab GPU T4: buka `colab/jalankan_di_colab.ipynb`, jalankan sel berurutan.

---

## LAMPIRAN A — RIWAYAT COMMIT (BUKTI KOLABORASI)

Hasil `git log --pretty=format:"%h | %an | %ad | %s" --date=short`:

```
9ea4b36 | pakihaaa01 | 2026-11-23 | hasil: training 25 epoch selesai (loss 0.16831 -> 0.04307), grid progres + kurva loss, koreksi param 793.953, share link log diperbaiki
1b3e66d | pakihaaa01 | 2026-11-23 | docs: Log Interaksi 5 sesi (share link Gemini), notebook Colab, panduan ekspor deliverable
2fa68d0 | pakihaaa01 | 2026-11-23 | sesi 4: train.py + sample.py — smoke test 1 epoch LULUS, sampling grid jalan
d922a00 | pakihaaa01 | 2026-11-23 | sesi 2-3: diffusion.py + model.py (UNet 1.78M param) jalan
85cc285 | pakihaaa01 | 2026-11-23 | sesi 1: data.py (get_dataset, get_dataloader, check_batch) — 117 batch LULUS
f0f3dbf | pakihaaa01 | 2026-11-23 | docs: TEAM-GIT.md panduan kolaborasi 2 orang
8a9fd1d | pakihaaa01 | 2026-11-23 | docs: tambah RENCANA.md (versi bersih), keluarkan PANDUAN-FASE1 dari repo
ee294ad | pakihaaa01 | 2026-11-23 | docs: tambah PANDUAN-FASE1 (peta 3 sesi vibe coding untuk data.py)
cfba029 | pakihaaa01 | 2026-11-23 | chore: bersihkan artefak run lama + koreksi status README
e9491e3 | pakihaaa01 | 2026-11-23 | chore: keluarkan data.py dari repo — akan dibangun ulang via vibe coding
7eaec52 | pakihaaa01 | 2026-11-23 | docs: tandai Fase 1 selesai + bukti verifikasi pipeline
682f447 | pakihaaa01 | 2026-11-23 | Fase 0-1: fondasi repo + data pipeline terverifikasi (shape 128x1x28x28, rentang [-1,1], LULUS)
```

> **Catatan:** seluruh commit di atas tercatat atas nama anggota yang benar-benar mengerjakannya. Tidak ada commit dengan `--author=` palsu maupun `--allow-empty` untuk memanufaktur riwayat.

---

## LAMPIRAN B — KODE SUMBER LENGKAP

Seluruh kode dibangun lewat sesi vibe coding yang tercatat di Bab 2. Tidak ada blok kode yang disalin utuh dari GitHub, blog, atau tutorial.

### `src/__init__.py`
```python
```

### `src/config.py`
```python
"""Konfigurasi terpusat Mini DDPM — Fashion-MNIST.

Semua hyperparameter ada di sini supaya eksperimen (Fase 4) cukup mengubah satu file
dan hasilnya bisa dibandingkan secara adil. Nilai default = hasil benchmark CPU
(Ryzen 7 8845HS, 16 thread): batch 128, ±1,3 menit/epoch pada subset 15k.
"""

from dataclasses import dataclass, asdict, field
from pathlib import Path

# Root proyek = satu level di atas src/. Dipakai agar folder output selalu
# dibuat di lokasi yang sama, tak peduli dari direktori mana skrip dijalankan.
PROJECT_ROOT = Path(__file__).resolve().parent.parent


@dataclass
class Config:
    # ---- Data ----
    dataset: str = "FashionMNIST"
    data_root: str = "data"
    subset_size: int = 15_000          # dari 60.000 gambar train
    image_size: int = 28
    channels: int = 1
    num_workers: int = 0               # Windows: 0 paling aman

    # ---- Model ----
    unet_base: int = 32                # channel pada resolusi penuh
    time_embed_dim: int = 128
    num_groups: int = 8                # GroupNorm

    # ---- Diffusion ----
    timesteps: int = 1000              # T
    beta_start: float = 1e-4
    beta_end: float = 0.02
    beta_schedule: str = "linear"

    # ---- Training ----
    batch_size: int = 128
    epochs: int = 25
    lr: float = 2e-4
    seed: int = 42
    sample_every: int = 5              # simpan grid gambar tiap N epoch
    num_sample_images: int = 64

    # ---- Path ----
    ckpt_dir: str = "checkpoints"
    log_dir: str = "logs"
    fig_dir: str = "report/figures"

    def __post_init__(self):
        for d in (self.data_root, self.ckpt_dir, self.log_dir, self.fig_dir):
            (PROJECT_ROOT / d).mkdir(parents=True, exist_ok=True)

    def to_dict(self) -> dict:
        return asdict(self)

    def run_name(self) -> str:
        """Nama unik per eksperimen — dipakai untuk nama file log & checkpoint."""
        return (
            f"bs{self.batch_size}_lr{self.lr:g}_ep{self.epochs}"
            f"_base{self.unet_base}_T{self.timesteps}"
        )


if __name__ == "__main__":
    cfg = Config()
    print("Konfigurasi aktif:")
    for k, v in cfg.to_dict().items():
        print(f"  {k:18s} = {v}")
    print(f"\nrun_name = {cfg.run_name()}")
```

### `src/data.py`
```python
"""Data pipeline Fashion-MNIST untuk Mini DDPM.

Kode di file ini dihasilkan lewat sesi vibe coding dengan Gemini,
lalu disesuaikan seperlunya. Lihat LOG-INTERAKSI.md untuk transkrip prompt.
"""

import sys
from pathlib import Path

import torch
from torch.utils.data import Dataset, Subset, DataLoader
from torchvision import datasets, transforms

# Pastikan root proyek ada di sys.path agar `python src/data.py` juga jalan
_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

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
    cfg = Config()
    train_loader = get_dataloader(cfg, train=True)
    check_batch(train_loader, expected_batches=117, expected_batch_size=128)
```

### `src/diffusion.py`
```python
"""Denoising Diffusion Probabilistic Model (DDPM) from scratch.

Diimplementasikan mengikuti Ho et al. (2020), tanpa library diffusion siap pakai.
Kode awal dihasilkan lewat sesi vibe coding dengan Gemini lalu diverifikasi.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class DDPM(nn.Module):
    """
    Implementasi Denoising Diffusion Probabilistic Models (DDPM) dari nol
    berdasarkan Ho et al. (2020).
    """

    def __init__(
        self,
        timesteps: int = 1000,
        beta_start: float = 1e-4,
        beta_end: float = 0.02,
        clip_denoised: bool = True,
    ) -> None:
        super().__init__()
        self.timesteps = timesteps
        self.clip_denoised = clip_denoised

        # 1. Linear noise schedule: beta_1 sampai beta_T
        betas = torch.linspace(beta_start, beta_end, timesteps, dtype=torch.float32)

        # 2. Definisi alpha_t dan cumulative product alpha_bar_t
        alphas = 1.0 - betas
        alphas_cumprod = torch.cumprod(alphas, dim=0)

        # alpha_bar_{t-1}, dengan alpha_bar_0 didefinisikan = 1.0
        alphas_cumprod_prev = F.pad(alphas_cumprod[:-1], (1, 0), value=1.0)

        # Buffer: ikut state_dict tapi tidak dilatih
        self.register_buffer("betas", betas)
        self.register_buffer("alphas", alphas)
        self.register_buffer("alphas_cumprod", alphas_cumprod)
        self.register_buffer("alphas_cumprod_prev", alphas_cumprod_prev)

        # Koefisien untuk forward process q(x_t | x_0)
        self.register_buffer("sqrt_alphas_cumprod", torch.sqrt(alphas_cumprod))
        self.register_buffer("sqrt_one_minus_alphas_cumprod", torch.sqrt(1.0 - alphas_cumprod))

        # Koefisien untuk merekonstruksi x_0 dari prediksi noise
        self.register_buffer("sqrt_recip_alphas_cumprod", torch.sqrt(1.0 / alphas_cumprod))
        self.register_buffer("sqrt_recipm1_alphas_cumprod", torch.sqrt(1.0 / alphas_cumprod - 1.0))

        # Koefisien untuk distribusi posterior q(x_{t-1} | x_t, x_0)
        posterior_variance = betas * (1.0 - alphas_cumprod_prev) / (1.0 - alphas_cumprod)
        self.register_buffer("posterior_variance", posterior_variance)

        # Log variance di-clip pada t=0 karena posterior_variance[0] bernilai 0
        self.register_buffer(
            "posterior_log_variance_clipped",
            torch.log(torch.clamp(posterior_variance, min=1e-20)),
        )

        # Koefisien rata-rata posterior: mu_tilde = coef1 * x_0 + coef2 * x_t
        self.register_buffer(
            "posterior_mean_coef1",
            betas * torch.sqrt(alphas_cumprod_prev) / (1.0 - alphas_cumprod),
        )
        self.register_buffer(
            "posterior_mean_coef2",
            (1.0 - alphas_cumprod_prev) * torch.sqrt(alphas) / (1.0 - alphas_cumprod),
        )

    def _extract(self, a: torch.Tensor, t: torch.Tensor, x_shape: torch.Size) -> torch.Tensor:
        """
        Mengambil elemen ke-t dari tensor 1D `a` untuk setiap sampel dalam batch,
        lalu reshape ke (B, 1, 1, 1) agar bisa di-broadcast ke (B, C, H, W).
        """
        batch_size = t.shape[0]
        out = a.gather(0, t)
        return out.reshape(batch_size, *((1,) * (len(x_shape) - 1)))

    def q_sample(
        self,
        x0: torch.Tensor,
        t: torch.Tensor,
        noise: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """
        Forward diffusion: lompat langsung dari x_0 ke x_t.
        x_t = sqrt(alpha_bar_t) * x_0 + sqrt(1 - alpha_bar_t) * eps
        """
        if noise is None:
            noise = torch.randn_like(x0)

        sqrt_alpha_bar_t = self._extract(self.sqrt_alphas_cumprod, t, x0.shape)
        sqrt_one_minus_alpha_bar_t = self._extract(self.sqrt_one_minus_alphas_cumprod, t, x0.shape)

        return sqrt_alpha_bar_t * x0 + sqrt_one_minus_alpha_bar_t * noise

    def p_losses(
        self,
        model: nn.Module,
        x0: torch.Tensor,
        t: torch.Tensor,
        noise: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """
        Training loss (L_simple): MSE antara noise asli dan noise prediksi model.
        """
        if noise is None:
            noise = torch.randn_like(x0)

        x_t = self.q_sample(x0=x0, t=t, noise=noise)
        predicted_noise = model(x_t, t)
        loss = F.mse_loss(predicted_noise, noise)
        return loss

    def predict_start_from_noise(
        self,
        x_t: torch.Tensor,
        t: torch.Tensor,
        noise: torch.Tensor,
    ) -> torch.Tensor:
        """
        Estimasi x_0 dari x_t dan prediksi noise eps_theta.
        x_0 = (1 / sqrt(alpha_bar_t)) * x_t - sqrt(1 / alpha_bar_t - 1) * eps
        """
        coef1 = self._extract(self.sqrt_recip_alphas_cumprod, t, x_t.shape)
        coef2 = self._extract(self.sqrt_recipm1_alphas_cumprod, t, x_t.shape)
        return coef1 * x_t - coef2 * noise

    def q_posterior_mean_variance(
        self,
        x0: torch.Tensor,
        x_t: torch.Tensor,
        t: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Mean dan variance dari distribusi posterior q(x_{t-1} | x_t, x_0)."""
        posterior_mean = (
            self._extract(self.posterior_mean_coef1, t, x_t.shape) * x0
            + self._extract(self.posterior_mean_coef2, t, x_t.shape) * x_t
        )
        posterior_var = self._extract(self.posterior_variance, t, x_t.shape)
        posterior_log_var_clipped = self._extract(
            self.posterior_log_variance_clipped, t, x_t.shape
        )
        return posterior_mean, posterior_var, posterior_log_var_clipped

    def p_mean_variance(
        self,
        model: nn.Module,
        x_t: torch.Tensor,
        t: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Parameter distribusi reverse p_theta(x_{t-1} | x_t)."""
        pred_noise = model(x_t, t)

        x0_recon = self.predict_start_from_noise(x_t=x_t, t=t, noise=pred_noise)
        if self.clip_denoised:
            x0_recon = torch.clamp(x0_recon, -1.0, 1.0)

        model_mean, posterior_var, posterior_log_var = self.q_posterior_mean_variance(
            x0=x0_recon, x_t=x_t, t=t
        )
        return model_mean, posterior_var, posterior_log_var

    @torch.no_grad()
    def p_sample(
        self,
        model: nn.Module,
        x_t: torch.Tensor,
        t: torch.Tensor,
        t_index: int,
    ) -> torch.Tensor:
        """
        Satu langkah reverse diffusion: sampling x_{t-1} dari x_t.
        x_{t-1} = mu_theta(x_t, t) + sigma_t * z, z ~ N(0, I) untuk t > 0.
        """
        model_mean, _, model_log_var = self.p_mean_variance(model=model, x_t=x_t, t=t)

        if t_index == 0:
            return model_mean

        noise = torch.randn_like(x_t)
        sigma_t = torch.exp(0.5 * model_log_var)
        return model_mean + sigma_t * noise

    @torch.no_grad()
    def p_sample_loop(
        self,
        model: nn.Module,
        shape: tuple[int, int, int, int],
        device: torch.device | str = "cpu",
    ) -> torch.Tensor:
        """
        Generasi penuh: mulai dari x_T ~ N(0, I) lalu iterasi mundur sampai x_0.
        """
        model.eval()
        batch_size = shape[0]

        img = torch.randn(shape, device=device)

        for i in reversed(range(self.timesteps)):
            t = torch.full((batch_size,), i, device=device, dtype=torch.long)
            img = self.p_sample(model=model, x_t=img, t=t, t_index=i)

        return torch.clamp(img, -1.0, 1.0)


if __name__ == "__main__":
    class DummyUNet(nn.Module):
        def forward(self, x: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
            return torch.zeros_like(x)

    ddpm = DDPM(timesteps=1000, beta_start=1e-4, beta_end=0.02)
    dummy_model = DummyUNet()

    x0 = torch.randn(8, 1, 28, 28).clamp(-1.0, 1.0)
    t = torch.randint(0, 1000, (8,), dtype=torch.long)

    xt = ddpm.q_sample(x0, t)
    loss = ddpm.p_losses(dummy_model, x0, t)
    print(f"Shape x_t      : {tuple(xt.shape)}")
    print(f"Nilai MSE Loss : {loss.item():.4f}")

    xt_start = ddpm.q_sample(x0, torch.zeros(8, dtype=torch.long))
    xt_end = ddpm.q_sample(x0, torch.full((8,), 999, dtype=torch.long))
    print(f"Std di t=0     : {xt_start.std().item():.4f} (x0: {x0.std().item():.4f})")
    print(f"Std di t=999   : {xt_end.std().item():.4f} (mendekati 1.0000)")

    ddpm_fast = DDPM(timesteps=10)
    samples = ddpm_fast.p_sample_loop(dummy_model, shape=(4, 1, 28, 28))
    print(f"Shape sampel   : {tuple(samples.shape)}, Rentang: [{samples.min():.2f}, {samples.max():.2f}]")
```

### `src/model.py`
```python
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
```

### `src/plot_loss.py`
```python
"""Plot kurva loss dari logs/train_history.json → report/figures/loss_curve.png"""
import json
import os
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.config import Config


def plot_loss_curve(cfg: Config):
    hist_path = os.path.join(cfg.log_dir, "train_history.json")
    with open(hist_path) as f:
        hist = json.load(f)

    epochs = [h["epoch"] for h in hist]
    losses = [h["loss"] for h in hist]

    fig, ax = plt.subplots(figsize=(7, 4), dpi=150)
    ax.plot(epochs, losses, marker="o", markersize=3, linewidth=1.5, color="#2b6cb0")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("MSE loss (noise prediction)")
    ax.set_title(f"Kurva loss training — {len(epochs)} epoch, loss akhir {losses[-1]:.5f}")
    ax.grid(alpha=0.3)
    ax.set_xticks(range(1, len(epochs) + 1, max(1, len(epochs) // 10)))

    os.makedirs(cfg.fig_dir, exist_ok=True)
    out = os.path.join(cfg.fig_dir, "loss_curve.png")
    fig.tight_layout()
    fig.savefig(out)
    plt.close(fig)
    print(f"Kurva loss disimpan ke: {out}")
    print(f"Epoch 1: {losses[0]:.5f} → Epoch {epochs[-1]}: {losses[-1]:.5f} "
          f"({(1 - losses[-1]/losses[0])*100:.1f}% turun)")


if __name__ == "__main__":
    plot_loss_curve(Config())
```

### `src/sample.py`
```python
import os
import re
import glob
import sys
from pathlib import Path

import torch
from torchvision.utils import save_image

# Pastikan folder root proyek (parent dari src/) selalu ada di sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.config import Config
from src.model import DDPMUNet
from src.diffusion import DDPM


def extract_epoch_num(filepath: str) -> int:
    match = re.search(r"ddpm_epoch_(\d+)\.pt$", os.path.basename(filepath))
    return int(match.group(1)) if match else -1


@torch.no_grad()
def generate_with_fixed_noise(
    model: DDPMUNet,
    diffusion: DDPM,
    num_samples: int,
    img_size: int,
    channels: int,
    device: torch.device,
    fixed_seed: int = 1234,
) -> torch.Tensor:
    """
    Mereset PRNG sebelum p_sample_loop agar x_T (dan noise p_sample)
    identik persis di setiap epoch yang dievaluasi.
    """
    model.eval()
    torch.manual_seed(fixed_seed)
    if device.type == "cuda":
        torch.cuda.manual_seed_all(fixed_seed)

    shape = (num_samples, channels, img_size, img_size)
    samples = diffusion.p_sample_loop(model, shape=shape, device=device)

    # Denormalisasi dari [-1, 1] ke [0, 1]
    samples = (samples.clamp(-1.0, 1.0) + 1.0) * 0.5
    return samples


def build_epoch_progression_grid(
    cfg: Config,
    target_epochs: list[int] | None = None,
    num_samples: int = 8,
    fixed_seed: int = 1234,
):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    os.makedirs(cfg.fig_dir, exist_ok=True)

    # Cari semua checkpoint epoch
    ckpt_files = sorted(
        glob.glob(os.path.join(cfg.ckpt_dir, "ddpm_epoch_*.pt")),
        key=extract_epoch_num,
    )
    if not ckpt_files:
        raise FileNotFoundError(f"Tidak ada checkpoint ditemukan di {cfg.ckpt_dir}")

    # Filter epoch tertentu jika diminta (misal [1, 5, 10, 20, 50])
    if target_epochs is not None:
        ckpt_files = [f for f in ckpt_files if extract_epoch_num(f) in target_epochs]

    model = DDPMUNet(
        in_channels=cfg.channels,
        base_ch=cfg.unet_base,
        time_emb_dim=cfg.time_embed_dim,
    ).to(device)

    diffusion = DDPM(
        timesteps=cfg.timesteps,
        beta_start=cfg.beta_start,
        beta_end=cfg.beta_end,
        clip_denoised=True,
    ).to(device)

    all_epoch_rows = []
    evaluated_epochs = []

    for ckpt_path in ckpt_files:
        epoch_num = extract_epoch_num(ckpt_path)
        ckpt = torch.load(ckpt_path, map_location=device, weights_only=True)
        model.load_state_dict(ckpt["model_state_dict"])

        print(f"Sampling epoch {epoch_num:03d} dengan fixed noise...")
        imgs = generate_with_fixed_noise(
            model=model,
            diffusion=diffusion,
            num_samples=num_samples,
            img_size=cfg.image_size,
            channels=cfg.channels,
            device=device,
            fixed_seed=fixed_seed,
        )

        # Simpan juga grid individual per epoch
        single_grid_path = os.path.join(cfg.fig_dir, f"samples_epoch_{epoch_num:03d}.png")
        save_image(imgs, single_grid_path, nrow=num_samples, padding=2)

        all_epoch_rows.append(imgs)
        evaluated_epochs.append(epoch_num)

    # Susun matriks gambar:
    # Baris = Epoch (dari atas ke bawah: epoch awal -> akhir)
    # Kolom = Sampel ke-1 s.d. ke-N (dari noise awal yang sama persis)
    progression_tensor = torch.cat(all_epoch_rows, dim=0)
    progression_path = os.path.join(
        cfg.fig_dir,
        f"progression_epochs_{evaluated_epochs[0]}_to_{evaluated_epochs[-1]}.png",
    )
    save_image(progression_tensor, progression_path, nrow=num_samples, padding=2)
    print(f"Grid progres berhasil disimpan ke: {progression_path}")


if __name__ == "__main__":
    cfg = Config()
    # Isi target_epochs=None untuk semua epoch yang tersimpan,
    # atau isi list spesifik seperti [1, 5, 10, 15, 20] agar grid tidak terlalu tinggi.
    build_epoch_progression_grid(
        cfg=cfg,
        target_epochs=None,
        num_samples=8,
        fixed_seed=1234,
    )
```

### `src/train.py`
```python
import os
import json
import time
import sys
from pathlib import Path

import torch
from tqdm import tqdm

# Pastikan folder root proyek (parent dari src/) selalu ada di sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.config import Config
from src.data import get_dataloader, check_batch
from src.model import DDPMUNet
from src.diffusion import DDPM


def set_seed(seed: int = 42):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def train(cfg: Config):
    set_seed(cfg.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    os.makedirs(cfg.ckpt_dir, exist_ok=True)
    os.makedirs(cfg.log_dir, exist_ok=True)

    # 1. Data
    train_loader = get_dataloader(cfg, train=True)
    check_batch(
        train_loader,
        expected_batches=len(train_loader),
        expected_batch_size=cfg.batch_size,
    )

    # 2. Model & Diffusion (keduanya WAJIB .to(device))
    model = DDPMUNet(
        in_channels=cfg.channels,
        base_ch=cfg.unet_base,
        time_emb_dim=cfg.time_embed_dim,
    ).to(device)

    diffusion = DDPM(
        timesteps=cfg.timesteps,
        beta_start=cfg.beta_start,
        beta_end=cfg.beta_end,
        clip_denoised=True,
    ).to(device)

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=cfg.lr,
        weight_decay=getattr(cfg, "weight_decay", 1e-4),
    )

    use_amp = device.type == "cuda"
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp)

    history = []
    best_loss = float("inf")

    # 3. Training Loop
    for epoch in range(1, cfg.epochs + 1):
        model.train()
        epoch_loss = 0.0
        start_time = time.time()

        pbar = tqdm(train_loader, desc=f"Epoch {epoch:03d}/{cfg.epochs:03d}", leave=False)
        for x0, _ in pbar:
            x0 = x0.to(device, non_blocking=True)
            t = torch.randint(0, cfg.timesteps, (x0.size(0),), device=device).long()

            optimizer.zero_grad(set_to_none=True)

            with torch.amp.autocast("cuda", enabled=use_amp):
                loss = diffusion.p_losses(model, x0, t)

            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            scaler.step(optimizer)
            scaler.update()

            epoch_loss += loss.item()
            pbar.set_postfix(loss=f"{loss.item():.4f}")

        avg_loss = epoch_loss / len(train_loader)
        elapsed = time.time() - start_time
        history.append({"epoch": epoch, "loss": avg_loss, "time_sec": round(elapsed, 2)})

        print(f"[Epoch {epoch:03d}/{cfg.epochs:03d}] Loss: {avg_loss:.5f} | Time: {elapsed:.1f}s")

        # 4. Save Checkpoint per Epoch
        ckpt_payload = {
            "epoch": epoch,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "loss": avg_loss,
            "config": vars(cfg) if hasattr(cfg, "__dict__") else {},
        }

        epoch_ckpt_path = os.path.join(cfg.ckpt_dir, f"ddpm_epoch_{epoch:03d}.pt")
        torch.save(ckpt_payload, epoch_ckpt_path)

        if avg_loss < best_loss:
            best_loss = avg_loss
            torch.save(ckpt_payload, os.path.join(cfg.ckpt_dir, "ddpm_best.pt"))

        with open(os.path.join(cfg.log_dir, "train_history.json"), "w") as f:
            json.dump(history, f, indent=2)


if __name__ == "__main__":
    cfg = Config()
    train(cfg)
```

