# LOG INTERAKSI LLM — KELOMPOK [NAMA KELOMPOK]

> **Aturan pengisian:** satu sesi dengan LLM = satu entri. Diisi langsung setelah sesi selesai.
> Setiap entri wajib punya tautan sesi chat yang masih aktif.
> Jangan menulis ulang di akhir proyek — kronologi asli yang dinilai.

**Anggota:**
- `[Nama A]` — Prompt Engineer & LLM Navigator + Training Loop & Hyperparameter Tuning
- `[Nama B]` — Data Pipeline & Preprocessing + Lead Documentarian & Error Handler

**Model:** Mini DDPM (Denoising Diffusion Probabilistic Model) — Fashion-MNIST 28×28
**Akun LLM yang dipakai:** `[ChatGPT / Claude / Gemini — pilih satu, konsisten]`

**Daftar isi sesi:**

| # | Tanggal | Target Modul | Operator | Status |
|---|---|---|---|---|
| 1 | | | | |
| 2 | | | | |
| 3 | | | | |

---

## TEMPLATE ENTRI (salin blok ini untuk setiap sesi baru)

```markdown
---

## Sesi #[N] — [DD/MM/YYYY]

- **Target Modul:** (contoh: Struktur Pemotong Patches pada ViT / Debugging Loss)
- **Operator:** [nama anggota yang memandu chat]
- **Tautan Sesi Chat:** [share link]
- **Model LLM:** [nama model + versi]

### 1. TAHAP INISIASI (Meminta Kode Awal)

**Prompt Kelompok:**
> "[tulis prompt spesifik, apa adanya, jangan diringkas]"

**Alasan Khusus:**
[mengapa pendekatan/instruksi ini dipilih — kaitkan dengan keputusan arsitektur]

**Output Model:**
[ringkasan skrip atau tempel cuplikan kode krusial]

**Hasil Uji:**
[apa yang terjadi saat kode dijalankan — jalan / error / shape tidak sesuai]

### 2. TAHAP PENANGANAN EROR / DEBUGGING

**Pesan Eror dari Notebook:**
```
[salin pesan error konsol secara UTUH, termasuk traceback]
```

**Prompt Perbaikan ke LLM:**
> "[bagaimana error ditanyakan ke AI]"

**Analisis & Solusi Kelompok:**
[bagaimana AI membantu memperbaiki error, dan apa pelajaran yang diambil]

**Perubahan Kode:**
```diff
- baris lama
+ baris baru
```

### 3. TAHAP OPTIMASI (opsional, isi bila ada)

**Prompt Optimasi:**
> "[prompt]"

**Sebelum vs Sesudah:**
| Metrik | Sebelum | Sesudah |
|---|---|---|
| | | |

**Keputusan:** [dipakai / ditolak, beserta alasan]

---
```

---

## SESI YANG SUDAH TERJADI (contoh nyata — silakan jadikan Entri #1)

### Sesi #1 — Benchmark arsitektur UNet di CPU (pra-proyek, validasi kelayakan)

Ini **bukan** bagian dari vibe coding kelompok — ini eksperimen validasi hardware yang dilakukan
sebelum proyek dimulai. Tetap dicatat karena menentukan keputusan arsitektur (dan menunjukkan proses berpikir).

**Temuan benchmark:**

| batch | waktu/step | epoch @15k |
|---|---|---|
| 64 | 0,394 s | 1,5 menit |
| 128 | 0,640 s | 1,3 menit |

Model: UNet 1,48 M param, input 28×28×1, CPU Ryzen 7 8845HS (16 thread).

**Kesimpulan:** Mini DDPM feasible di CPU tanpa GPU. 20–25 epoch = ±30 menit.

---

### Sesi #2 — Deretan error saat menyusun UNet (MATERI LOG TAHAP 2)

Lima error berikut muncul berurutan saat prototipe UNet disusun. **Semuanya layak dijadikan entri log**,
karena inilah pola error yang akan dihadapi kelompok saat membangun `model.py` sendiri.

#### Eror 2.1 — Time embedding shape mismatch

```
RuntimeError: addmm: mat1 and mat2 shapes cannot be multiplied (64x2 and 128x128)
```

**Penyebab:** time embedding dibuat dari `torch.stack([sin, cos])` yang hanya menghasilkan 2 fitur,
lalu dimasukkan ke `nn.Linear(128, 128)` yang mengharapkan 128 fitur.

**Solusi:** ganti dengan `nn.Embedding(1000, td)` — indeks timestep langsung dipetakan ke vektor
`td`-dimensi. Lebih stabil dan lebih sederhana daripada sinusoidal manual.

**Pelajaran:** lapisan pertama time embedding menentukan lebar vektor; pastikan dimensi input
`nn.Linear` cocok dengan keluaran tahap sebelumnya. Selalu `print(x.shape)` sebelum lapisan baru.

---

#### Eror 2.2 — Argumen `t` tidak diteruskan ke blok

```
TypeError: Blk.forward() missing 1 required positional argument: 't'
```

**Penyebab:** blok konvolusi didefinisikan `forward(self, x, temb)` tetapi dipanggil `blk(x)` saja.

**Solusi:** teruskan `te` di setiap pemanggilan blok.

**Pelajaran:** begitu `forward` diubah signature-nya, semua call site harus ikut diubah. Error ini
muncul karena refactor setengah jalan.

---

#### Eror 2.3 — Channel mismatch di jalur down

```
RuntimeError: Given groups=1, weight of size [128, 128, 3, 3],
expected input[64, 64, 7, 7] to have 128 channels, but got 64 channels instead
```

**Penyebab:** blok `d2` (64→128 channel) lupa dipanggil di `forward`, sehingga `mid` (yang
mengharapkan 128 channel) menerima 64 channel.

**Solusi:** tambahkan `h = self.d2(...)` di antara `d1` dan `mid`.

**Pelajaran:** modul yang dideklarasikan di `__init__` belum tentu terpakai di `forward`. Cara cepat
mendeteksi: cetak bentuk tensor di setiap tahap, atau hitung jumlah modul vs jumlah pemanggilan.

---

#### Eror 2.4 — Skip connection mengambil level yang salah

```
RuntimeError: Given groups=1, weight of size [64, 256, 3, 3],
expected input[64, 192, 14, 14] to have 256 channels, but got 192 channels instead
```

**Penyebab:** `b` (skip connection) diambil dari keluaran `d1` (64 ch) padahal `u2` mengharapkan
pasangan dari `d2` (128 ch). Konkatenasi menghasilkan 192 channel, bukan 256.

**Solusi:** ambil skip connection dari keluaran blok yang resolusinya sama.

**Pelajaran:** di UNet, skip connection **wajib** dipasangkan berdasarkan resolusi spasial, bukan
berdasarkan urutan penulisan kode.

---

#### Eror 2.5 — Ukuran spasial tidak cocok saat upsampling

```
RuntimeError: Sizes of tensors must match except in dimension 1.
Expected size 14 but got size 7 for tensor number 1 in the list.
```

**Penyebab:** jumlah tahap down (3× pool: 28→14→7→3) tidak sepadan dengan jumlah tahap up
(2× upsample: 7→14→28). Tensor 7×7 di-upsample jadi 14×14, tetapi skip yang tersedia 7×7.

**Solusi:** struktur UNet dibuat simetris eksplisit — simpan `h0` (28), `h1` (14), `h2` (7),
lalu naik bertahap `u2` (7, concat `h2`) → `u1` (14, concat `h1`) → `u0` (28, concat `h0`).

**Pelajaran:** gambar 28×28 tidak habis dibagi 2 tiga kali (28→14→7→3). Rencanakan kedalaman
UNet berdasarkan faktorisasi ukuran gambar **sebelum** menulis kode. Ini juga alasan DDPM
umumnya memakai 32×32 (pangkat dua) — keputusan desain, bukan kebetulan.

---

## Ringkasan Pelajaran (diisi di akhir proyek)

1. ...
2. ...
3. ...
