# Rencana kerja

Proyek ini dibangun modul per modul. Satu sesi dengan LLM menghasilkan satu modul, dan satu sesi
langsung dicatat sebagai satu entri di `LOG-INTERAKSI.md`. Tidak ada entri yang ditulis belakangan.

Urutan modul sengaja dari yang paling dasar ke yang paling bergantung: data dulu (bisa dites tanpa
model), lalu proses diffusion (bisa dites tanpa jaringan saraf), lalu U-Net (bisa dites tanpa
training), baru training dan sampling.

## Sesi 1 — dataset

Yang dibikin: `get_dataset()` di `src/data.py`.

Isinya ambil FashionMNIST lewat torchvision, pasang transform, lalu potong jadi 15.000 gambar dari
60.000. Pemotongan pakai seed tetap supaya tiap run dapat gambar yang sama. Tanpa itu, perbandingan
antar eksperimen jadi tidak adil karena datanya beda-beda.

Normalisasi ke rentang -1 sampai 1, bukan 0 sampai 1. Alasannya karena proses diffusion menambah
noise yang rata-ratanya nol. Kalau data masih di 0 sampai 1, model melihat noise jauh lebih besar
daripada sinyalnya.

Selesai kalau: satu sampel bisa diambil, bentuknya (1, 28, 28), nilainya antara -1 dan 1.

## Sesi 2 — dataloader

Yang dibikin: `get_dataloader()` dan `check_batch()`.

`check_batch()` mencetak jumlah batch, bentuk tensor, nilai min/max/mean/std, dan daftar label, lalu
menutup dengan `STATUS: LULUS` atau `STATUS: GAGAL` plus daftar masalahnya.

Angka yang gampang dipakai buat cek cepat: jumlah batch harus 117. Hitungannya 15000 dibagi 128,
sisa ke bawah. Kalau keluar 118, berarti batch terakhir yang isinya kurang dari 128 masih ikut.

Di Windows `num_workers` sebaiknya 0. Kalau dinaikkan, proses kadang menggantung atau kena
BrokenPipeError. `pin_memory` matikan saja karena training di CPU.

Selesai kalau: `python src/data.py` mencetak `STATUS: LULUS`.

## Sesi 3 — proses diffusion

Yang dibikin: `src/diffusion.py`, kelas `DDPM`.

Isinya jadwal noise linear 1e-4 sampai 0.02 dengan T=1000, fungsi `q_sample` yang bisa lompat
langsung dari x0 ke xt, `p_losses` dengan target noise epsilon, dan `p_sample_loop` untuk
membangkitkan gambar dari noise murni.

Yang perlu dijelasin di laporan: kenapa targetnya epsilon bukan x0, dan dari mana asal sigma_t
(posterior variance, bukan beta mentah).

Selesai kalau: `python src/diffusion.py` jalan dan std di t=0 kira-kira sama dengan std data,
sedangkan std di t=999 mendekati 1.

## Sesi 4 — arsitektur U-Net

Yang dibikin: `src/model.py`.

U-Net dipilih karena tugasnya regresi per piksel, bukan klasifikasi — output harus seukuran input,
dan skip connection menjaga detail spasial tidak hilang di bottleneck. Downsampling berhenti di
7x7 karena 28 ke 14 ke 7 pas, sedangkan 7 ke 3,5 tidak bisa tanpa padding.

Target ±1,5 juta parameter. Timestep masuk lewat encoding sinusoidal, normalisasi pakai GroupNorm
(bukan BatchNorm) karena statistik lintas-batch tidak stabil saat batch kecil dan saat sampling.

Selesai kalau: `python src/model.py` mencetak jumlah parameter dan bentuk output sama dengan input.

## Sesi 5 — training dan sampling

Yang dibikin: `src/train.py` dan `src/sample.py`.

Training loop menyimpan checkpoint tiap epoch, mencatat loss per epoch ke JSON, dan memakai AdamW
dengan gradient clipping. Sampling memuat checkpoint lalu membangkitkan grid gambar; grid progres
antar-epoch memakai noise awal yang sama persis supaya perbedaan hasil benar-benar berasal dari
perbaikan model, bukan dari noise yang beda.

Selesai kalau: training jalan beberapa epoch tanpa error dan keluar file PNG grid gambar.

## Setelah semua lulus

1. Jalankan training penuh (Colab GPU T4, atau CPU ±33 menit untuk 25 epoch).
2. Jalankan `python src/sample.py` → grid progres.
3. Ekspor lampiran commit dan kode (lihat `deliverable/CARA-EKSPOR.md`).
4. Rakit jadi satu file `.md`.

## Catatan

Kalau satu sesi butuh lima kali tanya-jawab dengan LLM, tetap dihitung satu entri. Semua error yang
muncul masuk ke bagian penanganan error di entri yang sama, tidak perlu dipecah.

Dua hal yang paling sering bikin error dan sudah terbukti di proyek ini:

- **Nama field/class ditebak LLM.** Tempel isi `config.py` dan signature fungsi secara verbatim di
  prompt, jangan diringkas jadi "sudah jadi".
- **Path relatif.** Selalu hitung root proyek dari lokasi file modul (`Path(__file__).resolve()`),
  bukan dari folder kerja.
