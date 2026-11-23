# Cara kerja git buat dua orang

Repo ini harus kelihatan dikerjakan dua orang. Yang dinilai bukan jumlah commit, tapi
apakah commit-nya mencerminkan kerja nyata masing-masing. Jangan ada commit yang dibuat
satu orang tapi diberi nama orang lain — itu ketahuan dari pola waktunya, dan kalau
ditanya saat presentasi langsung ketahuan.

## Langkah 1 — pilih cara berbagi repo

Repo sekarang masih lokal di laptop kamu. Ada dua pilihan:

**A. Lewat GitHub (paling rapi, disarankan)**

1. Bikin repo baru di GitHub, set **Private** (jangan public, ada `LOG-INTERAKSI.md` isinya obrolan)
2. Di laptop kamu:
   ```bash
   cd "/c/Users/ASUS/Documents/MiniDDPM-FashionMNIST"
   git remote add origin https://github.com/USERNAME/miniddpm-fashionmnist.git
   git branch -M main
   git push -u origin main
   ```
3. Di laptop rekanmu:
   ```bash
   git clone https://github.com/USERNAME/miniddpm-fashionmnist.git
   ```
4. Tambahkan dia sebagai collaborator di Settings → Collaborators

**B. Lewat file (kalau belum mau pakai GitHub)**

Zip foldernya, kirim lewat WhatsApp/Drive, dia extract. Kekurangannya: nanti harus saling
kirim ulang tiap kali ada perubahan, dan gampang bentrok. Pakai ini cuma kalau GitHub
benar-benar tidak bisa.

## Langkah 2 — set identitas masing-masing

Ini wajib dan harus beda. Di laptop rekanmu:

```bash
cd miniddpm-fashionmnist
git config --local user.name "Nama Rekanmu"
git config --local user.email "email.rekanmu@gmail.com"
```

Cek hasilnya:

```bash
git config --local user.name
git config --local user.email
```

Kalau dua-duanya sama dengan punya kamu, commit-nya bakal kebaca satu orang.

## Langkah 3 — pembagian kerja yang menghasilkan commit nyata

Modul inti (`src/`) sudah selesai dikerjakan lewat 5 sesi Gemini. Yang tersisa harus dikerjakan
rekanmu supaya kontribusinya nyata, bukan cuma ikut nama:

| Sesi / modul | Pemilik | Status |
|---|---|---|
| `src/data.py` — `get_dataset()` | kamu | ✅ selesai (sesi 1) |
| `src/data.py` — `get_dataloader()` + `check_batch()` | kamu | ✅ selesai (sesi 2) |
| `src/diffusion.py` — kelas `DDPM` | kamu | ✅ selesai (sesi 3) |
| `src/model.py` — U-Net | kamu | ✅ selesai (sesi 4) |
| `src/train.py` + `src/sample.py` | kamu | ✅ selesai (sesi 5) |
| **`src/evaluate.py` — FID / histogram piksel** | **rekanmu** | ⬜ belum, sesi 6 |
| **`src/ablation.py` — banding schedule / base_ch** | **rekanmu** | ⬜ belum, sesi 7 |
| **`report/figures/` — kurva loss + grid progres** | **rekanmu** | ⬜ belum, sesi 8 |
| **Entri log sesi 6–8 di `LOG-INTERAKSI.md`** | **rekanmu** | ⬜ belum |
| **`deliverable/LAPORAN-PROYEK1.md` — rakit final** | **kamu** | ⬜ belum, sesi 9 |

Tiga sesi terakhir (6–8) **wajib** dibuka rekanmu sendiri di Gemini dengan akunnya sendiri, lalu
prompt + jawaban + error-nya dicatat sebagai entri baru di `LOG-INTERAKSI.md`. Kalau dia cuma
menerima kode jadi dari kamu lalu commit, log-nya kosong dan itu langsung kelihatan saat dinilai.

Ide isi sesi 6–8 (pilih minimal dua):

- **Evaluasi kuantitatif** — hitung histogram intensitas piksel gambar asli vs gambar hasil
  sampling, atau FID sederhana pakai fitur dari model yang sudah dilatih. Tujuannya supaya ada
  angka, bukan cuma "kelihatan mirip".
- **Ablation schedule** — jalankan 3 epoch dengan `beta_schedule="linear"` vs `"cosine"`, bandingkan
  kurva loss. Cukup 3 epoch, jangan 25.
- **Ablation kapasitas** — `unet_base=32` vs `unet_base=48`, bandingkan jumlah parameter dan loss.
- **Grid progres epoch** — ambil checkpoint epoch 1/5/10/25, susun jadi satu gambar dengan noise
  awal yang sama, jelaskan kenapa noise-nya harus dikunci.

Yang penting: setiap sesi menghasilkan commit nyata dari akun rekanmu.

## Langkah 4 — aturan commit

Commit kecil, sering, pesan jelas. Contoh yang bagus:

```bash
git commit -m "sesi 2: get_dataloader + check_batch, batch=117 terverifikasi"
git commit -m "sesi 2: fix drop_last, batch jadi 117 dari 118"
git commit -m "log: entri sesi 2 (error num_workers Windows)"
```

Contoh yang buruk:

```bash
git commit -m "update"
git commit -m "fix"
git commit -m "final banget ini mah"
```

Jangan satu commit raksasa di akhir proyek. Pola waktu commit juga dilihat — kalau semua
commit keluar di hari yang sama persis, itu tanda rekonstruksi.

## Langkah 5 — hindari bentrok

Sebelum mulai kerja, selalu tarik perubahan terbaru:

```bash
git pull
```

Kalau lupa dan muncul konflik, jangan panik dan jangan `git checkout .` (itu membuang
pekerjaan orang). Kirim isi pesan errornya ke chat ini.

Jangan dua orang menyentuh file yang sama di waktu yang sama. Karena itu pembagian di
Langkah 3 dipisah per fungsi, bukan per file.

## Cara cek hasil akhirnya

Di akhir proyek, jalankan:

```bash
git shortlog -sne
```

Keluarannya jumlah commit per orang. Kalau cuma muncul satu nama, berarti ada yang salah
dan harus dibenerin sebelum submit.

```bash
git log --oneline --graph --all
```

Lihat alurnya masuk akal atau tidak: ada selang waktu, ada commit perbaikan, bukan
semuanya numpuk di satu jam.

## Yang jangan dilakukan

- `git commit --author="Nama Rekanmu"` — ini memalsukan penulis. Ketahuan dari email yang
  tidak terdaftar di akun GitHub-nya.
- Bikin commit kosong (`--allow-empty`) cuma biar jumlahnya banyak.
- Commit semua file di akhir proyek sekaligus dengan pesan "selesai".
