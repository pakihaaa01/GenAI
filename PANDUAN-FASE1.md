# PANDUAN FASE 1 — Membangun `src/data.py` via Vibe Coding

> Dokumen ini **panduan kerja**, bukan kode. Yang dinilai dosen adalah sesi chat kalian
> dengan LLM dan isi `LOG-INTERAKSI.md`, bukan file ini.
>
> **Aturan:** jangan salin prompt di bawah mentah-mentah. Ubah kalimatnya, dan **isi sendiri
> bagian "Alasan Khusus"** — itu yang membedakan kelompok yang benar-benar memandu LLM
> dengan yang sekadar menyalin.

---

## Target akhir Fase 1

Satu file `src/data.py` yang, saat dijalankan `python src/data.py`, mencetak:

```
STATUS: LULUS
```

dengan syarat:
- shape batch = `(128, 1, 28, 28)`
- nilai piksel berada di rentang `[-1, 1]`
- label unik = 10 kelas (0–9)
- tersimpan satu gambar grid di `report/figures/batch_contoh.png`

**Dilarang** membuka `C:\Users\ASUS\Documents\MiniDDPM-REFERENSI\data_referensi.py` sebelum
kalian punya versi sendiri yang jalan. File itu hanya untuk membandingkan di akhir.

---

## Struktur 3 sesi

| Sesi | Target modul | Keluaran |
|---|---|---|
| 1 | Dataset + transform + subset | fungsi `get_dataset()` jalan, 1 sampel bisa diambil |
| 2 | DataLoader + verifikasi bentuk | `get_dataloader()` + `check_batch()` mencetak shape & rentang |
| 3 | Visualisasi batch | `save_grid()` menghasilkan PNG grid |

Satu sesi = satu entri log. Kalau satu sesi butuh 5 kali tanya-jawab, tetap satu entri,
tapi semua error dicatat di bagian Tahap 2.

---

## SESI 1 — Dataset & transform

### Prompt pembuka (ubah kalimatnya)

> Saya membangun model DDPM (denoising diffusion) dari nol untuk gambar Fashion-MNIST 28×28
> grayscale, dan harus jalan di CPU tanpa GPU. Saya butuh modul `data.py` yang isinya fungsi
> `get_dataset(cfg, train=True)` memakai `torchvision.datasets.FashionMNIST`.
>
> Tiga hal yang saya minta kamu jelaskan alasannya, bukan cuma menuliskan kodenya:
> 1. Normalisasi piksel harus ke rentang berapa untuk DDPM, dan kenapa bukan `[0, 1]`?
> 2. Saya mau memakai subset 15.000 dari 60.000 gambar train supaya training di CPU tidak
>    kelamaan. Bagaimana cara mengambil subset yang **deterministik** (seed tetap) supaya
>    eksperimen bisa dibandingkan adil, dan bagaimana memastikan distribusi kelas tetap wajar?
> 3. Apakah saya perlu augmentasi (flip/rotate/crop)? Beri rekomendasi beserta alasannya
>    untuk kasus DDPM, bukan untuk kasus classifier.

### Yang harus kalian cek sendiri setelah kode keluar

- Apakah `transforms.Normalize(mean=[0.5], std=[0.5])` benar menghasilkan `[-1, 1]`?
  Hitung manual: input `[0,1]` → `(0-0.5)/0.5 = -1`, `(1-0.5)/0.5 = 1`. Cocok.
- Apakah subset diambil sebelum atau sesudah transform? (tidak berpengaruh, tapi tahu urutannya penting)
- Apakah `root` dataset absolut atau relatif? Kalau relatif, hasilnya beda tergantung
  dari folder mana kalian jalankan — ini bug nyata yang pernah terjadi di proyek ini.

### Error yang kemungkinan muncul

| Gejala | Penyebab umum |
|---|---|
| `RuntimeError: Dataset not found` / download lambat | `download=True` belum diset, atau koneksi lambat |
| Shape `(3, 28, 28)` bukan `(1, 28, 28)` | transform salah, dataset ter-load sebagai RGB |
| Nilai min/max `0..1` bukan `-1..1` | `Normalize` lupa dipasang |
| Subset berbeda tiap run | `torch.randperm` tanpa `generator` ber-seed |

---

## SESI 2 — DataLoader & verifikasi bentuk

### Prompt pembuka

> Lanjut dari `get_dataset()`. Sekarang saya butuh `get_dataloader(cfg, train=True, shuffle=True)`
> dan fungsi `check_batch(cfg)` yang mengembalikan dict berisi: jumlah batch, shape tensor,
> dtype, nilai min/max/mean/std, shape label, dan daftar label unik.
>
> Konteks penting: saya jalan di **CPU saja**, Windows. Tolong jelaskan pengaruh
> `num_workers`, `pin_memory`, dan `drop_last` untuk kondisi ini — mana yang saya butuhkan
> dan mana yang cuma menambah risiko di Windows?
>
> Saya juga ingin `check_batch` punya dua flag boolean: `shape_ok` (shape ==
> `(batch_size, channels, 28, 28)`) dan `range_ok` (min ≥ -1.01 dan max ≤ 1.01), supaya
> saya bisa jadikan pengecekan otomatis, bukan mata-matain angka.

### Yang harus kalian cek sendiri

- `drop_last=True` → jumlah batch harus **117**, bukan 118. Hitung: `15000 // 128 = 117,18` → 117.
  Kalau keluar 118, `drop_last` belum aktif.
- `num_workers=0` di Windows itu pilihan sadar, bukan default malas — tanya LLM alasannya,
  lalu tulis jawabannya di log dengan kalimat kalian.
- `pin_memory=False` karena CPU-only. Kalau `True`, tidak ada manfaat dan bisa memakan RAM.

### Error yang kemungkinan muncul

| Gejala | Penyebab umum |
|---|---|
| `TypeError: expected Tensor as element 0` | transform belum dipasang di dataset |
| Jumlah batch 118 | `drop_last=False` |
| `BrokenPipeError` / hang di Windows | `num_workers > 0` |
| mean ≈ 0.0 dan std ≈ 1.0 | normalisasi ganda |

---

## SESI 3 — Visualisasi batch

### Prompt pembuka

> Saya butuh fungsi `save_grid(cfg, path=None, n=32)` yang mengambil satu batch dari
> dataloader lalu menyimpannya sebagai grid gambar PNG ke `report/figures/batch_contoh.png`,
> dengan label kelas di atas tiap gambar.
>
> Dua hal yang saya ingin kamu tangani secara eksplisit:
> 1. Tensor saya sudah dinormalisasi ke `[-1, 1]`. Bagaimana cara mengembalikannya ke `[0, 1]`
>    supaya bisa ditampilkan `imshow` tanpa gambar jadi aneh/terbalik?
> 2. Saya ingin fungsi ini tetap jalan walau dipanggil dari direktori mana pun
>    (`python src/data.py` maupun `cd src && python data.py`). Bagaimana cara menentukan
>    path root proyek dari dalam modul itu sendiri?
>
> Gunakan backend matplotlib non-interaktif supaya tidak membuka jendela.

### Yang harus kalian cek sendiri

- Buka PNG-nya. Apakah gambarnya benar pakaian/sepatu/tas, bukan noise abu-abu rata?
  Kalau abu-abu rata → denormalisasi salah.
- Jalankan dari dua folder berbeda. Kalau PNG-nya muncul di dua tempat berbeda → bug path.
- Label di atas gambar harus cocok dengan isi gambar. Cek manual 3–4 gambar:
  0 = kaos, 5 = sandal, 8 = tas, 9 = sepatu boot.

### Error yang kemungkinan muncul

| Gejala | Penyebab umum |
|---|---|
| `UserWarning: Matplotlib is currently using agg` | backend belum diset sebelum `pyplot` diimpor |
| PNG muncul di folder aneh (`src/report/...`) | path relatif ke cwd, bukan ke root proyek |
| Gambar terlalu gelap/terang | denormalisasi lupa `* 0.5 + 0.5` |
| `TypeError: Invalid shape ... for image data` | tensor masih 3D, perlu `x[i, 0]` bukan `x[i]` |

---

## Setelah `STATUS: LULUS` keluar

1. Commit: `git add src/data.py && git commit -m "Fase 1: data pipeline (vibe coding, N sesi)"`
2. Baru **boleh** buka `data_referensi.py` dan bandingkan. Tulis di laporan: apa yang
   beda, dan kenapa versi kalian memilih begitu. Perbedaan yang bisa dijelaskan itu **nilai plus**,
   bukan kesalahan.
3. Tempel `src/data.py` hasil kalian ke chat ini untuk saya review.

---

## Isi minimum tiap entri log (checklist)

- [ ] Tanggal sesi
- [ ] Target modul
- [ ] Tautan sesi chat (uji: buka di mode incognito, masih bisa diakses?)
- [ ] Prompt asli, apa adanya
- [ ] **Alasan Khusus** — kalimat kalian sendiri
- [ ] Ringkasan output LLM / cuplikan kode krusial
- [ ] Hasil uji (jalan / error / shape salah)
- [ ] Kalau error: pesan utuh + prompt perbaikan + analisis + pelajaran
- [ ] Kalau tidak ada error: **tetap tulis** apa yang kalian verifikasi dan kenapa itu cukup
