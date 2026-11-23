# Rencana kerja Fase 1

Fase 1 isinya satu file: `src/data.py`. Tugasnya ambil Fashion-MNIST, normalisasi, bagi jadi batch,
terus simpan satu gambar contoh biar kelihatan datanya benar.

Kita kerjakan dalam tiga sesi biar tiap langkah bisa dites sendiri-sendiri. Satu sesi satu entri log.

## Sesi 1 — dataset

Yang dibikin: fungsi `get_dataset()`.

Isinya ambil FashionMNIST lewat torchvision, pasang transform, lalu potong jadi 15.000 gambar
dari 60.000. Pemotongan pakai seed tetap supaya tiap run dapat gambar yang sama. Tanpa itu,
perbandingan antar eksperimen jadi tidak adil karena datanya beda-beda.

Normalisasi ke rentang -1 sampai 1, bukan 0 sampai 1. Alasannya karena proses diffusion
menambah noise yang rata-ratanya nol. Kalau data masih di 0 sampai 1, model melihat noise
jauh lebih besar daripada sinyalnya.

Selesai kalau: satu sampel bisa diambil, bentuknya (1, 28, 28), nilainya antara -1 dan 1.

## Sesi 2 — dataloader

Yang dibikin: `get_dataloader()` dan `check_batch()`.

`check_batch()` balikin jumlah batch, bentuk tensor, nilai min/max/mean/std, sama daftar label.
Dua penanda yang penting: `shape_ok` dan `range_ok`. Dua-duanya harus `True`.

Angka yang gampang dipakai buat cek cepat: jumlah batch harus 117. Hitungannya 15000 dibagi 128,
sisa ke bawah. Kalau keluar 118, berarti batch terakhir yang isinya kurang dari 128 masih ikut.

Di Windows `num_workers` sebaiknya 0. Kalau dinaikkan, proses kadang menggantung atau
kena BrokenPipeError. `pin_memory` matikan saja karena kita tidak pakai GPU.

Selesai kalau: `python src/data.py` mencetak `STATUS: LULUS`.

## Sesi 3 — gambar contoh

Yang dibikin: `save_grid()`.

Ambil satu batch, kembalikan nilainya ke 0 sampai 1, susun jadi grid, simpan ke
`report/figures/batch_contoh.png`, kasih label kelas di atas tiap gambar.

Perhatikan soal path. Kalau path-nya relatif, hasilnya ikut folder tempat perintah dijalankan.
Waktu saya coba dulu, gambarnya nyasar ke `src/report/figures/` gara-gara ini. Solusinya hitung
root proyek dari lokasi file modulnya sendiri, bukan dari folder kerja.

Selesai kalau: PNG-nya benar berisi pakaian dan sepatu, bukan abu-abu rata, dan tetap muncul
di tempat yang sama walaupun dijalankan dari folder berbeda.

## Setelah semua lulus

Commit `src/data.py`, lalu isi log untuk ketiga sesi. Baru setelah itu kita bandingkan hasilnya
dengan catatan pembanding yang saya simpan di luar repo.

## Catatan

Kalau satu sesi butuh lima kali tanya-jawab dengan LLM, tetap dihitung satu entri. Semua error
yang muncul masuk ke bagian penanganan error di entri yang sama, tidak perlu dipecah.
