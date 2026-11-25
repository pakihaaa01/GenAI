# BRIEF UNTUK HERMES — Eksekusi Sesi 6–8 (Proyek 1 CAK4KBB3)

Dokumen ini dipakai oleh **Hermes milik Mia**. Isinya instruksi operasional lengkap:
apa yang dikerjakan, urutannya, dan batasan yang tidak boleh dilanggar.

**Repo:** https://github.com/pakihaaa01/GenAI (Private, Mia sudah collaborator)
**Pemilik pekerjaan:** Mia. Hermes bertindak sebagai operator teknis, bukan penulis konten.

---

## 0. Ringkas konteks

Repo berisi implementasi DDPM (diffusion model) dari nol untuk Fashion-MNIST 28×28
grayscale, dibangun lewat vibe coding. Anggota pertama (Faqih) sudah menyelesaikan
`src/data.py`, `diffusion.py`, `model.py`, `train.py`, `sample.py`, `plot_loss.py`
dan training 25 epoch (loss 0.16831 → 0.04307, model 793.953 parameter).

Tugas Mia adalah **sesi 6, 7, dan 8**: `src/evaluate.py`, `src/ablation.py`, dan figur
tambahan untuk laporan.

Rubrik penilaian 100 poin:

| Kriteria | Bobot |
|---|---|
| Vibe Coding / orisinalitas prompt | 30% |
| Log Interaksi LLM | 25% |
| Fungsionalitas | 25% |
| Kerjasama tim (git) | 20% |

55% nilai berasal dari dokumentasi. Kode yang jalan tanpa log yang jujur tetap
kehilangan separuh nilai.

---

## 1. Aturan yang mengikat (jangan dilanggar)

1. **Jangan mengarang error.** Kalau kode dari Gemini langsung jalan, tulis
   "tidak ada error pada sesi ini". Kalau error, salin traceback utuh dari konsol.
2. **Jangan mengarang isi percakapan Gemini.** Setiap entri log harus berasal dari
   percakapan yang benar-benar terjadi dan bisa dibuka lewat share link.
3. **Jangan `git commit --author=`** dan **jangan `git commit --allow-empty`**.
   Atribusi commit harus jujur: nama Mia, karena Mia yang mengoperasikan sesi.
4. **Jangan menyentuh** `src/data.py`, `src/diffusion.py`, `src/model.py`,
   `src/train.py`, `src/sample.py`, `src/plot_loss.py`, `src/config.py`. Milik Faqih,
   sudah diverifikasi. Kalau butuh nilai konfigurasi, baca saja, jangan ubah.
5. **Jangan copy-paste kode utuh** dari GitHub, blog, atau tutorial. Kode berasal dari
   jawaban Gemini yang diminta lewat prompt di repo ini.
6. **Jangan menulis prompt atas nama Mia tanpa sepengetahuannya.** Sebelum setiap prompt
   dikirim ke Gemini, tampilkan drafnya ke Mia dan minta persetujuan. Mia boleh mengubah
   kalimatnya. Ini yang menjaga kriteria 30%.
7. **Kredensial tidak disimpan.** Personal Access Token GitHub, kata sandi, kode 2FA
   tidak boleh ditulis ke file, commit, log, atau ringkasan mana pun. Kalau git meminta
   token, minta Mia mengetiknya sendiri.

---

## 2. Prasyarat yang harus dicek dulu

Sebelum mulai, pastikan:

| Prasyarat | Cara cek |
|---|---|
| Python 3.10+ | `python --version` |
| git | `git --version` |
| Dependency | `python -c "import torch, torchvision, matplotlib"` |
| Mia sudah login Gemini di Chrome | buka `https://gemini.google.com` di browser, lihat ada nama akun Mia |
| Akses repo | `git ls-remote https://github.com/pakihaaa01/GenAI.git` harus keluar hash |
| `checkpoints/ddpm_best.pt` ada | `ls -la checkpoints/` — kalau kosong, minta Mia mengambil file dari Faqih |

Kalau salah satu gagal, **berhenti dan laporkan ke Mia**. Jangan mengakali dengan
membuat checkpoint palsu atau melewati evaluasi.

---

## 3. Tahap setup

### 3.1 Clone

```bash
cd ~/Documents
git clone https://github.com/pakihaaa01/GenAI.git
cd GenAI
```

Kalau repo Private dan git meminta kredensial: minta Mia mengetik username GitHub-nya
dan PAT (scope `repo`, dibuat di https://github.com/settings/tokens). Hermes tidak
mengetik token itu sendiri dan tidak menyimpannya.

### 3.2 Identitas git

Ini menentukan 20% nilai. **Tanyakan ke Mia** nama dan email yang dipakai di akun
GitHub-nya, lalu:

```bash
git config --local user.name "<nama Mia>"
git config --local user.email "<email GitHub Mia>"
```

Verifikasi sebelum commit pertama:

```bash
git config --local user.email   # harus email Mia, bukan anakabah142@gmail.com
```

Kalau masih menampilkan `anakabah142@gmail.com` atau `pakihaaa01`, hentikan pekerjaan
dan perbaiki dulu. Commit dengan identitas salah membuat kontribusi Mia tidak terhitung.

### 3.3 Dependency dan dataset

```bash
python -m pip install -r requirements.txt
python src/data.py
```

`src/data.py` harus mencetak `STATUS: LULUS`. Ini sekaligus mengunduh Fashion-MNIST
(21 MB) ke `data/`.

### 3.4 Baca konteks

Baca berurutan, jangan dilewat:

1. `BRIEF-MIA.md` — panduan kerja Mia, bagian 5–8 berisi detail tiga sesi
2. `TEAM-GIT.md` — aturan commit dan pembagian kerja
3. `LOG-INTERAKSI.md` — **tiru format entri Sesi #1 sampai #5**
4. `src/sample.py`, `src/config.py`, `src/diffusion.py` — API yang akan dipakai

Ringkas ke Mia: isi tiga tugas, format log yang harus diikuti, dan nama field konfigurasi
yang tersedia (`ckpt_dir`, `fig_dir`, `num_sample_images`, `beta_schedule`, `unet_base`,
`timesteps`, `subset_size`, dan seterusnya).

---

## 4. Alur kerja per sesi

Tiga sesi, satu sesi = satu percakapan Gemini baru = satu entri log = satu commit.
Jangan gabungkan beberapa sesi dalam satu percakapan.

### 4.1 Urutan operasi

Untuk setiap sesi bernomor N:

**Langkah 1 — Siapkan bahan prompt.**
Baca file sumber yang relevan, ambil signature asli (nama fungsi, parameter, return,
nama field `Config`). Ini yang paling menentukan: empat dari lima error di proyek ini
terjadi karena LLM tidak diberi signature yang benar.

**Langkah 2 — Susun draf prompt, minta persetujuan Mia.**
Tampilkan draf ke Mia dalam blok kode. Draf wajib memuat:

- konteks: DDPM Fashion-MNIST 28×28 grayscale, sudah dilatih
- apa yang diminta, spesifik per fungsi
- **kebingungan yang jujur** — 2–3 pertanyaan teknis yang benar-benar relevan
- signature asli yang ditempel verbatim
- batasan: jangan ubah file lain, jangan pakai library diffusion siap pakai

Tunggu Mia bilang setuju atau mengubahnya. Kalimat akhir prompt harus terasa seperti
tulisan Mia, bukan template.

**Langkah 3 — Kirim ke Gemini lewat browser.**

Buka `https://gemini.google.com/app`, mulai percakapan baru, tempel prompt, kirim.

Teknis yang sudah terbukti bekerja di proyek ini:

- Gemini menolak memproses kiriman kalau halaman dianggap tidak aktif. Paksa dengan
  `cdp('Emulation.setFocusEmulationEnabled', enabled=True)` sebelum mengirim.
- Isi kotak prompt lewat event `input` + `beforeinput`, bukan dengan mengubah
  `textContent` saja. Kalau tombol kirim tidak aktif, event belum terpicu.
- Setelah kirim, tunggu jawaban selesai dengan memantau pertumbuhan
  `document.body.innerText.length` dan keberadaan tombol berhenti
  (`/Hentikan|Stop/`). Ulangi pemeriksaan setiap ±10 detik, hentikan setelah tiga kali
  berturut-turut tidak ada perubahan.
- Kalau halaman menjadi lambat atau tidak responsif, buka percakapan baru dan
  lanjutkan di sana. Percakapan lama tetap tersimpan.

**Langkah 4 — Ambil share link.**
Di percakapan yang sama: klik tombol menu (`aria-label` mengandung
`Buka menu untuk tindakan percakapan`), pilih item menu `Bagikan percakapan`,
lalu baca tautan dari `[role="dialog"]`. Simpan tautannya. Verifikasi dengan membuka
link itu di tab baru: isinya harus percakapan yang benar.

**Langkah 5 — Tulis file.**
Tulis jawaban Gemini ke file target apa adanya. Jangan "rapikan" logika, jangan
menambal sendiri kalau ada yang janggal. Bagian yang janggal justru bahan prompt
perbaikan berikutnya. Sesuaikan hanya yang wajib agar konsisten dengan repo:

- import absolut `from src.X import ...`
- bootstrap `sys.path` di level modul (bukan di dalam `__main__`), karena
  `python src/evaluate.py` menaruh `src/` di `sys.path[0]`, bukan root proyek
- nama field `Config` yang benar (`ckpt_dir`, bukan `checkpoint_dir`; `fig_dir`;
  `data_root`, bukan `data_dir`)
- `num_workers=0` di Windows

**Langkah 6 — Jalankan, tangkap error asli.**

```bash
python src/evaluate.py
```

Kalau error: salin traceback **utuh** dari konsol. Jangan diringkas, jangan diperbaiki
manual dulu. Error itu yang dilaporkan ke Gemini.

**Langkah 7 — Kirim error ke Gemini, minta perbaikan.**
Sertakan traceback utuh, isi file yang error, dan signature modul lain yang dipanggil.
Kalau Gemini meminta signature file lain, kirim apa adanya. Ambil share link baru untuk
pertukaran ini.

**Langkah 8 — Ulangi langkah 5–7 sampai jalan.** Biasanya 1–3 putaran.

**Langkah 9 — Tulis entri log.**
Tambahkan ke `LOG-INTERAKSI.md` di bawah Sesi #5, format mengikuti entri sebelumnya:

```markdown
## Sesi #N — <tanggal>

- **Target Modul:** `src/<file>.py`
- **Operator:** <nama Mia>
- **Tautan Sesi Chat:** <share link, satu baris per pertukaran kalau ada beberapa>
- **Model LLM:** Gemini

### 1. TAHAP INISIASI (Meminta Kode Awal)

**Prompt Kelompok:**

> "<prompt verbatim, termasuk salah tulis kalau ada>"

**Alasan Khusus:**
<alasan teknis pemilihan pendekatan, apa yang membingungkan>

**Output Model:**
<ringkasan jawaban: fungsi apa yang dibuat, pendekatan apa yang dipilih>

**Hasil Uji:**
<output konsol asli: angka, nama file gambar yang dihasilkan>

### 2. TAHAP PENANGANAN EROR / DEBUGGING

<"Tidak ada error pada sesi ini" kalau memang begitu.
Kalau ada: traceback utuh, prompt perbaikan, analisis, hasil uji ulang>
```

Perbarui juga tabel daftar sesi di bagian atas `LOG-INTERAKSI.md`.

**Langkah 10 — Commit dan push.**

```bash
git add src/<file>.py report/figures/ LOG-INTERAKSI.md
git commit -m "sesi N: <file> - <ringkasan satu baris>"
git push
```

Kalau push ditolak karena ketinggalan:

```bash
git pull --rebase
git push
```

Kalau muncul konflik, jangan `git checkout .` (itu membuang kerjaan orang). Laporkan
isi konfliknya ke Mia.

---

## 5. Isi tiap sesi

### Sesi 6 — `src/evaluate.py`

Tujuan: memberi angka, bukan hanya kesan visual.

Bahan yang perlu dibaca: `src/sample.py` (cara generate sampel), `src/config.py`
(nama field), `checkpoints/ddpm_best.pt` (harus ada).

Arah prompt: minta evaluasi kuantitatif hasil sampling. Kebingungan yang wajar
untuk diangkat:

- FID lazim dipakai untuk diffusion, tetapi InceptionV3 mengharapkan masukan 299×299
  RGB. Dataset ini 28×28 grayscale. Apakah FID masih bermakna setelah resize dan
  replikasi channel, atau justru menyesatkan?
- Kalau FID tidak cocok, metrik apa yang lebih jujur untuk kasus ini? Histogram
  intensitas piksel antara data asli dan hasil sampling, misalnya — cukup ilmiah
  atau tidak?
- Bagaimana membaca hasilnya sehingga bisa disimpulkan model bagus atau belum?

Keluaran yang diharapkan: `python src/evaluate.py` berjalan tanpa error, mencetak
minimal satu angka metrik, dan menghasilkan minimal satu gambar di `report/figures/`.

### Sesi 7 — `src/ablation.py`

Tujuan: membandingkan dua konfigurasi, membuktikan pilihan di `config.py` masuk akal.

Pilih **satu** opsi, jangan keduanya:

- **Opsi A:** `beta_schedule` linear (1e-4 → 0.02) versus cosine. Cek dulu apakah
  `src/diffusion.py` mendukung cosine; kalau belum, itu bahan prompt yang bagus.
- **Opsi B:** `unet_base=32` (793.953 parameter) versus `unet_base=48`.

Cukup **3 epoch per konfigurasi**, bukan 25. Tujuannya melihat tren, bukan hasil final.
Perkirakan waktu: sekitar 4 menit per epoch di CPU, jadi 2 × 3 epoch ≈ 25 menit.
Jalankan di latar belakang dan laporkan progres ke Mia.

Keluaran: `src/ablation.py` berjalan, tabel atau grafik perbandingan di
`report/figures/`.

Catatan: checkpoint hasil ablation **jangan** menimpa `checkpoints/ddpm_best.pt`.
Pakai subfolder atau nama berkas berbeda.

### Sesi 8 — figur tambahan

Yang sudah ada, jangan dibuat ulang: `loss_curve.png`,
`progression_epochs_1_to_25.png`, `samples_epoch_001..025.png`.

Yang bisa ditambahkan:

- grid perbandingan citra asli dataset di baris atas dan hasil sampling di baris bawah
- histogram overlay intensitas piksel asli versus hasil
- grid ablasi: dua kurva loss dalam satu plot

Keluaran: minimal dua berkas baru di `report/figures/`.

---

## 6. Yang harus dilaporkan ke Mia

Setelah tiap sesi selesai, sampaikan ringkas:

- berkas yang dibuat atau diubah
- hasil eksekusi nyata (angka, nama gambar)
- share link percakapan
- nomor commit
- error yang muncul beserta cara mengatasinya

Di akhir ketiga sesi:

```
git log --author="<email Mia>" --oneline
```

Harus keluar minimal tiga commit atas nama Mia. Kalau nol, ada yang salah pada
`git config --local user.email` dan harus diperbaiki sebelum menyerahkan hasil.

---

## 7. Kalau ada yang tidak bisa dikerjakan

Lebih baik dilaporkan daripada ditambal. Kondisi yang harus dihentikan dan
dikomunikasikan:

- Gemini tidak bisa diakses atau sesi login Mia tidak tersedia
- `checkpoints/ddpm_best.pt` tidak ada
- git push ditolak dan penyebabnya bukan sekadar ketinggalan commit
- konflik merge yang melibatkan file milik Faqih
- waktu eksekusi ablation jauh melebihi perkiraan (misalnya lebih dari satu jam)

Untuk kasus terakhir, kurangi jumlah epoch menjadi 2, bukan memalsukan hasil.

---

## 8. Satu hal yang perlu diingat

Nilai tertinggi proyek ini ada pada kejujuran dokumentasi. Percakapan Gemini yang
berantakan, prompt yang awalnya salah, error yang butuh tiga kali perbaikan — semua itu
justru bukti proses yang dinilai. Yang merugikan adalah log yang terlalu rapi dan tidak
bisa dipertanggungjawabkan saat ditanya.
