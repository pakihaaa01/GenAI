# BRIEF UNTUK MIA — Proyek 1 Vibe Coding (CAK4KBB3)

Halo Mia. Ini panduan lengkap buat bagian kamu. Baca dari atas sampai bawah dulu, baru mulai.

**Repo:** https://github.com/pakihaaa01/GenAI (Private — kamu harus jadi collaborator dulu)

---

## 0. Apa yang lagi kita kerjakan

Kita bikin **Mini DDPM** (diffusion model) dari nol buat generate gambar Fashion-MNIST 28×28.
Semua kode **wajib** dibangun lewat vibe coding — ngobrol sama LLM (Gemini), bukan copy-paste dari GitHub.

Yang **sudah selesai** (jangan disentuh, punya Faqih):
- `src/data.py` — dataset + dataloader
- `src/diffusion.py` — noise schedule, forward/reverse process
- `src/model.py` — U-Net 793.953 parameter
- `src/train.py` — training loop
- `src/sample.py` — sampling dari checkpoint
- Training 25 epoch sudah jalan, loss 0.168 → 0.043

Yang **jadi bagian kamu** (3 sesi):
- `src/evaluate.py` — evaluasi kuantitatif hasil sampling
- `src/ablation.py` — eksperimen banding konfigurasi
- `report/figures/` — figur tambahan untuk laporan

---

## 1. Nilai apa yang diambil dari kerjaan kamu

Rubrik penilaian proyek ini 100 poin:

| Kriteria | Bobot | Dari mana |
|---|---|---|
| Vibe Coding / Orisinalitas prompt | 30% | prompt kamu di Gemini |
| **Log Interaksi LLM** | **25%** | entri sesi 6–8 yang kamu tulis di `LOG-INTERAKSI.md` |
| Fungsionalitas | 25% | kode kamu jalan |
| **Kerjasama Tim (git)** | **20%** | commit kamu di repo, atas nama kamu sendiri |

Jadi **55% nilai proyek ini datang dari dokumentasi**. Kalau kamu cuma commit kode tanpa log, dua kriteria langsung jeblok.

**Yang paling penting:** kamu **wajib** buka Gemini pakai akun kamu sendiri untuk sesi 6–8. Kalau kode kamu terima jadi dari Faqih lalu kamu commit, log-nya kosong dan itu kelihatan banget saat dinilai.

---

## 2. Setup awal (sekali saja)

### 2.1 Clone repo

Repo ini Private. `git clone` bakal minta username + password, dan **password-nya bukan password GitHub** — harus Personal Access Token (PAT).

Bikin token:
1. Buka https://github.com/settings/tokens
2. **Generate new token (classic)**
3. Nama: `colab-miniddpm`, Expiration: 30 days
4. Centang scope **`repo`** (satu-satunya yang perlu)
5. **Generate token** → copy (cuma tampil sekali, simpan dulu)

Clone:

```bash
git clone https://github.com/pakihaaa01/GenAI.git
cd GenAI
```

Username: username GitHub kamu. Password: tempel token tadi.

### 2.2 Set identitas git (WAJIB, jangan dilewat)

Ini harus nama + email kamu sendiri, **bukan** punya Faqih. Kalau salah, semua commit kamu kebaca atas nama dia dan kontribusi kamu nggak dihitung.

```bash
git config --local user.name "Mia <nama-lengkap-kamu>"
git config --local user.email "email-yang-dipakai-di-github-kamu@gmail.com"
```

Cek:

```bash
git config --local user.name
git config --local user.email
```

Harus keluar nama kamu, bukan `pakihaaa01`.

### 2.3 Install dependency

```bash
python -m pip install -r requirements.txt
```

Itu install torch CPU (~2 GB), torchvision, numpy, matplotlib, tqdm. Sabar, agak lama.

Kalau laptop kamu lemot atau storage mepet, pakai Google Colab saja — buka `colab/jalankan_di_colab.ipynb`, jalankan selnya berurutan.

### 2.4 Download dataset (sekali)

```bash
python src/data.py
```

Harus cetak `STATUS: LULUS`. Ini sekalian download Fashion-MNIST 21 MB.

### 2.5 Cek checkpoint ada

Kamu butuh model terlatih buat evaluasi. Cek folder `checkpoints/`:

```bash
ls checkpoints/
```

Kalau kosong, minta Faqih kirim `ddpm_best.pt` (bobot model, ~3 MB). File ini sengaja tidak masuk git karena besar.

---

## 3. Baca konteks dulu (10 menit)

Tiga file ini wajib dibaca sebelum mulai:

| File | Isi |
|---|---|
| `TEAM-GIT.md` | aturan commit, pembagian kerja, yang dilarang |
| `LOG-INTERAKSI.md` | contoh format entri log — **tiru gaya ini** |
| `RENCANA.md` | rencana kerja per sesi |

Perhatikan `LOG-INTERAKSI.md` bagian Sesi #1: ada **Prompt Kelompok** (verbatim), **Alasan Khusus**, **Output Model**, **Hasil Uji**, lalu **Tahap Penanganan Eror** kalau ada error. Format itu yang kamu pakai.

---

## 4. Aturan main vibe coding

Ini bukan formalitas — ini yang dinilai.

1. **Buka Gemini pakai akun kamu sendiri.** Setiap sesi = satu percakapan baru.
2. **Tulis prompt sendiri**, jangan minta orang lain yang nulis. Prompt kamu harus menunjukkan kamu paham apa yang kamu minta.
3. **Setiap prompt disertai alasan teknis** — kenapa kamu pilih pendekatan itu, apa yang kamu bingungkan.
4. **Tempel signature kode asli** kalau minta LLM nyambung ke kode yang sudah ada. LLM nggak bisa nebak nama fungsi/kelas. (Di proyek ini 4 dari 5 error terjadi gara-gara ini.)
5. **Error jangan dihapus atau dipalsukan.** Kalau kode LLM langsung jalan, tulis "tidak ada error". Kalau error, salin traceback utuh.
6. **Ambil share link tiap percakapan:** menu ⋮ di kanan atas → **Bagikan percakapan** → copy link.
7. **Isi log langsung setelah sesi selesai**, jangan nunggu akhir proyek. Kronologi asli yang dinilai.
8. **Commit tiap sesi selesai**, jangan numpuk di akhir.

**Yang dilarang keras:**
- `git commit --author="Nama Orang Lain"` — memalsukan penulis
- `git commit --allow-empty` — commit kosong biar jumlahnya banyak
- Copy-paste blok kode utuh dari GitHub/blog/tutorial
- Nyentuh file yang sudah selesai (`data.py`, `diffusion.py`, `model.py`, `train.py`, `sample.py`)

---

## 5. Sesi 6 — `src/evaluate.py`

**Tujuan:** kasih angka, bukan cuma "kelihatan mirip".

**Yang perlu kamu siapkan sebelum prompt:**
- Baca `src/sample.py` — cari fungsi buat generate sampel
- Baca `src/config.py` — nama field yang ada (`ckpt_dir`, `fig_dir`, `num_sample_images`, dst)
- Pastikan `checkpoints/ddpm_best.pt` ada

**Ide prompt (tulis ulang pakai bahasamu sendiri, jangan copy ini mentah):**

> "saya punya DDPM yang sudah dilatih buat Fashion-MNIST 28x28 grayscale. sekarang saya butuh `src/evaluate.py` buat evaluasi kuantitatif hasil sampling.
>
> yang saya bingung:
> 1. FID katanya standar buat diffusion, tapi FID butuh InceptionV3 yang expect gambar 299x299 RGB. dataset saya 28x28 grayscale. apa FID masih valid kalau saya resize + replikasi ke 3 channel? apa malah angkanya jadi menyesatkan?
> 2. kalau FID nggak cocok, metrik apa yang lebih jujur buat kasus saya? saya kepikiran histogram intensitas piksel (gambar asli vs hasil sampling) — apakah itu cukup ilmiah?
> 3. gimana cara baca hasilnya biar bisa dibilang 'model bagus' atau 'model jelek'?
>
> signature yang sudah ada di repo saya: [TEMPEL SIGNATURE ASLI]

**Output yang diharapkan:** `src/evaluate.py` yang bisa dijalankan `python src/evaluate.py`, menghasilkan minimal satu gambar di `report/figures/` + angka tercetak di konsol.

**Commit:**
```bash
git add src/evaluate.py report/figures/
git commit -m "sesi 6: evaluate.py — histogram intensitas asli vs sampling"
```

---

## 6. Sesi 7 — `src/ablation.py`

**Tujuan:** banding dua konfigurasi, buktikan pilihan di `config.py` masuk akal.

**Pilih satu** (jangan dua-duanya, waktunya mepet):

**Opsi A — banding beta schedule:**
- `beta_schedule="linear"` (1e-4 → 0.02) vs `"cosine"`
- **Cukup 3 epoch**, jangan 25. Tujuannya lihat tren, bukan hasil final.
- Bandingkan kurva loss di epoch 1, 2, 3.

**Opsi B — banding kapasitas model:**
- `unet_base=32` (793.953 param) vs `unet_base=48`
- Bandingkan jumlah parameter + loss setelah 3 epoch.

**Catatan penting:** kalau pilih Opsi A, cek dulu apakah `src/diffusion.py` sudah mendukung `"cosine"`. Kalau belum, itu jadi bahan prompt yang bagus — tanya LLM cara nambahin schedule cosine.

**Ide prompt:**

> "saya mau ablation study buat DDPM saya. config saya punya `beta_schedule: str = 'linear'` tapi setahu saya cuma linear yang diimplementasi. saya mau banding linear vs cosine, tapi cuma 3 epoch per run biar cepat.
>
> yang saya bingung:
> 1. kenapa orang pilih cosine? apa bedanya secara praktis sama linear?
> 2. 3 epoch cukup nggak buat lihat bedanya, atau hasilnya bakal noise semua?
> 3. gimana cara bikin `ablation.py` yang ngejalanin dua konfigurasi berurutan tanpa saling nimpa checkpoint?
>
> signature: [TEMPEL SIGNATURE ASLI]

**Output:** `src/ablation.py` + tabel/grafik perbandingan di `report/figures/`.

**Commit:**
```bash
git add src/ablation.py report/figures/
git commit -m "sesi 7: ablation.py — linear vs cosine, 3 epoch"
```

---

## 7. Sesi 8 — figur tambahan

**Tujuan:** lengkapi gambar buat laporan.

Yang sudah ada (jangan bikin ulang): `loss_curve.png`, `progression_epochs_1_to_25.png`, `samples_epoch_*.png`.

Yang bisa kamu tambah:
- **Grid perbandingan** gambar asli dataset vs hasil sampling, berdampingan — ini yang paling meyakinkan buat pembaca
- **Histogram overlay** intensitas piksel asli vs hasil (kalau sesi 6 belum menghasilkan gambar)
- **Grid ablasi** — dua kurva loss dalam satu plot

**Ide prompt:**

> "saya butuh figur buat laporan: grid yang nunjukin gambar asli Fashion-MNIST di baris atas dan hasil sampling model saya di baris bawah, biar pembaca bisa banding langsung. [lanjutkan dengan kebingungan kamu sendiri]"

**Commit:**
```bash
git add report/figures/
git commit -m "sesi 8: figur perbandingan asli vs sampling untuk laporan"
```

---

## 8. Cara nulis entri log

Setelah tiap sesi, buka `LOG-INTERAKSI.md`, tambah bagian baru di **bawah Sesi #5**, formatnya:

```markdown
## Sesi #6 — <tanggal>

- **Target Modul:** `src/evaluate.py`
- **Operator:** <nama kamu>
- **Tautan Sesi Chat:** https://share.gemini.google/<token>
- **Model LLM:** Gemini

### 1. TAHAP INISIASI (Meminta Kode Awal)

**Prompt Kelompok:**

> "<prompt kamu verbatim, apa adanya — termasuk typo kalau ada>"

**Alasan Khusus:**
<kenapa kamu pilih pendekatan itu, apa yang kamu bingungkan, kenapa 3 epoch bukan 25>

**Output Model:**
<ringkasan jawaban Gemini: fungsi apa yang dibuat, pendekatan apa yang dipilih>

**Hasil Uji:**
<output konsol asli: angka, nama file gambar yang dihasilkan>

### 2. TAHAP PENANGANAN EROR / DEBUGGING

<tulis "Tidak ada error pada sesi ini — kode langsung jalan." kalau memang begitu.
kalau ada error: traceback utuh → prompt perbaikan → analisis → hasil uji>
```

Jangan lupa tambah baris di tabel daftar isi paling atas `LOG-INTERAKSI.md`:

```markdown
| 6 | <tanggal> | `src/evaluate.py` — evaluasi kuantitatif | <nama kamu> | ✅ LULUS |
```

---

## 9. Alur kerja git

Sebelum mulai kerja **selalu** tarik perubahan terbaru:

```bash
git pull
```

Setelah satu sesi selesai:

```bash
git add <file-yang-kamu-ubah>
git commit -m "sesi 6: evaluate.py — histogram intensitas asli vs sampling"
git push
```

Kalau muncul konflik, **jangan** `git checkout .` (itu membuang kerjaan orang). Kirim pesan errornya ke chat.

Kalau `git push` ditolak karena ketinggalan:

```bash
git pull --rebase
git push
```

---

## 10. Checklist sebelum bilang selesai

- [ ] `git config --local user.email` keluar email **kamu**, bukan punya Faqih
- [ ] `src/evaluate.py` jalan tanpa error (`python src/evaluate.py`)
- [ ] `src/ablation.py` jalan tanpa error
- [ ] Minimal 2 gambar baru di `report/figures/`
- [ ] Entri Sesi #6, #7, #8 ada di `LOG-INTERAKSI.md`, lengkap dengan share link
- [ ] Tabel daftar isi di atas `LOG-INTERAKSI.md` sudah ditambah barisnya
- [ ] `git log --author="<email kamu>" --oneline` keluar minimal 3 commit
- [ ] `git push` sudah, semua perubahan ada di GitHub

---

## 11. Kalau pakai Hermes Agent

Kalau kamu pakai Hermes, prompt ini bisa langsung ditempel:

```
Saya anggota kelompok 2 orang untuk proyek CAK4KBB3 (Generative AI).
Repo: https://github.com/pakihaaa01/GenAI.git (Private, saya collaborator)

Tugas saya 3 sesi vibe coding (sesi 6, 7, 8). Baca BRIEF-MIA.md di repo itu
dulu — semua aturan ada di sana.

PENTING: jangan kerjakan sendiri. Saya harus buka Gemini pakai akun saya
sendiri, karena semua percakapan wajib direkam di LOG-INTERAKSI.md.

Mulai dengan: bantu saya clone repo, set identitas git lokal pakai nama+email
saya sendiri, install requirements, lalu pandu saya sesi 6.
```

---

## 12. Batas waktu

Kalau bingung atau mentok, tanya Faqih. Yang **jangan** dilakukan: diam-diam commit kode yang bukan hasil sesi kamu sendiri. Itu lebih merugikan daripada telat.

Kerjaan kamu yang paling berharga bukan kodenya — tapi **log percakapan yang jujur**. 25% nilai ada di situ.
