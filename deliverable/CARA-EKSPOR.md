# CARA EKSPOR — Deliverable & Lampiran

Panduan singkat untuk mengubah repo ini menjadi **satu file `.md`** yang dikumpulkan.

---

## 1. Yang dikumpulkan

Satu file: `LAPORAN-PROYEK1.md` (nama bebas). Isinya:

| Bagian | Sumber |
|---|---|
| Judul, anggota, pembagian tugas | `README.md` bagian Pembagian Tugas |
| Ringkasan arsitektur & hyperparameter | `README.md` + `src/config.py` |
| **Log Interaksi LLM (inti, 25%)** | `LOG-INTERAKSI.md` — tempel utuh |
| Bukti kolaborasi git (20%) | `LAMPIRAN-COMMIT.md` — hasil perintah di bawah |
| Kode sumber lengkap | 6 file `src/*.py` ditempel sebagai lampiran |
| Hasil (grafik loss, grid sampel) | gambar dari `report/figures/` disisipkan |

---

## 2. Ekspor riwayat commit

Jalankan dari root repo:

```bash
git log --pretty=format:"%h | %an | %ad | %s" --date=short > LAMPIRAN-COMMIT.md
```

Cek kontribusi per orang (harus muncul dua nama):

```bash
git shortlog -sne
```

Kalau hanya muncul satu nama, rekan belum punya commit — minta dia commit bagiannya sendiri
(lihat `TEAM-GIT.md`) sebelum ekspor ulang.

---

## 3. Ekspor kode sumber

Opsi A — satu perintah, semua file:

```bash
for f in src/*.py; do
  echo "### \`$f\`" >> LAMPIRAN-KODE.md
  echo '```python' >> LAMPIRAN-KODE.md
  cat "$f" >> LAMPIRAN-KODE.md
  echo '```' >> LAMPIRAN-KODE.md
  echo >> LAMPIRAN-KODE.md
done
```

Opsi B — kalau `cat` dianggap terlalu mentah, pakai ini agar ada nomor baris:

```bash
for f in src/*.py; do
  echo "### \`$f\`" >> LAMPIRAN-KODE.md
  echo '```python' >> LAMPIRAN-KODE.md
  awk '{printf "%4d| %s\n", NR, $0}' "$f" >> LAMPIRAN-KODE.md
  echo '```' >> LAMPIRAN-KODE.md
  echo >> LAMPIRAN-KODE.md
done
```

---

## 4. Ekspor share link Gemini

Setiap sesi di `LOG-INTERAKSI.md` sudah punya link `share.gemini.google/...`.
**Buka tiap link di jendela incognito** sebelum submit — kalau terbuka, berarti link masih publik
dan penilai bisa membacanya. Kalau minta login, link-nya belum di-share.

Cara membuat share link (kalau ada sesi yang belum):
1. Buka percakapan di Gemini
2. Tombol menu (⋮) di kanan atas → **Bagikan percakapan**
3. Salin link → tempel ke entri sesi di `LOG-INTERAKSI.md`

---

## 5. Sisipkan gambar

Dari `report/figures/`:

- `progression_epochs_*.png` — grid progres antar-epoch (paling penting, bukti model belajar)
- `loss_curve.png` — kurva loss (dibuat di notebook Colab sel 9)
- `samples_epoch_*.png` — grid per epoch

Tempel dengan sintaks markdown biasa:

```markdown
![Progres sampling antar-epoch](report/figures/progression_epochs_1_to_25.png)
```

> Kalau deliverable harus **satu file berdiri sendiri** (tanpa folder gambar), ubah gambar jadi
> base64 data URI, atau serahkan `.md` + folder `figures/` dalam satu ZIP.

---

## 6. Urutan perakitan final

1. Jalankan training penuh (Colab GPU T4, 40 epoch ≈ 3–5 menit; CPU 25 epoch ≈ 33 menit).
2. Jalankan `python src/sample.py` → grid progres.
3. Jalankan sel 9 notebook → `loss_curve.png`.
4. Ekspor `LAMPIRAN-COMMIT.md` dan `LAMPIRAN-KODE.md`.
5. Susun `LAPORAN-PROYEK1.md`:
   - Bab 1: latar belakang + arsitektur (dari README)
   - Bab 2: **Log Interaksi utuh** (tempel `LOG-INTERAKSI.md`)
   - Bab 3: hasil & pembahasan (gambar + angka loss)
   - Lampiran A: kode sumber
   - Lampiran B: riwayat commit
6. Cek `git shortlog -sne` — dua nama harus muncul.
7. Buka semua share link di incognito — harus bisa dibaca.
8. Submit.
