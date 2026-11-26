"""Verifikasi determinisme sampling terhadap ukuran batch (Sesi #9, susulan).

Pertanyaan yang dijawab: apakah 6 sampel pada baris linear
`report/figures/ablation_schedule_grid.png` sama persis dengan 6 sampel
pertama bila sampling dijalankan dengan batch N=128?

Hasil yang diperoleh: TIDAK sama. Noise awal (x_T) identik, tetapi forward
pass U-Net dengan batch berbeda menghasilkan selisih numerik sekitar 1e-6,
dan selisih itu teramplifikasi oleh 1000 langkah reverse diffusion sampai
gambar akhirnya berbeda secara visual. Grid-nya sendiri reproducible
bit-exact (sha256 PNG sama).

Skrip ini menulis ringkasan ke `report/verify_seed_n128.json` dan tensor ke
`report/verify_seed_batches.pt` (tidak di-commit).
"""

from pathlib import Path
import hashlib
import json
import sys
import time

# Bootstrap sys.path agar import `from src...` berjalan saat skrip dijalankan langsung
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import matplotlib

matplotlib.use("Agg")

import numpy as np
import torch
from PIL import Image

from src.figures_ablation import (
    COSINE_CKPT_DIR,
    COSINE_HISTORY_PATH,
    FIXED_SEED,
    LINEAR_CKPT_DIR,
    OUTPUT_PATH,
    _find_checkpoint_file,
    _sample_cosine_checkpoint,
    _sample_linear_checkpoint,
    build_ablation_grid,
)

OUT_JSON = PROJECT_ROOT / "report" / "verify_seed_n128.json"
TMP_PNG = PROJECT_ROOT / "report" / "figures" / "_grid_repro_tmp.png"
BATCH_N = 128


def main() -> None:
    res: dict = {}
    device = torch.device("cpu")

    lin_ckpt = _find_checkpoint_file(LINEAR_CKPT_DIR)
    cos_ckpt = _find_checkpoint_file(COSINE_CKPT_DIR)
    print("checkpoint linear:", lin_ckpt, flush=True)
    print("checkpoint cosine:", cos_ckpt, flush=True)

    t0 = time.perf_counter()
    s6 = _sample_linear_checkpoint(lin_ckpt, device, 6, FIXED_SEED)
    t_n6 = time.perf_counter() - t0
    print(f"N=6 linear selesai: {t_n6:.2f} s", flush=True)

    t0 = time.perf_counter()
    s128 = _sample_linear_checkpoint(lin_ckpt, device, BATCH_N, FIXED_SEED)
    t_n128 = time.perf_counter() - t0
    print(f"N={BATCH_N} linear selesai: {t_n128:.2f} s", flush=True)

    first6 = s128[:6]
    diff = (s6 - first6).abs()
    a8 = (s6.clamp(0, 1) * 255).round().to(torch.uint8)
    b8 = (first6.clamp(0, 1) * 255).round().to(torch.uint8)

    res["linear_ckpt"] = str(lin_ckpt)
    res["seed"] = FIXED_SEED
    res["batch_compared"] = BATCH_N
    res["time_n6_s"] = round(t_n6, 2)
    res["time_n128_s"] = round(t_n128, 2)
    res["float_equal"] = bool(torch.equal(s6, first6))
    res["max_abs_diff"] = float(diff.max())
    res["mean_abs_diff"] = float(diff.mean())
    res["uint8_equal"] = bool(torch.equal(a8, b8))
    res["n_pixels_8bit_diff"] = int((a8 != b8).sum().item())
    res["n_pixels_total"] = int(a8.numel())
    print(json.dumps(res, indent=2), flush=True)

    t0 = time.perf_counter()
    s6cos = _sample_cosine_checkpoint(cos_ckpt, COSINE_HISTORY_PATH, device, 6, FIXED_SEED)
    res["time_n6_cosine_s"] = round(time.perf_counter() - t0, 2)
    print(f"N=6 cosine selesai: {res['time_n6_cosine_s']:.2f} s", flush=True)

    build_ablation_grid(s6, s6cos, TMP_PNG)
    old = np.asarray(Image.open(OUTPUT_PATH).convert("RGB")).astype(np.int16)
    new = np.asarray(Image.open(TMP_PNG).convert("RGB")).astype(np.int16)
    res["grid_shape_old"] = list(old.shape)
    res["grid_shape_new"] = list(new.shape)
    if old.shape == new.shape:
        gd = np.abs(old - new)
        res["grid_identical"] = bool(gd.max() == 0)
        res["grid_max_channel_diff"] = int(gd.max())
        res["grid_pixels_diff"] = int((gd.sum(axis=2) > 0).sum())
    else:
        res["grid_identical"] = False
        res["grid_shape_mismatch"] = True
    res["sha256_existing_png"] = hashlib.sha256(OUTPUT_PATH.read_bytes()).hexdigest()
    res["sha256_reproduced_png"] = hashlib.sha256(TMP_PNG.read_bytes()).hexdigest()
    print(json.dumps(res, indent=2), flush=True)

    torch.save(
        {"s6": s6, "s128_first6": first6, "s6_cosine": s6cos},
        PROJECT_ROOT / "report" / "verify_seed_batches.pt",
    )
    OUT_JSON.write_text(json.dumps(res, indent=2), encoding="utf-8")
    TMP_PNG.unlink(missing_ok=True)
    print("DONE", flush=True)


if __name__ == "__main__":
    main()
