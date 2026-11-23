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
