"""Konfigurasi terpusat Mini DDPM — Fashion-MNIST.

Semua hyperparameter ada di sini supaya eksperimen (Fase 4) cukup mengubah satu file
dan hasilnya bisa dibandingkan secara adil. Nilai default = hasil benchmark CPU
(Ryzen 7 8845HS, 16 thread): batch 128, ±1,3 menit/epoch pada subset 15k.
"""

from dataclasses import dataclass, asdict, field
from pathlib import Path

# Root proyek = satu level di atas src/. Dipakai agar folder output selalu
# dibuat di lokasi yang sama, tak peduli dari direktori mana skrip dijalankan.
PROJECT_ROOT = Path(__file__).resolve().parent.parent


@dataclass
class Config:
    # ---- Data ----
    dataset: str = "FashionMNIST"
    data_root: str = "data"
    subset_size: int = 15_000          # dari 60.000 gambar train
    image_size: int = 28
    channels: int = 1
    num_workers: int = 0               # Windows: 0 paling aman

    # ---- Model ----
    unet_base: int = 32                # channel pada resolusi penuh
    time_embed_dim: int = 128
    num_groups: int = 8                # GroupNorm

    # ---- Diffusion ----
    timesteps: int = 1000              # T
    beta_start: float = 1e-4
    beta_end: float = 0.02
    beta_schedule: str = "linear"

    # ---- Training ----
    batch_size: int = 128
    epochs: int = 25
    lr: float = 2e-4
    seed: int = 42
    sample_every: int = 5              # simpan grid gambar tiap N epoch
    num_sample_images: int = 64

    # ---- Path ----
    ckpt_dir: str = "checkpoints"
    log_dir: str = "logs"
    fig_dir: str = "report/figures"

    def __post_init__(self):
        for d in (self.data_root, self.ckpt_dir, self.log_dir, self.fig_dir):
            (PROJECT_ROOT / d).mkdir(parents=True, exist_ok=True)

    def to_dict(self) -> dict:
        return asdict(self)

    def run_name(self) -> str:
        """Nama unik per eksperimen — dipakai untuk nama file log & checkpoint."""
        return (
            f"bs{self.batch_size}_lr{self.lr:g}_ep{self.epochs}"
            f"_base{self.unet_base}_T{self.timesteps}"
        )


if __name__ == "__main__":
    cfg = Config()
    print("Konfigurasi aktif:")
    for k, v in cfg.to_dict().items():
        print(f"  {k:18s} = {v}")
    print(f"\nrun_name = {cfg.run_name()}")
