import os
import json
import time
import sys
from pathlib import Path

import torch
from tqdm import tqdm

# Pastikan folder root proyek (parent dari src/) selalu ada di sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.config import Config
from src.data import get_dataloader, check_batch
from src.model import DDPMUNet
from src.diffusion import DDPM


def set_seed(seed: int = 42):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def train(cfg: Config):
    set_seed(cfg.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    os.makedirs(cfg.ckpt_dir, exist_ok=True)
    os.makedirs(cfg.log_dir, exist_ok=True)

    # 1. Data
    train_loader = get_dataloader(cfg, train=True)
    check_batch(
        train_loader,
        expected_batches=len(train_loader),
        expected_batch_size=cfg.batch_size,
    )

    # 2. Model & Diffusion (keduanya WAJIB .to(device))
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

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=cfg.lr,
        weight_decay=getattr(cfg, "weight_decay", 1e-4),
    )

    use_amp = device.type == "cuda"
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp)

    history = []
    best_loss = float("inf")

    # 3. Training Loop
    for epoch in range(1, cfg.epochs + 1):
        model.train()
        epoch_loss = 0.0
        start_time = time.time()

        pbar = tqdm(train_loader, desc=f"Epoch {epoch:03d}/{cfg.epochs:03d}", leave=False)
        for x0, _ in pbar:
            x0 = x0.to(device, non_blocking=True)
            t = torch.randint(0, cfg.timesteps, (x0.size(0),), device=device).long()

            optimizer.zero_grad(set_to_none=True)

            with torch.amp.autocast("cuda", enabled=use_amp):
                loss = diffusion.p_losses(model, x0, t)

            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            scaler.step(optimizer)
            scaler.update()

            epoch_loss += loss.item()
            pbar.set_postfix(loss=f"{loss.item():.4f}")

        avg_loss = epoch_loss / len(train_loader)
        elapsed = time.time() - start_time
        history.append({"epoch": epoch, "loss": avg_loss, "time_sec": round(elapsed, 2)})

        print(f"[Epoch {epoch:03d}/{cfg.epochs:03d}] Loss: {avg_loss:.5f} | Time: {elapsed:.1f}s")

        # 4. Save Checkpoint per Epoch
        ckpt_payload = {
            "epoch": epoch,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "loss": avg_loss,
            "config": vars(cfg) if hasattr(cfg, "__dict__") else {},
        }

        epoch_ckpt_path = os.path.join(cfg.ckpt_dir, f"ddpm_epoch_{epoch:03d}.pt")
        torch.save(ckpt_payload, epoch_ckpt_path)

        if avg_loss < best_loss:
            best_loss = avg_loss
            torch.save(ckpt_payload, os.path.join(cfg.ckpt_dir, "ddpm_best.pt"))

        with open(os.path.join(cfg.log_dir, "train_history.json"), "w") as f:
            json.dump(history, f, indent=2)


if __name__ == "__main__":
    cfg = Config()
    train(cfg)
