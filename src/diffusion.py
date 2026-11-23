"""Denoising Diffusion Probabilistic Model (DDPM) from scratch.

Diimplementasikan mengikuti Ho et al. (2020), tanpa library diffusion siap pakai.
Kode awal dihasilkan lewat sesi vibe coding dengan Gemini lalu diverifikasi.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class DDPM(nn.Module):
    """
    Implementasi Denoising Diffusion Probabilistic Models (DDPM) dari nol
    berdasarkan Ho et al. (2020).
    """

    def __init__(
        self,
        timesteps: int = 1000,
        beta_start: float = 1e-4,
        beta_end: float = 0.02,
        clip_denoised: bool = True,
    ) -> None:
        super().__init__()
        self.timesteps = timesteps
        self.clip_denoised = clip_denoised

        # 1. Linear noise schedule: beta_1 sampai beta_T
        betas = torch.linspace(beta_start, beta_end, timesteps, dtype=torch.float32)

        # 2. Definisi alpha_t dan cumulative product alpha_bar_t
        alphas = 1.0 - betas
        alphas_cumprod = torch.cumprod(alphas, dim=0)

        # alpha_bar_{t-1}, dengan alpha_bar_0 didefinisikan = 1.0
        alphas_cumprod_prev = F.pad(alphas_cumprod[:-1], (1, 0), value=1.0)

        # Buffer: ikut state_dict tapi tidak dilatih
        self.register_buffer("betas", betas)
        self.register_buffer("alphas", alphas)
        self.register_buffer("alphas_cumprod", alphas_cumprod)
        self.register_buffer("alphas_cumprod_prev", alphas_cumprod_prev)

        # Koefisien untuk forward process q(x_t | x_0)
        self.register_buffer("sqrt_alphas_cumprod", torch.sqrt(alphas_cumprod))
        self.register_buffer("sqrt_one_minus_alphas_cumprod", torch.sqrt(1.0 - alphas_cumprod))

        # Koefisien untuk merekonstruksi x_0 dari prediksi noise
        self.register_buffer("sqrt_recip_alphas_cumprod", torch.sqrt(1.0 / alphas_cumprod))
        self.register_buffer("sqrt_recipm1_alphas_cumprod", torch.sqrt(1.0 / alphas_cumprod - 1.0))

        # Koefisien untuk distribusi posterior q(x_{t-1} | x_t, x_0)
        posterior_variance = betas * (1.0 - alphas_cumprod_prev) / (1.0 - alphas_cumprod)
        self.register_buffer("posterior_variance", posterior_variance)

        # Log variance di-clip pada t=0 karena posterior_variance[0] bernilai 0
        self.register_buffer(
            "posterior_log_variance_clipped",
            torch.log(torch.clamp(posterior_variance, min=1e-20)),
        )

        # Koefisien rata-rata posterior: mu_tilde = coef1 * x_0 + coef2 * x_t
        self.register_buffer(
            "posterior_mean_coef1",
            betas * torch.sqrt(alphas_cumprod_prev) / (1.0 - alphas_cumprod),
        )
        self.register_buffer(
            "posterior_mean_coef2",
            (1.0 - alphas_cumprod_prev) * torch.sqrt(alphas) / (1.0 - alphas_cumprod),
        )

    def _extract(self, a: torch.Tensor, t: torch.Tensor, x_shape: torch.Size) -> torch.Tensor:
        """
        Mengambil elemen ke-t dari tensor 1D `a` untuk setiap sampel dalam batch,
        lalu reshape ke (B, 1, 1, 1) agar bisa di-broadcast ke (B, C, H, W).
        """
        batch_size = t.shape[0]
        out = a.gather(0, t)
        return out.reshape(batch_size, *((1,) * (len(x_shape) - 1)))

    def q_sample(
        self,
        x0: torch.Tensor,
        t: torch.Tensor,
        noise: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """
        Forward diffusion: lompat langsung dari x_0 ke x_t.
        x_t = sqrt(alpha_bar_t) * x_0 + sqrt(1 - alpha_bar_t) * eps
        """
        if noise is None:
            noise = torch.randn_like(x0)

        sqrt_alpha_bar_t = self._extract(self.sqrt_alphas_cumprod, t, x0.shape)
        sqrt_one_minus_alpha_bar_t = self._extract(self.sqrt_one_minus_alphas_cumprod, t, x0.shape)

        return sqrt_alpha_bar_t * x0 + sqrt_one_minus_alpha_bar_t * noise

    def p_losses(
        self,
        model: nn.Module,
        x0: torch.Tensor,
        t: torch.Tensor,
        noise: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """
        Training loss (L_simple): MSE antara noise asli dan noise prediksi model.
        """
        if noise is None:
            noise = torch.randn_like(x0)

        x_t = self.q_sample(x0=x0, t=t, noise=noise)
        predicted_noise = model(x_t, t)
        loss = F.mse_loss(predicted_noise, noise)
        return loss

    def predict_start_from_noise(
        self,
        x_t: torch.Tensor,
        t: torch.Tensor,
        noise: torch.Tensor,
    ) -> torch.Tensor:
        """
        Estimasi x_0 dari x_t dan prediksi noise eps_theta.
        x_0 = (1 / sqrt(alpha_bar_t)) * x_t - sqrt(1 / alpha_bar_t - 1) * eps
        """
        coef1 = self._extract(self.sqrt_recip_alphas_cumprod, t, x_t.shape)
        coef2 = self._extract(self.sqrt_recipm1_alphas_cumprod, t, x_t.shape)
        return coef1 * x_t - coef2 * noise

    def q_posterior_mean_variance(
        self,
        x0: torch.Tensor,
        x_t: torch.Tensor,
        t: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Mean dan variance dari distribusi posterior q(x_{t-1} | x_t, x_0)."""
        posterior_mean = (
            self._extract(self.posterior_mean_coef1, t, x_t.shape) * x0
            + self._extract(self.posterior_mean_coef2, t, x_t.shape) * x_t
        )
        posterior_var = self._extract(self.posterior_variance, t, x_t.shape)
        posterior_log_var_clipped = self._extract(
            self.posterior_log_variance_clipped, t, x_t.shape
        )
        return posterior_mean, posterior_var, posterior_log_var_clipped

    def p_mean_variance(
        self,
        model: nn.Module,
        x_t: torch.Tensor,
        t: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Parameter distribusi reverse p_theta(x_{t-1} | x_t)."""
        pred_noise = model(x_t, t)

        x0_recon = self.predict_start_from_noise(x_t=x_t, t=t, noise=pred_noise)
        if self.clip_denoised:
            x0_recon = torch.clamp(x0_recon, -1.0, 1.0)

        model_mean, posterior_var, posterior_log_var = self.q_posterior_mean_variance(
            x0=x0_recon, x_t=x_t, t=t
        )
        return model_mean, posterior_var, posterior_log_var

    @torch.no_grad()
    def p_sample(
        self,
        model: nn.Module,
        x_t: torch.Tensor,
        t: torch.Tensor,
        t_index: int,
    ) -> torch.Tensor:
        """
        Satu langkah reverse diffusion: sampling x_{t-1} dari x_t.
        x_{t-1} = mu_theta(x_t, t) + sigma_t * z, z ~ N(0, I) untuk t > 0.
        """
        model_mean, _, model_log_var = self.p_mean_variance(model=model, x_t=x_t, t=t)

        if t_index == 0:
            return model_mean

        noise = torch.randn_like(x_t)
        sigma_t = torch.exp(0.5 * model_log_var)
        return model_mean + sigma_t * noise

    @torch.no_grad()
    def p_sample_loop(
        self,
        model: nn.Module,
        shape: tuple[int, int, int, int],
        device: torch.device | str = "cpu",
    ) -> torch.Tensor:
        """
        Generasi penuh: mulai dari x_T ~ N(0, I) lalu iterasi mundur sampai x_0.
        """
        model.eval()
        batch_size = shape[0]

        img = torch.randn(shape, device=device)

        for i in reversed(range(self.timesteps)):
            t = torch.full((batch_size,), i, device=device, dtype=torch.long)
            img = self.p_sample(model=model, x_t=img, t=t, t_index=i)

        return torch.clamp(img, -1.0, 1.0)


if __name__ == "__main__":
    class DummyUNet(nn.Module):
        def forward(self, x: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
            return torch.zeros_like(x)

    ddpm = DDPM(timesteps=1000, beta_start=1e-4, beta_end=0.02)
    dummy_model = DummyUNet()

    x0 = torch.randn(8, 1, 28, 28).clamp(-1.0, 1.0)
    t = torch.randint(0, 1000, (8,), dtype=torch.long)

    xt = ddpm.q_sample(x0, t)
    loss = ddpm.p_losses(dummy_model, x0, t)
    print(f"Shape x_t      : {tuple(xt.shape)}")
    print(f"Nilai MSE Loss : {loss.item():.4f}")

    xt_start = ddpm.q_sample(x0, torch.zeros(8, dtype=torch.long))
    xt_end = ddpm.q_sample(x0, torch.full((8,), 999, dtype=torch.long))
    print(f"Std di t=0     : {xt_start.std().item():.4f} (x0: {x0.std().item():.4f})")
    print(f"Std di t=999   : {xt_end.std().item():.4f} (mendekati 1.0000)")

    ddpm_fast = DDPM(timesteps=10)
    samples = ddpm_fast.p_sample_loop(dummy_model, shape=(4, 1, 28, 28))
    print(f"Shape sampel   : {tuple(samples.shape)}, Rentang: [{samples.min():.2f}, {samples.max():.2f}]")
