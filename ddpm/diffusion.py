"""Gaussian diffusion from "Denoising Diffusion Probabilistic Models" (Ho et al., 2020).

https://arxiv.org/abs/2006.11239

Notation follows the paper:
    beta_t        noise variance added at step t            (Sec. 2, Eq. 2)
    alpha_t       1 - beta_t                                (Sec. 2, above Eq. 4)
    alpha_bar_t   prod_{s=1..t} alpha_s                     (Sec. 2, above Eq. 4)

Timesteps are 1-indexed in the paper (t = 1..T). In code they are 0-indexed
(t = 0..T-1), so code index t corresponds to paper step t + 1.
"""

import torch
import torch.nn.functional as F


def linear_beta_schedule(T, beta_1=1e-4, beta_T=0.02):
    """Linearly spaced betas from beta_1 to beta_T (Sec. 4).

    The paper uses T=1000, beta_1=1e-4, beta_T=0.02. For a smaller T we scale
    both endpoints by 1000/T so that alpha_bar_T still ends close to 0, i.e. x_T
    is still (almost) pure noise. This is the same rescaling used by Nichol &
    Dhariwal (2021) when changing T.
    """
    scale = 1000 / T
    assert beta_T * scale < 1, f"T={T} is too small for this rescaling (beta_T would be >= 1)"
    return torch.linspace(beta_1 * scale, beta_T * scale, T, dtype=torch.float64)


def extract(values, t, x_shape):
    """Pick values[t] for each batch element and reshape to broadcast over an image batch."""
    out = values.gather(0, t)
    return out.reshape(t.shape[0], *([1] * (len(x_shape) - 1)))


class GaussianDiffusion:
    """Forward process q, training loss (Algorithm 1) and ancestral sampler (Algorithm 2)."""

    def __init__(self, T=300, device="cpu"):
        self.T = T
        betas = linear_beta_schedule(T)
        alphas = 1.0 - betas
        alpha_bars = torch.cumprod(alphas, dim=0)

        # Precompute in float64 for accuracy, store as float32 for use with the model.
        def to(x):
            return x.to(device=device, dtype=torch.float32)

        self.betas = to(betas)
        self.alphas = to(alphas)
        self.alpha_bars = to(alpha_bars)
        self.sqrt_alpha_bars = to(alpha_bars.sqrt())
        self.sqrt_one_minus_alpha_bars = to((1.0 - alpha_bars).sqrt())
        self.sqrt_recip_alphas = to((1.0 / alphas).sqrt())
        # Reverse-process variance. The paper reports sigma_t^2 = beta_t and
        # sigma_t^2 = beta_tilde_t giving similar results (Sec. 3.2); we use beta_t.
        self.sigmas = to(betas.sqrt())

    def q_sample(self, x0, t, noise=None):
        """Sample x_t ~ q(x_t | x_0) in closed form (Eq. 4).

        q(x_t | x_0) = N(x_t; sqrt(alpha_bar_t) x_0, (1 - alpha_bar_t) I), so
        x_t = sqrt(alpha_bar_t) * x_0 + sqrt(1 - alpha_bar_t) * eps,  eps ~ N(0, I).
        """
        if noise is None:
            noise = torch.randn_like(x0)
        return (
            extract(self.sqrt_alpha_bars, t, x0.shape) * x0
            + extract(self.sqrt_one_minus_alpha_bars, t, x0.shape) * noise
        )

    def q_step(self, x_prev, t, noise=None):
        """Sample one forward step x_t ~ q(x_t | x_{t-1}) = N(sqrt(1 - beta_t) x_{t-1}, beta_t I) (Eq. 2).

        Not needed for training (q_sample jumps straight to step t); used in tests
        to check that chaining single steps matches the closed form.
        """
        if noise is None:
            noise = torch.randn_like(x_prev)
        beta = extract(self.betas, t, x_prev.shape)
        return (1.0 - beta).sqrt() * x_prev + beta.sqrt() * noise

    def loss(self, model, x0):
        """Simplified training objective L_simple (Eq. 14, Algorithm 1).

        1. t ~ Uniform({1, ..., T})
        2. eps ~ N(0, I)
        3. loss = || eps - eps_theta(sqrt(alpha_bar_t) x_0 + sqrt(1 - alpha_bar_t) eps, t) ||^2
        """
        t = torch.randint(0, self.T, (x0.shape[0],), device=x0.device)
        noise = torch.randn_like(x0)
        x_t = self.q_sample(x0, t, noise)
        return F.mse_loss(model(x_t, t), noise)

    @torch.no_grad()
    def p_sample(self, model, x_t, t_index):
        """One reverse step x_{t-1} ~ p_theta(x_{t-1} | x_t) (Algorithm 2, line 4).

        x_{t-1} = 1/sqrt(alpha_t) * (x_t - beta_t / sqrt(1 - alpha_bar_t) * eps_theta(x_t, t)) + sigma_t z
        with z ~ N(0, I) for t > 1 and z = 0 at the final step.
        """
        t = torch.full((x_t.shape[0],), t_index, device=x_t.device, dtype=torch.long)
        eps = model(x_t, t)
        mean = extract(self.sqrt_recip_alphas, t, x_t.shape) * (
            x_t - extract(self.betas, t, x_t.shape) / extract(self.sqrt_one_minus_alpha_bars, t, x_t.shape) * eps
        )
        if t_index == 0:
            return mean
        return mean + extract(self.sigmas, t, x_t.shape) * torch.randn_like(x_t)

    @torch.no_grad()
    def sample(self, model, shape, device="cpu", keep_every=None):
        """Ancestral sampling, Algorithm 2: start at x_T ~ N(0, I) and apply p_sample T times.

        If keep_every is set, also returns the intermediate x_t every `keep_every`
        steps (plus the final x_0) for visualising the denoising trajectory.
        """
        x = torch.randn(shape, device=device)
        trajectory = [x]
        for t_index in reversed(range(self.T)):
            x = self.p_sample(model, x, t_index)
            if keep_every is not None and (t_index % keep_every == 0):
                trajectory.append(x)
        if keep_every is None:
            return x
        return x, trajectory
