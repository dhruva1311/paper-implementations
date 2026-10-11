"""Tests for the diffusion process and model. Run with `python -m unittest -v` (or pytest)."""

import unittest

import torch

from diffusion import GaussianDiffusion, linear_beta_schedule
from model import SinusoidalTimeEmbedding, UNet

torch.set_num_threads(2)


class TestSchedule(unittest.TestCase):
    def test_linear_betas(self):
        betas = linear_beta_schedule(1000)
        self.assertEqual(betas.shape, (1000,))
        self.assertAlmostEqual(betas[0].item(), 1e-4)
        self.assertAlmostEqual(betas[-1].item(), 0.02)
        self.assertTrue(torch.all(betas[1:] > betas[:-1]))

    def test_alpha_bar_ends_near_zero(self):
        """x_T should be almost pure noise, also for the reduced T used in training."""
        for T in (1000, 300):
            d = GaussianDiffusion(T)
            self.assertTrue(torch.all(d.alpha_bars[1:] < d.alpha_bars[:-1]))
            self.assertLess(d.alpha_bars[-1].item(), 1e-3)


class TestForwardProcess(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(0)
        self.d = GaussianDiffusion(T=300)

    def test_q_sample_shape(self):
        x0 = torch.randn(8, 1, 28, 28)
        t = torch.randint(0, 300, (8,))
        self.assertEqual(self.d.q_sample(x0, t).shape, x0.shape)

    def test_q_sample_with_zero_noise_is_scaled_x0(self):
        x0 = torch.randn(4, 1, 28, 28)
        t = torch.tensor([0, 10, 100, 299])
        out = self.d.q_sample(x0, t, noise=torch.zeros_like(x0))
        expected = self.d.sqrt_alpha_bars[t].view(-1, 1, 1, 1) * x0
        torch.testing.assert_close(out, expected)

    def test_q_sample_statistics_match_closed_form(self):
        """Empirical mean/variance of x_t ~ q(x_t|x_0) match sqrt(alpha_bar_t) x_0 and 1 - alpha_bar_t (Eq. 4)."""
        n = 20_000
        x0_value = 0.7
        for t_index in (0, 50, 150, 299):
            x0 = torch.full((n, 1), x0_value)
            t = torch.full((n,), t_index)
            xt = self.d.q_sample(x0, t)
            alpha_bar = self.d.alpha_bars[t_index].item()
            self.assertAlmostEqual(xt.mean().item(), alpha_bar ** 0.5 * x0_value, delta=0.02)
            self.assertAlmostEqual(xt.var().item(), 1.0 - alpha_bar, delta=0.03)

    def test_chained_steps_match_closed_form(self):
        """Applying q(x_t|x_{t-1}) (Eq. 2) step by step gives the same marginal as the closed form (Eq. 4)."""
        n, t_target, x0_value = 20_000, 120, -0.5
        x = torch.full((n, 1), x0_value)
        for t_index in range(t_target + 1):
            x = self.d.q_step(x, torch.full((n,), t_index))
        alpha_bar = self.d.alpha_bars[t_target].item()
        self.assertAlmostEqual(x.mean().item(), alpha_bar ** 0.5 * x0_value, delta=0.02)
        self.assertAlmostEqual(x.var().item(), 1.0 - alpha_bar, delta=0.03)


class TestModelAndSampling(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(0)
        self.model = UNet(base_ch=16)

    def test_time_embedding_shape_and_range(self):
        emb = SinusoidalTimeEmbedding(32)(torch.tensor([0, 1, 299]))
        self.assertEqual(emb.shape, (3, 32))
        self.assertLessEqual(emb.abs().max().item(), 1.0)
        self.assertFalse(torch.allclose(emb[0], emb[1]))

    def test_unet_output_shape(self):
        x = torch.randn(3, 1, 28, 28)
        t = torch.tensor([0, 5, 299])
        self.assertEqual(self.model(x, t).shape, x.shape)

    def test_loss_is_scalar_and_differentiable(self):
        d = GaussianDiffusion(T=50)
        loss = d.loss(self.model, torch.randn(4, 1, 28, 28))
        self.assertEqual(loss.dim(), 0)
        loss.backward()
        self.assertIsNotNone(self.model.stem.weight.grad)

    def test_sample_shape_and_trajectory(self):
        d = GaussianDiffusion(T=50)
        x, traj = d.sample(self.model, (2, 1, 28, 28), keep_every=25)
        self.assertEqual(x.shape, (2, 1, 28, 28))
        self.assertTrue(torch.isfinite(x).all())
        self.assertEqual(len(traj), 3)  # x_T, then t_index 25 and 0
        torch.testing.assert_close(traj[-1], x)


if __name__ == "__main__":
    unittest.main()
