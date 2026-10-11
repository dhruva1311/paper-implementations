"""Generate figures from a trained checkpoint.

Writes to assets/:
    samples.png             grid of samples from Algorithm 2 (EMA weights)
    forward_process.png     real digits noised with q(x_t | x_0) at increasing t
    denoising_trajectory.png  x_T -> x_0 for a few samples during reverse diffusion

Usage:
    python sample.py [--n 64] [--seed 0]
"""

import argparse
import os

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import torch
from torchvision.utils import make_grid

from diffusion import GaussianDiffusion
from model import UNet
from train import HERE, load_mnist


def to_image(x):
    """Map model space [-1, 1] back to [0, 1] pixels."""
    return ((x.clamp(-1, 1) + 1) / 2).cpu()


def save_grid(x, path, nrow, title=None):
    grid = make_grid(to_image(x), nrow=nrow, padding=2, pad_value=1.0)
    fig, ax = plt.subplots(figsize=(nrow * 0.6, grid.shape[1] / grid.shape[2] * nrow * 0.6 + (0.4 if title else 0)))
    ax.imshow(grid[0], cmap="gray", vmin=0, vmax=1)
    ax.axis("off")
    if title:
        ax.set_title(title, fontsize=10)
    fig.tight_layout()
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def save_strip(rows, labels, path):
    """rows: list over time of (n, 1, 28, 28) tensors; plotted as n rows x len(rows) columns."""
    n, cols = rows[0].shape[0], len(rows)
    fig, axes = plt.subplots(n, cols, figsize=(cols * 0.9, n * 0.9 + 0.3))
    axes = axes.reshape(n, cols)
    for j, (x, label) in enumerate(zip(rows, labels)):
        for i in range(n):
            axes[i, j].imshow(to_image(x[i, 0]), cmap="gray", vmin=0, vmax=1)
            axes[i, j].axis("off")
        axes[0, j].set_title(label, fontsize=8)
    fig.tight_layout()
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", default=os.path.join(HERE, "checkpoints", "ddpm_mnist.pt"))
    parser.add_argument("--n", type=int, default=64)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--threads", type=int, default=2)
    args = parser.parse_args()

    torch.set_num_threads(args.threads)
    torch.manual_seed(args.seed)
    assets = os.path.join(HERE, "assets")
    os.makedirs(assets, exist_ok=True)

    ckpt = torch.load(args.checkpoint, map_location="cpu")
    cfg = ckpt["config"]
    diffusion = GaussianDiffusion(T=cfg["T"])
    model = UNet(base_ch=cfg["base_ch"])
    model.load_state_dict(ckpt["ema_model"])
    model.eval()
    print(f"loaded checkpoint: step {ckpt['step']}, T={cfg['T']}")

    # Forward process: real digits at evenly spaced t (code index t -> paper step t + 1).
    x0 = load_mnist(os.path.join(HERE, "data"))[:6]
    ts = [0, 25, 50, 100, 150, 200, 250, cfg["T"] - 1]
    noised = [diffusion.q_sample(x0, torch.full((x0.shape[0],), t)) for t in ts]
    save_strip([x0] + noised, ["x_0"] + [f"t={t + 1}" for t in ts], os.path.join(assets, "forward_process.png"))

    # Reverse process: sample a grid, keeping the trajectory of the first few samples.
    keep_every = cfg["T"] // 6
    samples, traj = diffusion.sample(model, (args.n, 1, 28, 28), keep_every=keep_every)
    save_grid(samples, os.path.join(assets, "samples.png"), nrow=8)

    # After the reverse step at code index t we hold paper's x_t; the last one is x_0.
    labels = [f"t={cfg['T']}"] + [f"t={t}" if t > 0 else "x_0" for t in reversed(range(0, cfg["T"], keep_every))]
    save_strip([x[:6] for x in traj], labels, os.path.join(assets, "denoising_trajectory.png"))
    print(f"saved figures to {assets}")


if __name__ == "__main__":
    main()
