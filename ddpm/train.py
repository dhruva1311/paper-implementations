"""Train a small DDPM on MNIST with the simplified loss (Algorithm 1), CPU only.

Usage:
    python train.py --base-ch 32 --threads 7 --minutes 50   # settings used for the README results
    python train.py --minutes 5     # shorter run
"""

import argparse
import copy
import json
import os
import time

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import torch
from torchvision import datasets

from diffusion import GaussianDiffusion
from model import UNet

HERE = os.path.dirname(os.path.abspath(__file__))


def load_mnist(root):
    """Return all 60k MNIST training images as one float tensor in [-1, 1], shape (N, 1, 28, 28).

    The paper scales pixel values to [-1, 1] (Sec. 3.3).
    """
    ds = datasets.MNIST(root, train=True, download=True)
    x = ds.data.float().div(255.0).unsqueeze(1)
    return x * 2.0 - 1.0


@torch.no_grad()
def ema_update(ema_model, model, decay):
    """Exponential moving average of the weights; the paper samples from an EMA model (decay 0.9999)."""
    for ema_p, p in zip(ema_model.parameters(), model.parameters()):
        ema_p.lerp_(p, 1.0 - decay)


def plot_loss(losses, path, window=100):
    steps = torch.arange(1, len(losses) + 1)
    loss_t = torch.tensor(losses)
    smooth = loss_t.unfold(0, window, 1).mean(dim=1) if len(losses) >= window else loss_t
    fig, ax = plt.subplots(figsize=(6, 3.5))
    ax.plot(steps, loss_t, color="#bbbbbb", linewidth=0.6, label="per step")
    ax.plot(steps[len(steps) - len(smooth):], smooth, color="#1f5fa8", linewidth=1.6, label=f"{window}-step mean")
    ax.set_xlabel("training step")
    ax.set_ylabel(r"$L_{simple}$ (MSE on $\epsilon$)")
    ax.set_yscale("log")
    ax.legend(frameon=False)
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--T", type=int, default=300)
    parser.add_argument("--base-ch", type=int, default=16)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--warmup", type=int, default=200)
    parser.add_argument("--ema-decay", type=float, default=0.995)
    parser.add_argument("--minutes", type=float, default=22.0, help="wall-clock training budget")
    parser.add_argument("--max-steps", type=int, default=100_000)
    parser.add_argument("--threads", type=int, default=2)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    torch.set_num_threads(args.threads)
    torch.manual_seed(args.seed)

    ckpt_dir = os.path.join(HERE, "checkpoints")
    assets_dir = os.path.join(HERE, "assets")
    os.makedirs(ckpt_dir, exist_ok=True)
    os.makedirs(assets_dir, exist_ok=True)

    data = load_mnist(os.path.join(HERE, "data"))
    diffusion = GaussianDiffusion(T=args.T)
    model = UNet(base_ch=args.base_ch)
    ema_model = copy.deepcopy(model).requires_grad_(False)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    print(f"{len(data)} images, {sum(p.numel() for p in model.parameters()):,} parameters, T={args.T}")

    def save(step, losses, elapsed):
        torch.save(
            {"model": model.state_dict(), "ema_model": ema_model.state_dict(), "config": vars(args),
             "step": step, "elapsed_sec": elapsed},
            os.path.join(ckpt_dir, "ddpm_mnist.pt"),
        )
        with open(os.path.join(ckpt_dir, "losses.json"), "w") as f:
            json.dump(losses, f)

    losses = []
    start = time.time()
    step = 0
    while step < args.max_steps and time.time() - start < args.minutes * 60:
        idx = torch.randint(0, len(data), (args.batch_size,))
        x0 = data[idx]

        lr = args.lr * min(1.0, (step + 1) / args.warmup)
        for group in optimizer.param_groups:
            group["lr"] = lr

        loss = diffusion.loss(model, x0)
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)  # paper clips gradients at norm 1
        optimizer.step()
        ema_update(ema_model, model, args.ema_decay)

        losses.append(loss.item())
        step += 1
        if step % 100 == 0:
            recent = sum(losses[-100:]) / 100
            print(f"step {step:6d} | loss (last 100) {recent:.4f} | {time.time() - start:6.0f}s", flush=True)
        if step % 1000 == 0:
            save(step, losses, time.time() - start)

    elapsed = time.time() - start
    save(step, losses, elapsed)
    plot_loss(losses, os.path.join(assets_dir, "loss_curve.png"))
    final = sum(losses[-500:]) / min(500, len(losses))
    print(f"done: {step} steps in {elapsed / 60:.1f} min, mean loss over last 500 steps {final:.4f}")


if __name__ == "__main__":
    main()
