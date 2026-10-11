"""Plots: training curves, position-embedding similarity (paper Fig. 7) and
attention rollout from [CLS] to the patches (paper Fig. 6 / Appendix D.8).

Can also be run standalone on a saved checkpoint:
    python visualize.py
"""

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import torch
import torch.nn.functional as F
from matplotlib.colors import LinearSegmentedColormap

from model import ViT

BLUE, ORANGE = "#2a78d6", "#eb6834"
INK, MUTED, GRID = "#0b0b0b", "#52514e", "#e4e3df"
SEQUENTIAL = LinearSegmentedColormap.from_list(
    "seq_blue", ["#fcfcfb", "#cde2fb", "#86b6ef", "#3987e5", "#256abf", "#104281", "#0d366b"])
DIVERGING = LinearSegmentedColormap.from_list(
    "div_orange_blue", ["#a8400f", ORANGE, "#f2f1ee", BLUE, "#104281"])
CLASSES = ["T-shirt", "Trouser", "Pullover", "Dress", "Coat",
           "Sandal", "Shirt", "Sneaker", "Bag", "Boot"]


def _style(ax):
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(GRID)
    ax.tick_params(colors=MUTED, labelsize=9)
    ax.grid(color=GRID, linewidth=0.8)
    ax.set_axisbelow(True)


def plot_curves(history: dict, path: Path) -> None:
    fig, (ax_loss, ax_acc) = plt.subplots(1, 2, figsize=(11, 4))
    fig.patch.set_facecolor("white")

    ax_loss.plot(history["step"], history["train_loss"], color=BLUE, lw=2, label="train (25-step mean)")
    ax_loss.plot(history["epoch_end_step"], history["test_loss"], color=ORANGE, lw=2,
                 marker="o", ms=8, label="test (end of epoch)")
    ax_loss.set_title("Cross-entropy loss", color=INK, loc="left")
    ax_loss.set_xlabel("optimizer step", color=MUTED)

    ax_acc.plot(history["epoch"], [100 * a for a in history["epoch_train_acc"]], color=BLUE, lw=2,
                marker="o", ms=8, label="train (running mean over epoch)")
    ax_acc.plot(history["epoch"], [100 * a for a in history["test_acc"]], color=ORANGE, lw=2,
                marker="o", ms=8, label="test")
    ax_acc.annotate(f"{100 * history['test_acc'][-1]:.2f}%",
                    (history["epoch"][-1], 100 * history["test_acc"][-1]),
                    textcoords="offset points", xytext=(0, -18), ha="center", color=INK, fontsize=9)
    ax_acc.set_title("Accuracy (%)", color=INK, loc="left")
    ax_acc.set_xlabel("epoch", color=MUTED)
    ax_acc.set_xticks(history["epoch"])

    for ax in (ax_loss, ax_acc):
        _style(ax)
        ax.legend(frameon=False, fontsize=9, labelcolor=INK)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


@torch.no_grad()
def plot_position_similarity(model: ViT, path: Path) -> None:
    """Cosine similarity of each patch's position embedding with all others (paper Fig. 7, center)."""
    pos = model.pos_embed[0, 1:]                              # drop the [CLS] position, (N, D)
    grid = int(pos.shape[0] ** 0.5)
    pos = F.normalize(pos, dim=-1)
    sim = (pos @ pos.T).reshape(grid, grid, grid, grid)       # sim[r, c] is a (grid, grid) map

    fig, axes = plt.subplots(grid, grid, figsize=(7.5, 7))
    fig.patch.set_facecolor("white")
    for r in range(grid):
        for c in range(grid):
            im = axes[r, c].imshow(sim[r, c].numpy(), cmap=DIVERGING, vmin=-1, vmax=1)
            axes[r, c].set_xticks([])
            axes[r, c].set_yticks([])
            if c == 0:
                axes[r, c].set_ylabel(str(r + 1), color=MUTED, fontsize=9, rotation=0, labelpad=8)
            if r == grid - 1:
                axes[r, c].set_xlabel(str(c + 1), color=MUTED, fontsize=9)
    fig.supxlabel("Input patch column", color=INK)
    fig.supylabel("Input patch row", color=INK)
    fig.suptitle("Position embedding similarity", color=INK)
    cbar = fig.colorbar(im, ax=axes, shrink=0.6, pad=0.03)
    cbar.set_label("Cosine similarity", color=MUTED)
    cbar.ax.tick_params(colors=MUTED, labelsize=8)
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)


@torch.no_grad()
def attention_rollout(model: ViT, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Attention rollout (Abnar & Zuidema, 2020), as used in the paper's Appendix D.8.

    Average attention over heads, add the identity for the residual connection,
    re-normalize rows, then multiply the matrices of all layers together.
    Returns the rollout [CLS] -> patch maps (B, grid, grid) and the logits.
    """
    model.eval()
    logits = model(x)
    rollout = None
    for attn in model.attention_maps():
        a = attn.mean(dim=1)                                  # (B, T, T)
        a = a + torch.eye(a.shape[-1])
        a = a / a.sum(dim=-1, keepdim=True)
        rollout = a if rollout is None else a @ rollout
    cls_to_patches = rollout[:, 0, 1:]                        # (B, N)
    grid = int(cls_to_patches.shape[-1] ** 0.5)
    return cls_to_patches.reshape(-1, grid, grid), logits


def plot_attention(model: ViT, x: torch.Tensor, y: torch.Tensor, path: Path,
                   mean: float, std: float) -> None:
    maps, logits = attention_rollout(model, x)
    preds = logits.argmax(1)
    maps = F.interpolate(maps[:, None], size=x.shape[-2:], mode="nearest")[:, 0]
    images = x[:, 0] * std + mean

    n = x.shape[0]
    fig, axes = plt.subplots(2, n, figsize=(1.6 * n, 3.6))
    fig.patch.set_facecolor("white")
    for i in range(n):
        axes[0, i].imshow(images[i].numpy(), cmap="gray_r")
        axes[0, i].set_title(f"{CLASSES[y[i]]}\npred: {CLASSES[preds[i]]}", fontsize=8, color=INK)
        m = maps[i]
        axes[1, i].imshow(((m - m.min()) / (m.max() - m.min() + 1e-8)).numpy(), cmap=SEQUENTIAL)
        for ax in axes[:, i]:
            ax.set_xticks([])
            ax.set_yticks([])
    axes[0, 0].set_ylabel("input", color=MUTED)
    axes[1, 0].set_ylabel("[CLS] rollout", color=MUTED)
    fig.suptitle("Attention rollout from [CLS] to patches (darker = more attention)", color=INK, fontsize=10)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


if __name__ == "__main__":
    import json

    from train import MEAN, ROOT, STD, get_loaders

    torch.set_num_threads(2)
    ckpt = torch.load(ROOT / "checkpoints" / "vit_fashionmnist.pt", map_location="cpu")
    a = ckpt["args"]
    model = ViT(patch_size=a["patch_size"], dim=a["dim"], depth=a["depth"],
                num_heads=a["heads"], mlp_dim=a["mlp_dim"])
    model.load_state_dict(ckpt["model"])
    assets = ROOT / "assets"
    history = json.loads((assets / "metrics.json").read_text())["history"]
    plot_curves(history, assets / "training_curves.png")
    plot_position_similarity(model, assets / "position_embedding_similarity.png")
    _, test_loader = get_loaders(batch_size=128)
    x, y = next(iter(test_loader))
    plot_attention(model, x[:8], y[:8], assets / "attention_maps.png", mean=MEAN, std=STD)
