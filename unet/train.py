"""Train a small U-Net on Oxford-IIIT Pet (pet vs. background) on CPU.

Usage:
    python train.py                       # defaults used for the README results
    python train.py --epochs 2 --base-channels 8 --train-subset 500   # quick smoke run

Writes the best checkpoint (by validation Dice) to checkpoints/unet.pt, the
loss curves to assets/loss_curves.png, a prediction grid to
assets/predictions.png and the final numbers to assets/metrics.json.
"""

import argparse
import json
import time
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Subset
from tqdm import tqdm

from dataset import PetSegmentation, load_pets
from model import UNet, dice_score, iou_score

HERE = Path(__file__).resolve().parent
TRAIN_COLOR = "#2a78d6"
VAL_COLOR = "#eb6834"
TEXT_SECONDARY = "#52514e"


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--size", type=int, default=64, help="images are resized to size x size")
    p.add_argument("--base-channels", type=int, default=8)
    p.add_argument("--depth", type=int, default=4)
    p.add_argument("--epochs", type=int, default=8)
    p.add_argument("--batch-size", type=int, default=32)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--val-fraction", type=float, default=0.1)
    p.add_argument("--train-subset", type=int, default=0, help="use only N training images (0 = all)")
    p.add_argument("--threads", type=int, default=2)
    p.add_argument("--seed", type=int, default=0)
    return p.parse_args()


def train_one_epoch(model, loader, optimizer) -> list[float]:
    model.train()
    losses = []
    for images, masks in tqdm(loader, desc="train", leave=False):
        # Section 3: pixel-wise softmax + cross-entropy.
        loss = F.cross_entropy(model(images), masks)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        losses.append(loss.item())
    return losses


@torch.no_grad()
def evaluate(model, loader) -> dict[str, float]:
    """Mean cross-entropy over pixels, and Dice/IoU of the pet class averaged over images."""
    model.eval()
    loss_sum, pixels, dice, iou = 0.0, 0, [], []
    for images, masks in loader:
        logits = model(images)
        loss_sum += F.cross_entropy(logits, masks, reduction="sum").item()
        pixels += masks.numel()
        pred = logits.argmax(dim=1)
        dice.append(dice_score(pred, masks))
        iou.append(iou_score(pred, masks))
    return {
        "loss": loss_sum / pixels,
        "dice": torch.cat(dice).mean().item(),
        "iou": torch.cat(iou).mean().item(),
    }


def plot_curves(history: dict, steps_per_epoch: int, path: Path) -> None:
    fig, (ax_loss, ax_metric) = plt.subplots(1, 2, figsize=(11, 4))
    epochs = range(1, len(history["val_loss"]) + 1)

    step_epochs = [(i + 1) / steps_per_epoch for i in range(len(history["train_step_loss"]))]
    ax_loss.plot(step_epochs, history["train_step_loss"], color=TRAIN_COLOR, lw=1, alpha=0.35)
    ax_loss.plot(epochs, history["train_loss"], color=TRAIN_COLOR, lw=2, marker="o", ms=5, label="train (epoch mean)")
    ax_loss.plot(epochs, history["val_loss"], color=VAL_COLOR, lw=2, marker="o", ms=5, label="validation")
    ax_loss.set_title("Cross-entropy loss", loc="left")
    ax_loss.set_xlabel("epoch")
    ax_loss.legend(frameon=False)

    ax_metric.plot(epochs, history["val_dice"], color=TRAIN_COLOR, lw=2, marker="o", ms=5, label="Dice")
    ax_metric.plot(epochs, history["val_iou"], color=VAL_COLOR, lw=2, marker="o", ms=5, label="IoU")
    for key, color in (("val_dice", TRAIN_COLOR), ("val_iou", VAL_COLOR)):
        ax_metric.annotate(f"{history[key][-1]:.3f}", (len(history[key]), history[key][-1]),
                           xytext=(6, 0), textcoords="offset points", va="center", color=TEXT_SECONDARY)
    ax_metric.set_title("Validation Dice / IoU (pet class)", loc="left")
    ax_metric.set_xlabel("epoch")
    ax_metric.legend(frameon=False, loc="lower right")

    for ax in (ax_loss, ax_metric):
        ax.grid(alpha=0.25)
        ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    fig.savefig(path, dpi=120)
    plt.close(fig)


@torch.no_grad()
def plot_predictions(model, dataset, path: Path, n: int = 8) -> None:
    """Evenly spaced test images (the test split is sorted by breed), not cherry-picked."""
    model.eval()
    indices = torch.linspace(0, len(dataset) - 1, n).long().tolist()
    images, masks = zip(*(dataset[i] for i in indices))
    images, masks = torch.stack(images), torch.stack(masks)
    preds = model(images).argmax(dim=1)
    dice = dice_score(preds, masks)

    fig, axes = plt.subplots(3, n, figsize=(1.6 * n, 5.2))
    for i in range(n):
        axes[0, i].imshow(images[i].permute(1, 2, 0))
        axes[1, i].imshow(masks[i], cmap="gray", vmin=0, vmax=1)
        axes[2, i].imshow(preds[i], cmap="gray", vmin=0, vmax=1)
        axes[2, i].set_xlabel(f"Dice {dice[i]:.2f}", fontsize=9)
    for row, title in enumerate(("input", "ground truth", "prediction")):
        axes[row, 0].set_ylabel(title)
    for ax in axes.flat:
        ax.set_xticks([])
        ax.set_yticks([])
    fig.tight_layout()
    fig.savefig(path, dpi=120)
    plt.close(fig)


def main() -> None:
    args = parse_args()
    torch.set_num_threads(args.threads)
    torch.manual_seed(args.seed)
    (HERE / "checkpoints").mkdir(exist_ok=True)
    (HERE / "assets").mkdir(exist_ok=True)

    data = load_pets(HERE / "data", args.size)
    trainval_images, trainval_masks = data["trainval"]
    perm = torch.randperm(len(trainval_images), generator=torch.Generator().manual_seed(args.seed))
    n_val = int(len(perm) * args.val_fraction)
    val_idx, train_idx = perm[:n_val], perm[n_val:]
    if args.train_subset:
        train_idx = train_idx[: args.train_subset]

    train_set = PetSegmentation(trainval_images[train_idx], trainval_masks[train_idx], augment=True)
    val_set = PetSegmentation(trainval_images[val_idx], trainval_masks[val_idx])
    test_set = PetSegmentation(*data["test"])
    print(f"train {len(train_set)} | val {len(val_set)} | test {len(test_set)} images at {args.size}x{args.size}")

    train_loader = DataLoader(train_set, batch_size=args.batch_size, shuffle=True)
    val_loader = DataLoader(val_set, batch_size=128)
    test_loader = DataLoader(test_set, batch_size=128)

    model = UNet(in_channels=3, num_classes=2, base_channels=args.base_channels, depth=args.depth,
                 padding=True, batch_norm=True)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"UNet(base_channels={args.base_channels}, depth={args.depth}): {n_params:,} parameters")
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)

    history = {"train_step_loss": [], "train_loss": [], "val_loss": [], "val_dice": [], "val_iou": []}
    best_dice, ckpt_path = -1.0, HERE / "checkpoints" / "unet.pt"
    start = time.time()
    for epoch in range(1, args.epochs + 1):
        step_losses = train_one_epoch(model, train_loader, optimizer)
        scheduler.step()
        val = evaluate(model, val_loader)
        history["train_step_loss"] += step_losses
        history["train_loss"].append(sum(step_losses) / len(step_losses))
        history["val_loss"].append(val["loss"])
        history["val_dice"].append(val["dice"])
        history["val_iou"].append(val["iou"])
        print(f"epoch {epoch:2d} | train loss {history['train_loss'][-1]:.4f} | val loss {val['loss']:.4f} "
              f"| val Dice {val['dice']:.4f} | val IoU {val['iou']:.4f} | {time.time() - start:.0f}s")
        if val["dice"] > best_dice:
            best_dice = val["dice"]
            torch.save({"model": model.state_dict(), "args": vars(args), "epoch": epoch}, ckpt_path)
    train_seconds = time.time() - start

    checkpoint = torch.load(ckpt_path)
    model.load_state_dict(checkpoint["model"])
    test = evaluate(model, test_loader)
    print(f"test (best val checkpoint, epoch {checkpoint['epoch']}) | loss {test['loss']:.4f} "
          f"| Dice {test['dice']:.4f} | IoU {test['iou']:.4f}")

    plot_curves(history, len(train_loader), HERE / "assets" / "loss_curves.png")
    plot_predictions(model, test_set, HERE / "assets" / "predictions.png")
    results = {
        "args": vars(args),
        "parameters": n_params,
        "train_seconds": round(train_seconds, 1),
        "best_epoch": checkpoint["epoch"],
        "test": test,
        "history": {k: v for k, v in history.items() if k != "train_step_loss"},
    }
    (HERE / "assets" / "metrics.json").write_text(json.dumps(results, indent=2))


if __name__ == "__main__":
    main()
