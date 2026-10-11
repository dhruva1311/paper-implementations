"""Train a small ViT on FashionMNIST (CPU) and save metrics, checkpoint and plots.

Usage:
    python train.py                 # default config used for the README results
    python train.py --epochs 1      # quick smoke run
"""

import argparse
import json
import math
import random
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset
from torchvision import datasets

from model import ViT
from visualize import plot_attention, plot_curves, plot_position_similarity

ROOT = Path(__file__).resolve().parent
MEAN, STD = 0.2860, 0.3530   # FashionMNIST training-set statistics


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--epochs", type=int, default=4)
    p.add_argument("--batch-size", type=int, default=128)
    p.add_argument("--lr", type=float, default=3e-3)
    p.add_argument("--weight-decay", type=float, default=0.05)
    p.add_argument("--warmup-steps", type=int, default=200)
    p.add_argument("--patch-size", type=int, default=7)
    p.add_argument("--dim", type=int, default=64)
    p.add_argument("--depth", type=int, default=4)
    p.add_argument("--heads", type=int, default=4)
    p.add_argument("--mlp-dim", type=int, default=128)
    p.add_argument("--dropout", type=float, default=0.0)
    p.add_argument("--time-budget-min", type=float, default=13.0,
                   help="stop training early if wall time exceeds this many minutes")
    p.add_argument("--threads", type=int, default=2)
    p.add_argument("--seed", type=int, default=0)
    return p.parse_args()


def get_loaders(batch_size: int) -> tuple[DataLoader, DataLoader]:
    """Load FashionMNIST into memory as normalized tensors (no augmentation)."""
    data_dir = ROOT / "data"
    loaders = []
    for train in (True, False):
        ds = datasets.FashionMNIST(data_dir, train=train, download=True)
        x = (ds.data.float() / 255.0 - MEAN) / STD         # (n, 28, 28)
        tensors = TensorDataset(x.unsqueeze(1), ds.targets)  # (n, 1, 28, 28)
        loaders.append(DataLoader(tensors, batch_size=batch_size if train else 1000, shuffle=train))
    return loaders[0], loaders[1]


def lr_at(step: int, total_steps: int, warmup_steps: int, base_lr: float) -> float:
    """Linear warmup followed by cosine decay to zero (as in the paper's setup)."""
    if step < warmup_steps:
        return base_lr * (step + 1) / warmup_steps
    progress = (step - warmup_steps) / max(1, total_steps - warmup_steps)
    return 0.5 * base_lr * (1 + math.cos(math.pi * progress))


@torch.no_grad()
def evaluate(model: ViT, loader: DataLoader) -> tuple[float, float]:
    model.eval()
    total_loss, correct, n = 0.0, 0, 0
    for x, y in loader:
        logits = model(x)
        total_loss += F.cross_entropy(logits, y, reduction="sum").item()
        correct += (logits.argmax(1) == y).sum().item()
        n += y.numel()
    return total_loss / n, correct / n


def main() -> None:
    args = parse_args()
    torch.set_num_threads(args.threads)
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    train_loader, test_loader = get_loaders(args.batch_size)
    model = ViT(image_size=28, patch_size=args.patch_size, in_channels=1, num_classes=10,
                dim=args.dim, depth=args.depth, num_heads=args.heads,
                mlp_dim=args.mlp_dim, dropout=args.dropout)
    num_params = sum(p.numel() for p in model.parameters())
    print(f"ViT parameters: {num_params:,}")

    # No weight decay on biases, LayerNorm weights, cls token and position embeddings.
    decay, no_decay = [], []
    for name, p in model.named_parameters():
        if p.ndim < 2 or name in ("cls_token", "pos_embed"):
            no_decay.append(p)
        else:
            decay.append(p)
    optimizer = torch.optim.AdamW(
        [{"params": decay, "weight_decay": args.weight_decay},
         {"params": no_decay, "weight_decay": 0.0}],
        lr=args.lr,
    )

    total_steps = args.epochs * len(train_loader)
    history = {"step": [], "train_loss": [], "epoch": [], "epoch_train_acc": [],
               "test_loss": [], "test_acc": [], "epoch_end_step": []}
    step, stopped_early, running = 0, False, 0.0
    start = time.time()

    for epoch in range(1, args.epochs + 1):
        model.train()
        correct, seen = 0, 0
        for x, y in train_loader:
            for group in optimizer.param_groups:
                group["lr"] = lr_at(step, total_steps, args.warmup_steps, args.lr)

            logits = model(x)
            loss = F.cross_entropy(logits, y)
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

            running += loss.item()
            correct += (logits.argmax(1) == y).sum().item()
            seen += y.numel()
            step += 1
            if step % 25 == 0:
                history["step"].append(step)
                history["train_loss"].append(running / 25)
                running = 0.0
            if step % 100 == 0:
                print(f"epoch {epoch} step {step}/{total_steps} "
                      f"loss {history['train_loss'][-1]:.4f} "
                      f"elapsed {(time.time() - start) / 60:.1f} min", flush=True)
            if time.time() - start > args.time_budget_min * 60:
                stopped_early = True
                break

        test_loss, test_acc = evaluate(model, test_loader)
        history["epoch"].append(epoch)
        history["epoch_train_acc"].append(correct / seen)
        history["test_loss"].append(test_loss)
        history["test_acc"].append(test_acc)
        history["epoch_end_step"].append(step)
        print(f"== epoch {epoch}: train acc {correct / seen:.4f} | "
              f"test loss {test_loss:.4f} | test acc {test_acc:.4f}", flush=True)
        if stopped_early:
            print(f"Time budget of {args.time_budget_min} min reached, stopping.")
            break

    wall_min = (time.time() - start) / 60
    print(f"Training + evaluation wall time: {wall_min:.2f} min")

    (ROOT / "checkpoints").mkdir(exist_ok=True)
    torch.save({"model": model.state_dict(), "args": vars(args)}, ROOT / "checkpoints" / "vit_fashionmnist.pt")

    assets = ROOT / "assets"
    assets.mkdir(exist_ok=True)
    metrics = {
        "dataset": "FashionMNIST",
        "num_params": num_params,
        "final_test_acc": history["test_acc"][-1],
        "final_test_loss": history["test_loss"][-1],
        "epochs_completed": len(history["epoch"]),
        "stopped_early": stopped_early,
        "train_wall_time_min": round(wall_min, 2),
        "threads": args.threads,
        "config": vars(args),
        "history": history,
    }
    (assets / "metrics.json").write_text(json.dumps(metrics, indent=2))

    plot_curves(history, assets / "training_curves.png")
    plot_position_similarity(model, assets / "position_embedding_similarity.png")
    x, y = next(iter(test_loader))
    plot_attention(model, x[:8], y[:8], assets / "attention_maps.png", mean=MEAN, std=STD)
    print(f"Saved metrics and plots to {assets}")


if __name__ == "__main__":
    main()
