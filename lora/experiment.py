"""Full fine-tuning vs LoRA on a small pretrained MLP (CPU, ~14 min).

1. Pretrain an MLP on MNIST digits (cached in checkpoints/).
2. Adapt it to FashionMNIST (same 28x28 inputs, same 10-way output) with:
     - no adaptation (pretrained model as-is, chance-level reference)
     - head-only fine-tuning (train only the last Linear)
     - full fine-tuning (train every parameter)
     - LoRA with r in {1, 2, 4, 8} on every Linear layer
3. Check that merging the LoRA weights leaves test accuracy unchanged.
4. Save results/results.md, results/results.json and assets/*.png.

Usage: python experiment.py [--seeds 3] [--epochs 3] [--plot-only]
"""

import argparse
import json
import time
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision

from lora import count_parameters, inject_lora, lora_state_dict, merge_lora

torch.set_num_threads(2)

ROOT = Path(__file__).resolve().parent
DATA_DIR = ROOT / "data"
CKPT_DIR = ROOT / "checkpoints"
ASSETS_DIR = ROOT / "assets"
RESULTS_DIR = ROOT / "results"

RANKS = [1, 2, 4, 8]
LORA_ALPHA = 8  # fixed across ranks, so the update is scaled by 8 / r
TARGETS = ["fc1", "fc2", "head"]


class MLP(nn.Module):
    def __init__(self, hidden: int = 512, num_classes: int = 10):
        super().__init__()
        self.fc1 = nn.Linear(28 * 28, hidden)
        self.fc2 = nn.Linear(hidden, hidden)
        self.head = nn.Linear(hidden, num_classes)

    def forward(self, x):
        x = x.flatten(1)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        return self.head(x)


def load_dataset(name: str, train: bool):
    """Whole dataset as in-memory tensors; the images are tiny, so no DataLoader needed."""
    cls = {"mnist": torchvision.datasets.MNIST, "fashion": torchvision.datasets.FashionMNIST}[name]
    ds = cls(DATA_DIR, train=train, download=True)
    x = ds.data.float().div(255.0)
    x = (x - x.mean()) / x.std()
    return x, ds.targets.clone()


def train(model, x, y, epochs, lr, batch_size=128, seed=0):
    params = [p for p in model.parameters() if p.requires_grad]
    opt = torch.optim.AdamW(params, lr=lr, weight_decay=0.0)
    g = torch.Generator().manual_seed(seed)
    model.train()
    for _ in range(epochs):
        perm = torch.randperm(len(x), generator=g)
        for i in range(0, len(x), batch_size):
            idx = perm[i:i + batch_size]
            loss = F.cross_entropy(model(x[idx]), y[idx])
            opt.zero_grad()
            loss.backward()
            opt.step()


@torch.no_grad()
def accuracy(model, x, y, batch_size=2000):
    model.eval()
    correct = sum((model(x[i:i + batch_size]).argmax(1) == y[i:i + batch_size]).sum().item()
                  for i in range(0, len(x), batch_size))
    return correct / len(x)


def pretrain(epochs: int) -> MLP:
    path = CKPT_DIR / "mlp_mnist.pt"
    model = MLP()
    if path.exists():
        model.load_state_dict(torch.load(path))
        print(f"loaded pretrained model from {path}")
        return model
    torch.manual_seed(0)
    x, y = load_dataset("mnist", train=True)
    xt, yt = load_dataset("mnist", train=False)
    t0 = time.time()
    train(model, x, y, epochs=epochs, lr=1e-3)
    print(f"pretrained on MNIST in {time.time() - t0:.0f}s, test acc {accuracy(model, xt, yt):.4f}")
    CKPT_DIR.mkdir(exist_ok=True)
    torch.save(model.state_dict(), path)
    return model


def make_variant(pretrained: MLP, method: str, r: int | None) -> MLP:
    model = MLP()
    model.load_state_dict(pretrained.state_dict())
    if method == "head-only":
        for name, p in model.named_parameters():
            p.requires_grad = name.startswith("head.")
    elif method == "lora":
        inject_lora(model, TARGETS, r=r, alpha=LORA_ALPHA)
    return model


def run(args):
    for d in (CKPT_DIR, ASSETS_DIR, RESULTS_DIR):
        d.mkdir(exist_ok=True)
    if args.plot_only:
        plot(json.loads((RESULTS_DIR / "results.json").read_text())["rows"])
        return
    pretrained = pretrain(args.pretrain_epochs)
    x, y = load_dataset("fashion", train=True)
    xt, yt = load_dataset("fashion", train=False)

    configs = [("head-only", None, args.lr_full), ("full", None, args.lr_full)]
    configs += [("lora", r, args.lr_lora) for r in RANKS]

    rows = [{"method": "no adaptation", "r": None,
             "trainable": 0, "total": count_parameters(pretrained),
             "accs": [accuracy(pretrained, xt, yt)], "seconds": [0.0]}]

    for method, r, lr in configs:
        row = {"method": method if r is None else f"LoRA r={r}", "r": r, "accs": [], "seconds": []}
        for seed in range(args.seeds):
            torch.manual_seed(seed)
            model = make_variant(pretrained, method, r)
            t0 = time.time()
            train(model, x, y, epochs=args.epochs, lr=lr, seed=seed)
            row["seconds"].append(time.time() - t0)
            acc = accuracy(model, xt, yt)
            row["accs"].append(acc)

            if method == "lora":
                merge_lora(model)
                merged_acc = accuracy(model, xt, yt)
                assert merged_acc == acc, f"merge changed accuracy: {acc} -> {merged_acc}"
                row["merged_acc_matches"] = True
                if seed == 0:
                    torch.save(lora_state_dict(model), CKPT_DIR / f"lora_r{r}.pt")
            row["trainable"] = count_parameters(model, trainable_only=True)
            row["total"] = count_parameters(model)
            print(f"{row['method']:<12} seed={seed} acc={acc:.4f} "
                  f"trainable={row['trainable']:,} time={row['seconds'][-1]:.0f}s")
        rows.append(row)

    for row in rows:
        row["mean"] = float(np.mean(row["accs"]))
        row["std"] = float(np.std(row["accs"]))
        row["pct_of_base"] = 100.0 * row["trainable"] / rows[0]["total"]

    (RESULTS_DIR / "results.json").write_text(json.dumps({"args": vars(args), "rows": rows}, indent=2))
    write_table(rows, args)
    plot(rows)


def write_table(rows, args):
    lines = [
        f"FashionMNIST test accuracy after {args.epochs} epochs, mean +/- std over "
        f"{args.seeds} seeds (no adaptation is deterministic, 1 run).",
        "",
        "| Method | Trainable params | % of base model | Test accuracy | Train time / run |",
        "|---|---:|---:|---:|---:|",
    ]
    for row in rows:
        acc = f"{100 * row['mean']:.2f}%"
        if len(row["accs"]) > 1:
            acc += f" +/- {100 * row['std']:.2f}"
        secs = f"{np.mean(row['seconds']):.0f}s" if row["trainable"] else "-"
        lines.append(f"| {row['method']} | {row['trainable']:,} | {row['pct_of_base']:.2f}% "
                     f"| {acc} | {secs} |")
    table = "\n".join(lines) + "\n"
    (RESULTS_DIR / "results.md").write_text(table)
    print("\n" + table)


def plot(rows):
    lora_rows = [r for r in rows if r["r"] is not None]
    by_name = {r["method"]: r for r in rows}
    full, head, base = by_name["full"], by_name["head-only"], by_name["no adaptation"]
    blue, orange, aqua = "#2a78d6", "#eb6834", "#1baf7a"

    fig, ax = plt.subplots(figsize=(7, 4.5), dpi=150)
    xs = [r["trainable"] for r in lora_rows]
    ys = [100 * r["mean"] for r in lora_rows]
    es = [100 * r["std"] for r in lora_rows]
    ax.errorbar(xs, ys, yerr=es, color=blue, marker="o", markersize=7, linewidth=2,
                capsize=3, label="LoRA (r = 1, 2, 4, 8)")
    for r, xv, yv in zip(lora_rows, xs, ys):
        ax.annotate(f"r={r['r']}", (xv, yv), textcoords="offset points", xytext=(0, 9),
                    ha="center", fontsize=9, color="#0b0b0b")
    ax.errorbar([full["trainable"]], [100 * full["mean"]], yerr=[100 * full["std"]],
                color=orange, marker="s", markersize=8, capsize=3, linestyle="none",
                label="Full fine-tuning")
    ax.errorbar([head["trainable"]], [100 * head["mean"]], yerr=[100 * head["std"]],
                color=aqua, marker="D", markersize=7, capsize=3, linestyle="none",
                label="Head-only fine-tuning")
    for row, label in ((full, "full FT"), (head, "head only")):
        ax.annotate(label, (row["trainable"], 100 * row["mean"]), textcoords="offset points",
                    xytext=(0, 10), ha="center", fontsize=9, color="#0b0b0b")

    ax.set_xscale("log")
    ax.set_xlabel("Trainable parameters (log scale)")
    ax.set_ylabel("FashionMNIST test accuracy (%)")
    ax.set_title("MNIST-pretrained MLP adapted to FashionMNIST\n"
                 f"(no adaptation: {100 * base['mean']:.1f}%, off-scale)", fontsize=11)
    lo = min(ys + [100 * head["mean"]]) - 4
    ax.set_ylim(lo, 100 * full["mean"] + 3)
    ax.grid(True, which="major", color="#e4e3df", linewidth=0.8)
    ax.spines[["top", "right"]].set_visible(False)
    ax.legend(loc="lower right", frameon=False, fontsize=9)
    fig.tight_layout()
    out = ASSETS_DIR / "accuracy_vs_params.png"
    fig.savefig(out)
    print(f"saved {out}")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--seeds", type=int, default=3)
    p.add_argument("--epochs", type=int, default=3)
    p.add_argument("--pretrain-epochs", type=int, default=3)
    p.add_argument("--lr-full", type=float, default=1e-3)
    p.add_argument("--lr-lora", type=float, default=3e-3)
    p.add_argument("--plot-only", action="store_true", help="redraw the plot from results.json")
    run(p.parse_args())
