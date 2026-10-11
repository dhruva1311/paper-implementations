"""Reproduce the plain-vs-residual comparison of Fig. 6 (Sec. 4.2) at small scale on CPU.

Trains four networks with identical data, data order, and hyperparameters:
plain-A, plain-B, ResNet-A, ResNet-B (A < B layers), then plots training error.

Usage:
    python train.py                         # quick run (4k subset, 2 epochs)
    python train.py --depths 20 56 --subset 4000 --epochs 2 --batch-size 64 --lr 0.05
"""
import argparse
import json
import math
import os
import time

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F
import torchvision

from model import plainnet, resnet, count_weighted_layers

HERE = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(HERE, "data")
ASSETS_DIR = os.path.join(HERE, "assets")
CKPT_DIR = os.path.join(HERE, "checkpoints")

# Per-channel CIFAR-10 statistics. The paper subtracts the per-pixel mean (Sec. 4.2);
# per-channel normalization is the common, simpler equivalent.
CIFAR_MEAN = torch.tensor([0.4914, 0.4822, 0.4465]).view(1, 3, 1, 1)
CIFAR_STD = torch.tensor([0.2470, 0.2435, 0.2616]).view(1, 3, 1, 1)


def load_cifar10(train, subset=None, seed=0):
    """Load CIFAR-10 into memory as a uint8 tensor (N, 3, 32, 32) and int64 labels."""
    ds = torchvision.datasets.CIFAR10(DATA_DIR, train=train, download=True)
    images = torch.from_numpy(ds.data).permute(0, 3, 1, 2).contiguous()
    labels = torch.tensor(ds.targets, dtype=torch.long)
    if subset is not None and subset < len(labels):
        idx = torch.randperm(len(labels), generator=torch.Generator().manual_seed(seed))[:subset]
        images, labels = images[idx], labels[idx]
    return images, labels


def normalize(images_uint8):
    return (images_uint8.float() / 255.0 - CIFAR_MEAN) / CIFAR_STD


def augment(images, generator):
    """Paper's augmentation (Sec. 4.2): pad 4 pixels, random 32x32 crop, random horizontal flip."""
    n = images.shape[0]
    padded = F.pad(images, (4, 4, 4, 4))
    out = torch.empty_like(images)
    offsets = torch.randint(0, 9, (n, 2), generator=generator)
    flips = torch.rand(n, generator=generator) < 0.5
    for i in range(n):
        dy, dx = offsets[i].tolist()
        crop = padded[i, :, dy:dy + 32, dx:dx + 32]
        out[i] = crop.flip(-1) if flips[i] else crop
    return out


@torch.no_grad()
def evaluate(model, images, labels, batch_size=500):
    model.eval()
    wrong = 0
    for start in range(0, len(labels), batch_size):
        x = normalize(images[start:start + batch_size])
        wrong += (model(x).argmax(1) != labels[start:start + batch_size]).sum().item()
    model.train()
    return wrong / len(labels)


def train_one(name, model, train_data, test_data, args):
    """SGD as in Sec. 4.2: momentum 0.9, weight decay 1e-4 (paper: lr 0.1, batch 128, lr /10 at 32k and 48k iters).

    Our runs are far shorter than the paper's first lr phase, so the lr is kept constant.
    """
    torch.manual_seed(args.seed)
    generator = torch.Generator().manual_seed(args.seed)  # same batches + augmentations for every model
    images, labels = train_data
    optimizer = torch.optim.SGD(model.parameters(), lr=args.lr, momentum=0.9, weight_decay=1e-4)

    steps_per_epoch = math.ceil(len(labels) / args.batch_size)
    history = {"iter": [], "train_err": [], "epoch": [], "epoch_train_err": [], "epoch_test_err": []}
    window_wrong, window_seen, step = 0, 0, 0
    start_time = time.time()

    model.train()
    for epoch in range(1, args.epochs + 1):
        perm = torch.randperm(len(labels), generator=generator)
        epoch_wrong = 0
        for b in range(steps_per_epoch):
            idx = perm[b * args.batch_size:(b + 1) * args.batch_size]
            x = normalize(augment(images[idx], generator))
            y = labels[idx]

            logits = model(x)
            loss = F.cross_entropy(logits, y)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            wrong = (logits.argmax(1) != y).sum().item()
            epoch_wrong += wrong
            window_wrong += wrong
            window_seen += len(y)
            step += 1
            if step % args.log_every == 0:
                history["iter"].append(step)
                history["train_err"].append(window_wrong / window_seen)
                window_wrong, window_seen = 0, 0

        test_err = evaluate(model, *test_data)
        history["epoch"].append(epoch)
        history["epoch_train_err"].append(epoch_wrong / len(labels))
        history["epoch_test_err"].append(test_err)
        print(f"[{name}] epoch {epoch}/{args.epochs}  train err {epoch_wrong / len(labels):.3f}  "
              f"test err {test_err:.3f}  ({time.time() - start_time:.0f}s)", flush=True)

    history["seconds"] = time.time() - start_time
    os.makedirs(CKPT_DIR, exist_ok=True)
    torch.save(model.state_dict(), os.path.join(CKPT_DIR, f"{name}.pt"))
    return history


def plot(results, depths, path):
    """Two panels like Fig. 6: plain nets (left) and ResNets (right), training error vs iteration."""
    fig, axes = plt.subplots(1, 2, figsize=(11, 4), sharey=True)
    colors = {depths[0]: "#1f77b4", depths[1]: "#d62728"}
    for ax, kind, title in [(axes[0], "plain", "Plain networks"), (axes[1], "resnet", "ResNets")]:
        for d in depths:
            h = results[f"{kind}-{d}"]
            ax.plot(h["iter"], np.array(h["train_err"]) * 100, color=colors[d], label=f"{kind}-{d}")
        ax.set_title(title)
        ax.set_xlabel("iteration")
        ax.grid(alpha=0.3)
        ax.legend()
    axes[0].set_ylabel("training error (%)")
    fig.tight_layout()
    fig.savefig(path, dpi=120)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--depths", type=int, nargs=2, default=[20, 56], help="two depths of the form 6n+2")
    parser.add_argument("--subset", type=int, default=4000, help="number of CIFAR-10 training images")
    parser.add_argument("--test-subset", type=int, default=1000, help="number of CIFAR-10 test images")
    parser.add_argument("--epochs", type=int, default=2)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--lr", type=float, default=0.05, help="paper: 0.1 at batch 128; scaled linearly")
    parser.add_argument("--log-every", type=int, default=4, help="iterations per training-error point")
    parser.add_argument("--threads", type=int, default=2)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    torch.set_num_threads(args.threads)
    train_data = load_cifar10(train=True, subset=args.subset, seed=args.seed)
    test_data = load_cifar10(train=False, subset=args.test_subset, seed=args.seed)
    print(f"train {len(train_data[1])} images, test {len(test_data[1])} images, "
          f"{args.epochs} epochs, batch {args.batch_size}", flush=True)

    results = {}
    for kind, build in [("plain", plainnet), ("resnet", resnet)]:
        for depth in args.depths:
            name = f"{kind}-{depth}"
            torch.manual_seed(args.seed)
            model = build(depth)
            n_params = sum(p.numel() for p in model.parameters())
            print(f"--- {name}: {count_weighted_layers(model)} weighted layers, {n_params / 1e6:.3f}M params", flush=True)
            results[name] = train_one(name, model, train_data, test_data, args)

    os.makedirs(ASSETS_DIR, exist_ok=True)
    tag = f"{args.depths[0]}v{args.depths[1]}"
    plot(results, args.depths, os.path.join(ASSETS_DIR, f"plain_vs_resnet_{tag}.png"))
    with open(os.path.join(ASSETS_DIR, f"results_{tag}.json"), "w") as f:
        json.dump({"args": vars(args), "results": results}, f, indent=1)

    print("\nsummary (final epoch):")
    for name, h in results.items():
        print(f"  {name:>10}: train err {h['epoch_train_err'][-1]:.3f}  test err {h['epoch_test_err'][-1]:.3f}  "
              f"time {h['seconds']:.0f}s")


if __name__ == "__main__":
    main()
