"""Train the from-scratch Transformer on the string-reversal task (CPU friendly).

    python train.py                 # full run (defaults below), ~12 min on 2 CPU threads
    python train.py --steps 200     # quick smoke test

Outputs:
    checkpoints/transformer_reverse.pt   model weights + config (gitignored)
    assets/loss.png                      training/validation curves
    assets/attention.png                 cross-attention heatmaps
    assets/samples.txt                   sample predictions
    assets/metrics.json                  final metrics and training history
"""
import argparse
import json
import os
import random
import time

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from matplotlib.colors import LinearSegmentedColormap

import data
from model import LabelSmoothingLoss, Transformer, noam_lr

HERE = os.path.dirname(os.path.abspath(__file__))
ASSETS = os.path.join(HERE, "assets")
CHECKPOINTS = os.path.join(HERE, "checkpoints")

# Plot colours
BLUE, ORANGE = "#2a78d6", "#eb6834"
INK, MUTED = "#0b0b0b", "#52514e"
SEQ_CMAP = LinearSegmentedColormap.from_list("blues", ["#fcfcfb", "#9ec5f4", "#2a78d6", "#0d366b"])


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--d-model", type=int, default=128)
    p.add_argument("--heads", type=int, default=4)
    p.add_argument("--layers", type=int, default=2)
    p.add_argument("--d-ff", type=int, default=512)
    p.add_argument("--dropout", type=float, default=0.1)
    p.add_argument("--smoothing", type=float, default=0.1)
    p.add_argument("--batch-size", type=int, default=64)
    p.add_argument("--steps", type=int, default=1300)
    p.add_argument("--max-minutes", type=float, default=14.0,
                   help="hard wall-clock budget for the training loop (incl. periodic eval)")
    p.add_argument("--warmup", type=int, default=400)
    p.add_argument("--lr-factor", type=float, default=1.0,
                   help="multiplier on the Eq. 3 learning rate")
    p.add_argument("--eval-every", type=int, default=200)
    p.add_argument("--threads", type=int, default=2)
    p.add_argument("--seed", type=int, default=0)
    return p.parse_args()


# --------------------------------------------------------------------------- evaluation
@torch.no_grad()
def teacher_forced_metrics(model, loader, loss_fn):
    """Loss and per-token accuracy when the decoder is fed the ground-truth prefix."""
    model.eval()
    total_loss, total_tokens, correct = 0.0, 0, 0
    for src, tgt in loader:
        tgt_in, tgt_out = tgt[:, :-1], tgt[:, 1:]
        logits = model(src, tgt_in)
        n = (tgt_out != data.PAD).sum().item()
        total_loss += loss_fn(logits, tgt_out).item() * n
        correct += ((logits.argmax(-1) == tgt_out) & (tgt_out != data.PAD)).sum().item()
        total_tokens += n
    return total_loss / total_tokens, correct / total_tokens


@torch.no_grad()
def greedy_predictions(model, strings, batch_size=256):
    """Autoregressively decode every string; returns the predicted strings."""
    model.eval()
    preds = []
    for i in range(0, len(strings), batch_size):
        src, _ = data.collate([data.ReverseDataset(strings)[j]
                               for j in range(i, min(i + batch_size, len(strings)))])
        out = model.greedy_decode(src, data.BOS, data.EOS, max_len=src.size(1) + 2)
        preds.extend(data.decode(row.tolist()) for row in out)
    return preds


def exact_match(strings, preds):
    return float(np.mean([p == s[::-1] for s, p in zip(strings, preds)]))


# --------------------------------------------------------------------------- plots
def style_axes(ax):
    ax.grid(True, color="#e4e3df", linewidth=0.6)
    ax.set_axisbelow(True)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color("#b5b4ae")
    ax.tick_params(colors=MUTED, labelsize=9)


def plot_curves(history, path):
    fig, axes = plt.subplots(1, 3, figsize=(14, 4))
    ax = axes[0]
    ax.plot(history["train_step"], history["train_loss"], color=BLUE, lw=1.2, label="train (running mean)")
    ax.plot(history["eval_step"], history["val_loss"], color=ORANGE, lw=2, marker="o", ms=4, label="validation")
    ax.set_title("Label-smoothed loss (eps = 0.1)", color=INK, fontsize=11, loc="left")
    ax.set_xlabel("step", color=MUTED)
    ax.legend(frameon=False, fontsize=9)

    ax = axes[1]
    ax.plot(history["eval_step"], [100 * a for a in history["val_token_acc"]], color=BLUE, lw=2,
            marker="o", ms=4, label="token acc. (teacher forcing)")
    ax.plot(history["eval_step"], [100 * a for a in history["val_exact"]], color=ORANGE, lw=2,
            marker="o", ms=4, label="exact match (greedy)")
    ax.set_title("Validation accuracy (%)", color=INK, fontsize=11, loc="left")
    ax.set_xlabel("step", color=MUTED)
    ax.set_ylim(0, 101)
    ax.legend(frameon=False, fontsize=9, loc="lower right")

    ax = axes[2]
    ax.plot(history["lr_step"], history["lr"], color=BLUE, lw=1.5)
    ax.set_title("Learning rate (Eq. 3)", color=INK, fontsize=11, loc="left")
    ax.set_xlabel("step", color=MUTED)

    for ax in axes:
        style_axes(ax)
    fig.tight_layout()
    fig.savefig(path, dpi=130, facecolor="white")
    plt.close(fig)


@torch.no_grad()
def plot_attention(model, text, path):
    """Cross-attention of every decoder layer/head while producing reverse(text)."""
    model.eval()
    src = torch.tensor([data.encode(text)])
    tgt_in = torch.tensor([[data.BOS] + data.encode(text[::-1])])
    model(src, tgt_in)
    out_labels = list(text[::-1]) + ["</s>"]  # query i predicts tgt_out[i]
    num_layers, num_heads = len(model.decoder_layers), model.decoder_layers[0].cross_attn.num_heads

    fig, axes = plt.subplots(num_layers, num_heads, figsize=(3.1 * num_heads, 3.1 * num_layers),
                             squeeze=False)
    for l, layer in enumerate(model.decoder_layers):
        weights = layer.cross_attn.attn_weights[0]  # (heads, tgt_len, src_len)
        for h in range(num_heads):
            ax = axes[l, h]
            ax.imshow(weights[h].numpy(), cmap=SEQ_CMAP, vmin=0, vmax=1)
            ax.set_xticks(range(len(text)), list(text), fontsize=8, color=MUTED)
            ax.set_yticks(range(len(out_labels)), out_labels, fontsize=8, color=MUTED)
            ax.set_title(f"layer {l + 1}, head {h + 1}", fontsize=9, color=INK)
            for spine in ax.spines.values():
                spine.set_visible(False)
    fig.suptitle(f'Decoder cross-attention for "{text}" -> "{text[::-1]}"  '
                 "(rows: output token being predicted, columns: source token)",
                 color=INK, fontsize=11)
    fig.tight_layout()
    fig.savefig(path, dpi=130, facecolor="white")
    plt.close(fig)


# --------------------------------------------------------------------------- main
def main():
    args = parse_args()
    torch.set_num_threads(args.threads)
    torch.manual_seed(args.seed)
    random.seed(args.seed)
    os.makedirs(ASSETS, exist_ok=True)
    os.makedirs(CHECKPOINTS, exist_ok=True)

    splits = data.load_splits()
    train_loader = data.make_loader(splits["train"], args.batch_size, shuffle=True)
    val_loader = data.make_loader(splits["val"], 256, shuffle=False)
    print({k: len(v) for k, v in splits.items()})

    model = Transformer(data.VOCAB_SIZE, args.d_model, args.heads, args.layers, args.d_ff,
                        args.dropout, max_len=64, pad_idx=data.PAD)
    print(f"parameters: {sum(p.numel() for p in model.parameters()):,}")
    loss_fn = LabelSmoothingLoss(data.VOCAB_SIZE, data.PAD, args.smoothing)

    # Adam with beta2 = 0.98, eps = 1e-9 (Section 5.3). The base lr of 1.0 is
    # multiplied by the schedule, so the effective lr is exactly noam_lr(step).
    optimizer = torch.optim.Adam(model.parameters(), lr=1.0, betas=(0.9, 0.98), eps=1e-9)
    scheduler = torch.optim.lr_scheduler.LambdaLR(
        optimizer, lambda s: args.lr_factor * noam_lr(s + 1, args.d_model, args.warmup))

    history = {k: [] for k in ["train_step", "train_loss", "eval_step", "val_loss",
                               "val_token_acc", "val_exact", "lr_step", "lr"]}
    val_subset = splits["val"][:200]
    step, running, start = 0, [], time.time()
    while step < args.steps:
        for src, tgt in train_loader:
            model.train()
            # Teacher forcing: the decoder sees tgt[:-1] and predicts tgt[1:].
            logits = model(src, tgt[:, :-1])
            loss = loss_fn(logits, tgt[:, 1:])
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            history["lr_step"].append(step + 1)
            history["lr"].append(scheduler.get_last_lr()[0])
            scheduler.step()
            step += 1
            running.append(loss.item())

            if step % 50 == 0:
                history["train_step"].append(step)
                history["train_loss"].append(float(np.mean(running[-50:])))
            if step % args.eval_every == 0 or step == args.steps:
                val_loss, val_acc = teacher_forced_metrics(model, val_loader, loss_fn)
                val_exact = exact_match(val_subset, greedy_predictions(model, val_subset))
                for k, v in [("eval_step", step), ("val_loss", val_loss),
                             ("val_token_acc", val_acc), ("val_exact", val_exact)]:
                    history[k].append(v)
                print(f"step {step:5d} | train {np.mean(running[-50:]):.3f} | val {val_loss:.3f} "
                      f"| tok acc {100 * val_acc:5.1f}% | exact {100 * val_exact:5.1f}% "
                      f"| {time.time() - start:5.0f}s")
            if step >= args.steps:
                break
            if time.time() - start > 60 * args.max_minutes:
                print(f"time budget of {args.max_minutes} min reached at step {step}")
                args.steps = step
                break
    train_time = time.time() - start

    torch.save({"model": model.state_dict(), "args": vars(args)},
               os.path.join(CHECKPOINTS, "transformer_reverse.pt"))

    # ------------------------------------------------------------------- final evaluation
    test_loader = data.make_loader(splits["test"], 256, shuffle=False)
    test_loss, test_tok_acc = teacher_forced_metrics(model, test_loader, loss_fn)
    test_preds = greedy_predictions(model, splits["test"])
    long_preds = greedy_predictions(model, splits["test_long"])

    by_length = {}
    for s, p in zip(splits["test"] + splits["test_long"], test_preds + long_preds):
        by_length.setdefault(len(s), []).append(p == s[::-1])
    by_length = {n: float(np.mean(v)) for n, v in sorted(by_length.items())}

    metrics = {
        "train_time_sec": round(train_time, 1),
        "steps": step,
        "parameters": sum(p.numel() for p in model.parameters()),
        "final_train_loss_last50": float(np.mean(running[-50:])),
        "test_loss": test_loss,
        "test_token_acc": test_tok_acc,
        "test_exact_match": exact_match(splits["test"], test_preds),
        "test_long_exact_match": exact_match(splits["test_long"], long_preds),
        "exact_match_by_length": by_length,
        "args": vars(args),
        "history": history,
    }
    with open(os.path.join(ASSETS, "metrics.json"), "w") as f:
        json.dump(metrics, f, indent=1)

    with open(os.path.join(ASSETS, "samples.txt"), "w") as f:
        f.write("Greedy-decoded predictions (input -> prediction, [ok] = exact match)\n\n")
        f.write("Test set (lengths 4-16, same distribution as training):\n")
        rng = random.Random(0)  # fixed random subset, so samples are not alphabetical
        for s, p in rng.sample(list(zip(splits["test"], test_preds)), 15):
            f.write(f"  {s:<16} -> {p:<16} {'[ok]' if p == s[::-1] else '[WRONG, expected ' + s[::-1] + ']'}\n")
        f.write("\nLonger than anything seen in training (lengths 17-24):\n")
        for s, p in rng.sample(list(zip(splits["test_long"], long_preds)), 10):
            f.write(f"  {s:<24} -> {p:<24} {'[ok]' if p == s[::-1] else '[WRONG, expected ' + s[::-1] + ']'}\n")

    plot_curves(history, os.path.join(ASSETS, "loss.png"))
    plot_attention(model, "transformer", os.path.join(ASSETS, "attention.png"))

    print(f"\ntraining time {train_time:.0f}s")
    print(f"test loss {test_loss:.4f} | token acc {100 * test_tok_acc:.2f}% | "
          f"exact match {100 * metrics['test_exact_match']:.2f}% | "
          f"long (17-24) exact match {100 * metrics['test_long_exact_match']:.2f}%")
    print("exact match by length:", {n: round(100 * a, 1) for n, a in by_length.items()})


if __name__ == "__main__":
    main()
