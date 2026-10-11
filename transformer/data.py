"""Toy seq2seq task: reverse a string of lowercase letters.

    source: "transformer"  ->  target: "remrofsnart"

Splits are generated once with a fixed seed and cached in data/. Test and
validation strings never appear in the training set.
"""
import json
import os
import random

import torch
from torch.utils.data import DataLoader, Dataset

PAD, BOS, EOS = 0, 1, 2
SPECIALS = ["<pad>", "<s>", "</s>"]
ALPHABET = "abcdefghijklmnopqrstuvwxyz"
ITOS = SPECIALS + list(ALPHABET)
STOI = {s: i for i, s in enumerate(ITOS)}
VOCAB_SIZE = len(ITOS)

DATA_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data")


def encode(text):
    return [STOI[c] for c in text]


def decode(ids):
    """Token ids -> string, stopping at EOS and skipping specials."""
    chars = []
    for i in ids:
        if i == EOS:
            break
        if i >= len(SPECIALS):
            chars.append(ITOS[i])
    return "".join(chars)


def _random_strings(rng, n, min_len, max_len, exclude=frozenset()):
    out = set()
    while len(out) < n:
        length = rng.randint(min_len, max_len)
        s = "".join(rng.choice(ALPHABET) for _ in range(length))
        if s not in exclude:
            out.add(s)
    return sorted(out)


def build_splits(n_train=50_000, n_val=1_000, n_test=2_000, min_len=4, max_len=16,
                 n_long=500, long_min=17, long_max=24, seed=0):
    """Return {"train", "val", "test", "test_long"} lists of strings.

    "test_long" holds strings longer than anything seen in training, to probe
    length generalisation.
    """
    rng = random.Random(seed)
    test = _random_strings(rng, n_test, min_len, max_len)
    val = _random_strings(rng, n_val, min_len, max_len, exclude=set(test))
    train = _random_strings(rng, n_train, min_len, max_len, exclude=set(test) | set(val))
    test_long = _random_strings(rng, n_long, long_min, long_max)
    return {"train": train, "val": val, "test": test, "test_long": test_long}


def load_splits(**kwargs):
    """Load cached splits from data/, generating them on first use."""
    path = os.path.join(DATA_DIR, "reverse_splits.json")
    if os.path.exists(path):
        with open(path) as f:
            return json.load(f)
    splits = build_splits(**kwargs)
    os.makedirs(DATA_DIR, exist_ok=True)
    with open(path, "w") as f:
        json.dump(splits, f)
    return splits


class ReverseDataset(Dataset):
    def __init__(self, strings):
        self.strings = strings

    def __len__(self):
        return len(self.strings)

    def __getitem__(self, idx):
        s = self.strings[idx]
        src = encode(s)
        tgt = [BOS] + encode(s[::-1]) + [EOS]
        return torch.tensor(src), torch.tensor(tgt)


def collate(batch):
    """Right-pad a list of (src, tgt) pairs with PAD into two (batch, max_len) tensors."""
    srcs, tgts = zip(*batch)
    src = torch.nn.utils.rnn.pad_sequence(srcs, batch_first=True, padding_value=PAD)
    tgt = torch.nn.utils.rnn.pad_sequence(tgts, batch_first=True, padding_value=PAD)
    return src, tgt


def make_loader(strings, batch_size, shuffle):
    return DataLoader(ReverseDataset(strings), batch_size=batch_size,
                      shuffle=shuffle, collate_fn=collate)
