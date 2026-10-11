"""Shape and mask tests for model.py.

Run with `pytest test_model.py`, or without pytest via `python test_model.py`.
"""
import math

import torch

from model import (
    LabelSmoothingLoss,
    MultiHeadAttention,
    PositionalEncoding,
    Transformer,
    make_causal_mask,
    make_padding_mask,
    make_tgt_mask,
    noam_lr,
    scaled_dot_product_attention,
)

torch.set_num_threads(2)

PAD = 0
VOCAB = 20


def small_model():
    torch.manual_seed(0)
    model = Transformer(VOCAB, d_model=32, num_heads=4, num_layers=2, d_ff=64, pad_idx=PAD)
    return model.eval()  # disable dropout so outputs are deterministic


# ---------------------------------------------------------------- shapes
def test_scaled_dot_product_attention_shapes():
    q = torch.randn(2, 4, 5, 8)
    k = torch.randn(2, 4, 7, 8)
    v = torch.randn(2, 4, 7, 16)
    out, weights = scaled_dot_product_attention(q, k, v)
    assert out.shape == (2, 4, 5, 16)
    assert weights.shape == (2, 4, 5, 7)
    assert torch.allclose(weights.sum(-1), torch.ones(2, 4, 5))


def test_attention_matches_equation_1():
    q, k, v = torch.randn(3, 4), torch.randn(5, 4), torch.randn(5, 6)
    expected = torch.softmax(q @ k.T / math.sqrt(4), dim=-1) @ v
    out, _ = scaled_dot_product_attention(q, k, v)
    assert torch.allclose(out, expected, atol=1e-6)


def test_multi_head_attention_shapes():
    mha = MultiHeadAttention(d_model=32, num_heads=4, dropout=0.0)
    x = torch.randn(2, 6, 32)
    memory = torch.randn(2, 9, 32)
    assert mha(x, x, x).shape == (2, 6, 32)
    assert mha(x, memory, memory).shape == (2, 6, 32)  # cross-attention
    assert mha.attn_weights.shape == (2, 4, 6, 9)


def test_transformer_output_shape():
    model = small_model()
    src = torch.randint(3, VOCAB, (2, 7))
    tgt = torch.randint(3, VOCAB, (2, 5))
    assert model(src, tgt).shape == (2, 5, VOCAB)


def test_positional_encoding_values():
    d_model = 16
    pe = PositionalEncoding(d_model, max_len=50, dropout=0.0).pe[0]
    assert pe.shape == (50, d_model)
    pos, i = 7, 3
    angle = pos / 10000 ** (2 * i / d_model)
    assert math.isclose(pe[pos, 2 * i].item(), math.sin(angle), abs_tol=1e-6)
    assert math.isclose(pe[pos, 2 * i + 1].item(), math.cos(angle), abs_tol=1e-6)
    assert torch.all(pe[0, 0::2] == 0) and torch.all(pe[0, 1::2] == 1)


# ---------------------------------------------------------------- masks
def test_padding_mask():
    tokens = torch.tensor([[5, 6, 0, 0], [5, 6, 7, 8]])
    mask = make_padding_mask(tokens, PAD)
    assert mask.shape == (2, 1, 1, 4)
    assert mask[0, 0, 0].tolist() == [True, True, False, False]
    assert mask[1, 0, 0].tolist() == [True, True, True, True]


def test_causal_mask_is_lower_triangular():
    mask = make_causal_mask(4)[0, 0]
    expected = torch.tensor([
        [1, 0, 0, 0],
        [1, 1, 0, 0],
        [1, 1, 1, 0],
        [1, 1, 1, 1],
    ], dtype=torch.bool)
    assert torch.equal(mask, expected)


def test_tgt_mask_combines_padding_and_causal():
    tgt = torch.tensor([[1, 5, 6, 0]])
    mask = make_tgt_mask(tgt, PAD)
    assert mask.shape == (1, 1, 4, 4)
    expected = torch.tensor([
        [1, 0, 0, 0],
        [1, 1, 0, 0],
        [1, 1, 1, 0],
        [1, 1, 1, 0],  # the pad key (column 3) is blocked even for the last query
    ], dtype=torch.bool)
    assert torch.equal(mask[0, 0], expected)


def test_masked_positions_get_exactly_zero_weight():
    q = k = v = torch.randn(1, 1, 4, 8)
    mask = make_causal_mask(4)
    _, weights = scaled_dot_product_attention(q, k, v, mask)
    upper = ~mask[0, 0]
    assert torch.all(weights[0, 0][upper] == 0)
    assert torch.allclose(weights.sum(-1), torch.ones(1, 1, 4))


def test_decoder_is_causal():
    """Changing a future target token must not change earlier outputs."""
    model = small_model()
    src = torch.randint(3, VOCAB, (1, 6))
    tgt = torch.randint(3, VOCAB, (1, 6))
    tgt_changed = tgt.clone()
    tgt_changed[0, 4] = (tgt[0, 4] + 1 - 3) % (VOCAB - 3) + 3  # different non-special token
    with torch.no_grad():
        out = model(src, tgt)
        out_changed = model(src, tgt_changed)
    assert torch.allclose(out[0, :4], out_changed[0, :4], atol=1e-5)
    assert not torch.allclose(out[0, 4:], out_changed[0, 4:], atol=1e-5)


def test_source_padding_does_not_change_output():
    """Appending PAD tokens to the source must leave the decoder output unchanged."""
    model = small_model()
    src = torch.randint(3, VOCAB, (1, 5))
    src_padded = torch.cat([src, torch.full((1, 3), PAD)], dim=1)
    tgt = torch.randint(3, VOCAB, (1, 4))
    with torch.no_grad():
        out = model(src, tgt)
        out_padded = model(src_padded, tgt)
    assert torch.allclose(out, out_padded, atol=1e-5)


def test_cross_attention_ignores_source_padding():
    model = small_model()
    src = torch.tensor([[5, 6, 7, 0, 0]])
    tgt = torch.tensor([[1, 8, 9]])
    with torch.no_grad():
        model(src, tgt)
    for layer in model.decoder_layers:
        weights = layer.cross_attn.attn_weights  # (1, heads, tgt_len, src_len)
        assert torch.all(weights[..., 3:] == 0)


def test_batched_equals_unbatched_with_padding():
    """A short sequence padded inside a batch gives the same logits as on its own."""
    model = small_model()
    src_a, tgt_a = torch.tensor([[4, 5, 6, 7, 8]]), torch.tensor([[1, 9, 10, 11]])
    src_b, tgt_b = torch.tensor([[4, 5]]), torch.tensor([[1, 12]])
    src = torch.tensor([[4, 5, 6, 7, 8], [4, 5, 0, 0, 0]])
    tgt = torch.tensor([[1, 9, 10, 11], [1, 12, 0, 0]])
    with torch.no_grad():
        batched = model(src, tgt)
        alone_a = model(src_a, tgt_a)
        alone_b = model(src_b, tgt_b)
    assert torch.allclose(batched[0], alone_a[0], atol=1e-5)
    assert torch.allclose(batched[1, :2], alone_b[0], atol=1e-5)


# ---------------------------------------------------------------- training utilities
def test_label_smoothing_distribution_and_padding():
    loss_fn = LabelSmoothingLoss(vocab_size=5, pad_idx=PAD, smoothing=0.1)
    logits = torch.randn(1, 3, 5)
    target = torch.tensor([[2, 4, PAD]])
    log_probs = torch.log_softmax(logits, -1)[0]
    expected = []
    for pos in range(2):  # the PAD position is ignored
        dist = torch.full((5,), 0.1 / 3)
        dist[target[0, pos]] = 0.9
        dist[PAD] = 0.0
        expected.append(-(dist * log_probs[pos]).sum())
    assert torch.allclose(loss_fn(logits, target), torch.stack(expected).mean(), atol=1e-6)


def test_label_smoothing_zero_equals_cross_entropy():
    loss_fn = LabelSmoothingLoss(vocab_size=VOCAB, pad_idx=PAD, smoothing=0.0)
    logits = torch.randn(2, 4, VOCAB)
    target = torch.tensor([[3, 4, 5, PAD], [6, 7, PAD, PAD]])
    expected = torch.nn.functional.cross_entropy(
        logits.reshape(-1, VOCAB), target.reshape(-1), ignore_index=PAD
    )
    assert torch.allclose(loss_fn(logits, target), expected, atol=1e-6)


def test_noam_schedule_peaks_at_warmup():
    d_model, warmup = 512, 4000
    lrs = [noam_lr(s, d_model, warmup) for s in range(1, 20_000)]
    peak_step = max(range(len(lrs)), key=lrs.__getitem__) + 1
    assert peak_step == warmup
    assert math.isclose(lrs[warmup - 1], d_model ** -0.5 * warmup ** -0.5)
    assert lrs[99] < lrs[999]          # increasing during warm-up
    assert lrs[9999] < lrs[4999]       # decaying afterwards


def test_greedy_decode_shape():
    model = small_model()
    src = torch.randint(3, VOCAB, (3, 6))
    out = model.greedy_decode(src, bos_idx=1, eos_idx=2, max_len=8)
    assert out.shape[0] == 3 and out.shape[1] <= 8


if __name__ == "__main__":
    tests = [(name, fn) for name, fn in globals().items() if name.startswith("test_")]
    for name, fn in tests:
        fn()
        print(f"PASSED {name}")
    print(f"{len(tests)} passed")
