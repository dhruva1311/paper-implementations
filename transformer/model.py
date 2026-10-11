"""A from-scratch Transformer (Vaswani et al., 2017, "Attention Is All You Need").

Paper: https://arxiv.org/abs/1706.03762

Masks are boolean, True = "query may attend to this key". They broadcast
against attention scores of shape (batch, heads, query_len, key_len).
"""
import math

import torch
import torch.nn as nn
import torch.nn.functional as F


# --------------------------------------------------------------------------
# Attention (Section 3.2)
# --------------------------------------------------------------------------
def scaled_dot_product_attention(q, k, v, mask=None, dropout=None):
    """Attention(Q, K, V) = softmax(Q K^T / sqrt(d_k)) V   (Section 3.2.1, Eq. 1).

    Args:
        q: (..., len_q, d_k)
        k: (..., len_k, d_k)
        v: (..., len_k, d_v)
        mask: bool, broadcastable to (..., len_q, len_k); False positions are blocked.
        dropout: optional nn.Dropout applied to the attention weights.

    Returns:
        output (..., len_q, d_v) and attention weights (..., len_q, len_k).
    """
    d_k = q.size(-1)
    # Dividing by sqrt(d_k) stops the dot products from growing with d_k,
    # which would push softmax into regions with tiny gradients (footnote 4).
    scores = q @ k.transpose(-2, -1) / math.sqrt(d_k)
    if mask is not None:
        # -inf gives exactly zero weight after softmax (Section 3.2.3).
        scores = scores.masked_fill(~mask, float("-inf"))
    weights = F.softmax(scores, dim=-1)
    if dropout is not None:
        weights = dropout(weights)
    return weights @ v, weights


class MultiHeadAttention(nn.Module):
    """MultiHead(Q, K, V) = Concat(head_1, ..., head_h) W^O,
    head_i = Attention(Q W_i^Q, K W_i^K, V W_i^V)   (Section 3.2.2).

    The h per-head projections are packed into one d_model -> h * d_k linear
    layer and reshaped into heads, equivalent to h separate projections.
    """

    def __init__(self, d_model, num_heads, dropout=0.1):
        super().__init__()
        assert d_model % num_heads == 0, "d_model must be divisible by num_heads"
        self.num_heads = num_heads
        self.d_k = d_model // num_heads
        self.w_q = nn.Linear(d_model, d_model)
        self.w_k = nn.Linear(d_model, d_model)
        self.w_v = nn.Linear(d_model, d_model)
        self.w_o = nn.Linear(d_model, d_model)
        self.dropout = nn.Dropout(dropout)
        self.attn_weights = None  # last attention map, kept for visualisation

    def _split_heads(self, x):
        # (batch, len, d_model) -> (batch, heads, len, d_k)
        batch, length, _ = x.shape
        return x.view(batch, length, self.num_heads, self.d_k).transpose(1, 2)

    def forward(self, query, key, value, mask=None):
        q = self._split_heads(self.w_q(query))
        k = self._split_heads(self.w_k(key))
        v = self._split_heads(self.w_v(value))

        out, weights = scaled_dot_product_attention(q, k, v, mask, self.dropout)
        self.attn_weights = weights.detach()

        # (batch, heads, len, d_k) -> (batch, len, d_model): the "Concat" step.
        batch, _, length, _ = out.shape
        out = out.transpose(1, 2).contiguous().view(batch, length, -1)
        return self.w_o(out)


# --------------------------------------------------------------------------
# Feed-forward, embeddings, positional encoding (Sections 3.3 - 3.5)
# --------------------------------------------------------------------------
class PositionwiseFeedForward(nn.Module):
    """FFN(x) = max(0, x W_1 + b_1) W_2 + b_2   (Section 3.3, Eq. 2)."""

    def __init__(self, d_model, d_ff, dropout=0.1):
        super().__init__()
        self.linear1 = nn.Linear(d_model, d_ff)
        self.linear2 = nn.Linear(d_ff, d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        return self.linear2(self.dropout(F.relu(self.linear1(x))))


class PositionalEncoding(nn.Module):
    """Sinusoidal positional encoding (Section 3.5):

        PE(pos, 2i)   = sin(pos / 10000^(2i / d_model))
        PE(pos, 2i+1) = cos(pos / 10000^(2i / d_model))

    The table is fixed (not learned) and added to the token embeddings,
    followed by dropout (Section 5.4).
    """

    def __init__(self, d_model, max_len=512, dropout=0.1):
        super().__init__()
        self.dropout = nn.Dropout(dropout)
        position = torch.arange(max_len).unsqueeze(1).float()   # (max_len, 1)
        two_i = torch.arange(0, d_model, 2).float()              # (d_model / 2,)
        div_term = torch.pow(10000.0, two_i / d_model)           # 10000^(2i / d_model)
        pe = torch.zeros(max_len, d_model)
        pe[:, 0::2] = torch.sin(position / div_term)
        pe[:, 1::2] = torch.cos(position / div_term)
        self.register_buffer("pe", pe.unsqueeze(0))              # (1, max_len, d_model)

    def forward(self, x):
        return self.dropout(x + self.pe[:, : x.size(1)])


class Embeddings(nn.Module):
    """Token embedding multiplied by sqrt(d_model) (Section 3.4)."""

    def __init__(self, vocab_size, d_model):
        super().__init__()
        self.embed = nn.Embedding(vocab_size, d_model)
        self.d_model = d_model

    def forward(self, tokens):
        return self.embed(tokens) * math.sqrt(self.d_model)


# --------------------------------------------------------------------------
# Encoder and decoder layers (Section 3.1)
# --------------------------------------------------------------------------
class EncoderLayer(nn.Module):
    """Self-attention then feed-forward, each wrapped as
    LayerNorm(x + Dropout(Sublayer(x)))  (Section 3.1; residual dropout, Section 5.4).
    This is the paper's original "post-norm" arrangement.
    """

    def __init__(self, d_model, num_heads, d_ff, dropout=0.1):
        super().__init__()
        self.self_attn = MultiHeadAttention(d_model, num_heads, dropout)
        self.ffn = PositionwiseFeedForward(d_model, d_ff, dropout)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, src_mask):
        x = self.norm1(x + self.dropout(self.self_attn(x, x, x, src_mask)))
        x = self.norm2(x + self.dropout(self.ffn(x)))
        return x


class DecoderLayer(nn.Module):
    """Masked self-attention, encoder-decoder (cross) attention, then
    feed-forward, each with residual + LayerNorm (Section 3.1).

    In cross-attention the queries come from the decoder and the keys/values
    from the encoder output (Section 3.2.3).
    """

    def __init__(self, d_model, num_heads, d_ff, dropout=0.1):
        super().__init__()
        self.self_attn = MultiHeadAttention(d_model, num_heads, dropout)
        self.cross_attn = MultiHeadAttention(d_model, num_heads, dropout)
        self.ffn = PositionwiseFeedForward(d_model, d_ff, dropout)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.norm3 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, memory, src_mask, tgt_mask):
        x = self.norm1(x + self.dropout(self.self_attn(x, x, x, tgt_mask)))
        x = self.norm2(x + self.dropout(self.cross_attn(x, memory, memory, src_mask)))
        x = self.norm3(x + self.dropout(self.ffn(x)))
        return x


# --------------------------------------------------------------------------
# Masks (Section 3.2.3)
# --------------------------------------------------------------------------
def make_padding_mask(tokens, pad_idx):
    """(batch, len) -> (batch, 1, 1, len). Blocks attention *to* padding keys."""
    return (tokens != pad_idx).unsqueeze(1).unsqueeze(2)


def make_causal_mask(length, device=None):
    """(1, 1, len, len) lower-triangular mask: position i may attend to j <= i.
    This keeps the decoder auto-regressive (Section 3.1, "Decoder")."""
    mask = torch.tril(torch.ones(length, length, dtype=torch.bool, device=device))
    return mask.view(1, 1, length, length)


def make_tgt_mask(tgt, pad_idx):
    """Padding mask AND causal mask -> (batch, 1, len, len)."""
    return make_padding_mask(tgt, pad_idx) & make_causal_mask(tgt.size(1), tgt.device)


# --------------------------------------------------------------------------
# Full model (Figure 1)
# --------------------------------------------------------------------------
class Transformer(nn.Module):
    """Encoder-decoder Transformer (Figure 1).

    Source and target share one vocabulary here, so, as in Section 3.4, the
    source embedding, target embedding and pre-softmax linear layer all share
    one weight matrix.
    """

    def __init__(
        self,
        vocab_size,
        d_model=512,
        num_heads=8,
        num_layers=6,
        d_ff=2048,
        dropout=0.1,
        max_len=512,
        pad_idx=0,
    ):
        super().__init__()
        self.pad_idx = pad_idx
        self.embed = Embeddings(vocab_size, d_model)
        self.pos_enc = PositionalEncoding(d_model, max_len, dropout)
        self.encoder_layers = nn.ModuleList(
            [EncoderLayer(d_model, num_heads, d_ff, dropout) for _ in range(num_layers)]
        )
        self.decoder_layers = nn.ModuleList(
            [DecoderLayer(d_model, num_heads, d_ff, dropout) for _ in range(num_layers)]
        )
        self.generator = nn.Linear(d_model, vocab_size)
        self.generator.weight = self.embed.embed.weight  # weight tying (Section 3.4)
        self._reset_parameters()

    def _reset_parameters(self):
        # The paper does not specify an initialisation; Xavier-uniform on all
        # weight matrices is the common choice for Transformers.
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    def encode(self, src, src_mask):
        x = self.pos_enc(self.embed(src))
        for layer in self.encoder_layers:
            x = layer(x, src_mask)
        return x

    def decode(self, tgt, memory, src_mask, tgt_mask):
        x = self.pos_enc(self.embed(tgt))
        for layer in self.decoder_layers:
            x = layer(x, memory, src_mask, tgt_mask)
        return x

    def forward(self, src, tgt):
        """src: (batch, src_len), tgt: (batch, tgt_len) -> logits (batch, tgt_len, vocab)."""
        src_mask = make_padding_mask(src, self.pad_idx)
        tgt_mask = make_tgt_mask(tgt, self.pad_idx)
        memory = self.encode(src, src_mask)
        return self.generator(self.decode(tgt, memory, src_mask, tgt_mask))

    @torch.no_grad()
    def greedy_decode(self, src, bos_idx, eos_idx, max_len):
        """Greedy decoding (the paper uses beam size 4, Section 6.1).
        Returns (batch, generated_len) without BOS."""
        src_mask = make_padding_mask(src, self.pad_idx)
        memory = self.encode(src, src_mask)
        ys = torch.full((src.size(0), 1), bos_idx, dtype=torch.long, device=src.device)
        finished = torch.zeros(src.size(0), dtype=torch.bool, device=src.device)
        for _ in range(max_len):
            out = self.decode(ys, memory, src_mask, make_tgt_mask(ys, self.pad_idx))
            next_token = self.generator(out[:, -1]).argmax(dim=-1)
            # Sequences that already produced EOS keep emitting padding.
            next_token = next_token.masked_fill(finished, self.pad_idx)
            ys = torch.cat([ys, next_token.unsqueeze(1)], dim=1)
            finished |= next_token == eos_idx
            if finished.all():
                break
        return ys[:, 1:]


# --------------------------------------------------------------------------
# Training utilities (Sections 5.3 and 5.4)
# --------------------------------------------------------------------------
class LabelSmoothingLoss(nn.Module):
    """Cross-entropy against a smoothed target distribution (Section 5.4, eps_ls = 0.1).

    The true class gets probability 1 - eps and the remaining eps is spread
    uniformly over the other classes, excluding PAD (never a valid output).
    Padding positions in the target are ignored. Returns the mean loss per
    non-pad token.
    """

    def __init__(self, vocab_size, pad_idx, smoothing=0.1):
        super().__init__()
        self.vocab_size = vocab_size
        self.pad_idx = pad_idx
        self.smoothing = smoothing

    def forward(self, logits, target):
        logits = logits.reshape(-1, self.vocab_size)
        target = target.reshape(-1)
        log_probs = F.log_softmax(logits, dim=-1)

        # V - 2 classes share eps: everything except the true class and PAD.
        true_dist = torch.full_like(log_probs, self.smoothing / (self.vocab_size - 2))
        true_dist.scatter_(1, target.unsqueeze(1), 1.0 - self.smoothing)
        true_dist[:, self.pad_idx] = 0.0

        loss = -(true_dist * log_probs).sum(dim=-1)
        return loss[target != self.pad_idx].mean()


def noam_lr(step, d_model, warmup_steps):
    """lrate = d_model^-0.5 * min(step^-0.5, step * warmup_steps^-1.5)   (Section 5.3, Eq. 3).

    Linear warm-up for `warmup_steps` steps, then decay proportional to
    1 / sqrt(step). `step` is 1-indexed.
    """
    step = max(step, 1)
    return d_model ** -0.5 * min(step ** -0.5, step * warmup_steps ** -1.5)
