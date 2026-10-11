"""Small U-Net noise predictor eps_theta(x_t, t).

The paper (Sec. 4, Appendix B) uses a PixelCNN++-style U-Net with ~35.7M
parameters, two residual blocks per resolution, group normalisation, self-attention
at 16x16 and a Transformer sinusoidal timestep embedding added to every residual
block. This is the same design, shrunk to train on a CPU in minutes:
three resolutions (28 -> 14 -> 7), one residual block per level and
self-attention only at the 7x7 bottleneck.
"""

import math

import torch
import torch.nn as nn
import torch.nn.functional as F


class SinusoidalTimeEmbedding(nn.Module):
    """Transformer positional encoding (Vaswani et al., 2017) applied to the timestep t.

    emb[2i] = sin(t / 10000^(2i/dim)), emb[2i+1] = cos(t / 10000^(2i/dim)),
    written as [sin(...), cos(...)] halves, which is equivalent up to a permutation.
    """

    def __init__(self, dim):
        super().__init__()
        self.dim = dim

    def forward(self, t):
        half = self.dim // 2
        freqs = torch.exp(-math.log(10000) * torch.arange(half, device=t.device) / half)
        args = t.float()[:, None] * freqs[None, :]
        return torch.cat([args.sin(), args.cos()], dim=1)


class ResBlock(nn.Module):
    """GroupNorm -> SiLU -> Conv, with the time embedding added between the two convolutions."""

    def __init__(self, in_ch, out_ch, time_dim, groups=8):
        super().__init__()
        self.norm1 = nn.GroupNorm(groups, in_ch)
        self.conv1 = nn.Conv2d(in_ch, out_ch, 3, padding=1)
        self.time_proj = nn.Linear(time_dim, out_ch)
        self.norm2 = nn.GroupNorm(groups, out_ch)
        self.conv2 = nn.Conv2d(out_ch, out_ch, 3, padding=1)
        self.skip = nn.Conv2d(in_ch, out_ch, 1) if in_ch != out_ch else nn.Identity()

    def forward(self, x, t_emb):
        h = self.conv1(F.silu(self.norm1(x)))
        h = h + self.time_proj(F.silu(t_emb))[:, :, None, None]
        h = self.conv2(F.silu(self.norm2(h)))
        return h + self.skip(x)


class SelfAttention(nn.Module):
    """Single-head self-attention over spatial positions, with a residual connection."""

    def __init__(self, ch, groups=8):
        super().__init__()
        self.norm = nn.GroupNorm(groups, ch)
        self.qkv = nn.Conv2d(ch, 3 * ch, 1)
        self.out = nn.Conv2d(ch, ch, 1)

    def forward(self, x):
        b, c, h, w = x.shape
        q, k, v = self.qkv(self.norm(x)).reshape(b, 3, c, h * w).unbind(dim=1)
        attn = torch.softmax(q.transpose(1, 2) @ k / math.sqrt(c), dim=-1)  # (b, hw, hw)
        out = (v @ attn.transpose(1, 2)).reshape(b, c, h, w)
        return x + self.out(out)


class UNet(nn.Module):
    """U-Net mapping (x_t, t) to a prediction of the noise eps with the same shape as x_t.

    Shapes for a 28x28 input with base_ch=32:
        28x28x32 -> down -> 14x14x64 -> down -> 7x7x128 (attention) -> up -> 14x14x64 -> up -> 28x28x32
    with skip connections concatenated at each resolution.
    """

    def __init__(self, in_ch=1, base_ch=32, time_dim=128):
        super().__init__()
        c1, c2, c3 = base_ch, base_ch * 2, base_ch * 4

        self.time_embed = nn.Sequential(
            SinusoidalTimeEmbedding(base_ch),
            nn.Linear(base_ch, time_dim),
            nn.SiLU(),
            nn.Linear(time_dim, time_dim),
        )

        self.stem = nn.Conv2d(in_ch, c1, 3, padding=1)

        self.enc1 = ResBlock(c1, c1, time_dim)
        self.down1 = nn.Conv2d(c1, c1, 3, stride=2, padding=1)  # 28 -> 14
        self.enc2 = ResBlock(c1, c2, time_dim)
        self.down2 = nn.Conv2d(c2, c2, 3, stride=2, padding=1)  # 14 -> 7

        self.mid1 = ResBlock(c2, c3, time_dim)
        self.mid_attn = SelfAttention(c3)
        self.mid2 = ResBlock(c3, c3, time_dim)

        self.up2 = nn.Sequential(nn.Upsample(scale_factor=2, mode="nearest"), nn.Conv2d(c3, c2, 3, padding=1))
        self.dec2 = ResBlock(c2 + c2, c2, time_dim)
        self.up1 = nn.Sequential(nn.Upsample(scale_factor=2, mode="nearest"), nn.Conv2d(c2, c1, 3, padding=1))
        self.dec1 = ResBlock(c1 + c1, c1, time_dim)

        self.head = nn.Sequential(nn.GroupNorm(8, c1), nn.SiLU(), nn.Conv2d(c1, in_ch, 3, padding=1))

    def forward(self, x, t):
        t_emb = self.time_embed(t)

        h = self.stem(x)
        s1 = self.enc1(h, t_emb)                      # 28x28
        s2 = self.enc2(self.down1(s1), t_emb)         # 14x14

        h = self.mid1(self.down2(s2), t_emb)          # 7x7
        h = self.mid2(self.mid_attn(h), t_emb)

        h = self.dec2(torch.cat([self.up2(h), s2], dim=1), t_emb)  # 14x14
        h = self.dec1(torch.cat([self.up1(h), s1], dim=1), t_emb)  # 28x28
        return self.head(h)
