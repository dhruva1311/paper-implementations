"""Vision Transformer (ViT) from scratch.

Dosovitskiy et al., "An Image is Worth 16x16 Words: Transformers for Image
Recognition at Scale", ICLR 2021. https://arxiv.org/abs/2010.11929

Section 3.1 of the paper defines the model with four equations:

    z_0  = [x_class; x_p^1 E; x_p^2 E; ...; x_p^N E] + E_pos       (Eq. 1)
    z'_l = MSA(LN(z_{l-1})) + z_{l-1},          l = 1..L          (Eq. 2)
    z_l  = MLP(LN(z'_l)) + z'_l,                l = 1..L          (Eq. 3)
    y    = LN(z_L^0)                                              (Eq. 4)

Shapes used throughout: B = batch, N = number of patches, T = N + 1 tokens,
D = embedding dim, H = number of heads, d = D / H (per-head dim).
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class PatchEmbedding(nn.Module):
    """Split the image into P x P patches and project each to D dims (x_p^i E in Eq. 1).

    A Conv2d with kernel_size = stride = P is the same as a linear layer on
    each flattened patch.
    """

    def __init__(self, image_size: int, patch_size: int, in_channels: int, dim: int):
        super().__init__()
        assert image_size % patch_size == 0, "image size must be divisible by patch size"
        self.num_patches = (image_size // patch_size) ** 2
        self.proj = nn.Conv2d(in_channels, dim, kernel_size=patch_size, stride=patch_size)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.proj(x)                  # (B, D, H/P, W/P)
        x = x.flatten(2)                  # (B, D, N)
        return x.transpose(1, 2)          # (B, N, D)


class MultiHeadSelfAttention(nn.Module):
    """Multi-head self-attention (MSA in Eq. 2; Appendix A, Eqs. 5-8 of the paper).

    For each head: [q, k, v] = z U_qkv,  A = softmax(q k^T / sqrt(d)),  SA(z) = A v.
    MSA(z) = [SA_1(z); ...; SA_H(z)] U_msa.
    """

    def __init__(self, dim: int, num_heads: int, dropout: float = 0.0):
        super().__init__()
        assert dim % num_heads == 0, "dim must be divisible by num_heads"
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.scale = self.head_dim ** -0.5
        self.qkv = nn.Linear(dim, 3 * dim)        # U_qkv for all heads at once
        self.proj = nn.Linear(dim, dim)           # U_msa
        self.attn_drop = nn.Dropout(dropout)
        self.proj_drop = nn.Dropout(dropout)
        self.last_attn = None                     # (B, H, T, T), kept for visualization

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, T, D = x.shape
        qkv = self.qkv(x).reshape(B, T, 3, self.num_heads, self.head_dim)
        qkv = qkv.permute(2, 0, 3, 1, 4)          # (3, B, H, T, d)
        q, k, v = qkv[0], qkv[1], qkv[2]

        attn = (q @ k.transpose(-2, -1)) * self.scale   # (B, H, T, T)
        attn = attn.softmax(dim=-1)
        self.last_attn = attn.detach()
        attn = self.attn_drop(attn)

        out = attn @ v                            # (B, H, T, d)
        out = out.transpose(1, 2).reshape(B, T, D)  # concatenate heads
        return self.proj_drop(self.proj(out))


class MLP(nn.Module):
    """Two-layer MLP with GELU used inside each encoder block (Eq. 3)."""

    def __init__(self, dim: int, hidden_dim: int, dropout: float = 0.0):
        super().__init__()
        self.fc1 = nn.Linear(dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, dim)
        self.drop = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.drop(F.gelu(self.fc1(x)))
        return self.drop(self.fc2(x))


class EncoderBlock(nn.Module):
    """Pre-norm Transformer encoder block: Eq. 2 followed by Eq. 3."""

    def __init__(self, dim: int, num_heads: int, mlp_dim: int, dropout: float = 0.0):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim)
        self.attn = MultiHeadSelfAttention(dim, num_heads, dropout)
        self.norm2 = nn.LayerNorm(dim)
        self.mlp = MLP(dim, mlp_dim, dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.attn(self.norm1(x))   # Eq. 2
        x = x + self.mlp(self.norm2(x))    # Eq. 3
        return x


class ViT(nn.Module):
    """Vision Transformer classifier.

    Builds z_0 (Eq. 1), applies L encoder blocks (Eqs. 2-3), takes the final
    [CLS] token through LayerNorm (Eq. 4), and classifies it with a linear head.
    """

    def __init__(
        self,
        image_size: int = 28,
        patch_size: int = 4,
        in_channels: int = 1,
        num_classes: int = 10,
        dim: int = 64,
        depth: int = 6,
        num_heads: int = 4,
        mlp_dim: int = 128,
        dropout: float = 0.0,
    ):
        super().__init__()
        self.patch_embed = PatchEmbedding(image_size, patch_size, in_channels, dim)
        num_patches = self.patch_embed.num_patches

        self.cls_token = nn.Parameter(torch.zeros(1, 1, dim))                 # x_class
        self.pos_embed = nn.Parameter(torch.zeros(1, num_patches + 1, dim))   # E_pos
        self.pos_drop = nn.Dropout(dropout)

        self.blocks = nn.ModuleList(
            [EncoderBlock(dim, num_heads, mlp_dim, dropout) for _ in range(depth)]
        )
        self.norm = nn.LayerNorm(dim)
        self.head = nn.Linear(dim, num_classes)

        nn.init.trunc_normal_(self.pos_embed, std=0.02)
        nn.init.trunc_normal_(self.cls_token, std=0.02)
        self.apply(self._init_weights)

    @staticmethod
    def _init_weights(m: nn.Module) -> None:
        if isinstance(m, nn.Linear):
            nn.init.trunc_normal_(m.weight, std=0.02)
            if m.bias is not None:
                nn.init.zeros_(m.bias)
        elif isinstance(m, nn.LayerNorm):
            nn.init.ones_(m.weight)
            nn.init.zeros_(m.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.patch_embed(x)                                   # (B, N, D)
        cls = self.cls_token.expand(x.shape[0], -1, -1)           # (B, 1, D)
        x = torch.cat([cls, x], dim=1) + self.pos_embed           # Eq. 1, (B, T, D)
        x = self.pos_drop(x)

        for block in self.blocks:                                 # Eqs. 2-3
            x = block(x)

        y = self.norm(x[:, 0])                                    # Eq. 4
        return self.head(y)                                       # (B, num_classes)

    def attention_maps(self) -> list[torch.Tensor]:
        """Attention weights from the most recent forward pass, one (B, H, T, T) per layer."""
        return [block.attn.last_attn for block in self.blocks]
