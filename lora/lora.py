"""LoRA: Low-Rank Adaptation of Large Language Models (Hu et al., 2021).

Paper: https://arxiv.org/abs/2106.09685

A pretrained weight W0 (d x k) is frozen and its update is constrained to a
low-rank product, Delta W = B A with B (d x r), A (r x k), r << min(d, k).
The forward pass becomes (Eq. 3):

    h = W0 x + Delta W x = W0 x + (alpha / r) * B A x
"""

import math

import torch
import torch.nn as nn


class LoRALinear(nn.Module):
    """Wraps a frozen nn.Linear and adds a trainable low-rank update B A.

    Shapes follow PyTorch's nn.Linear convention, where weight is (out, in):
        W0: (out_features, in_features)   frozen
        A:  (r, in_features)              Gaussian init
        B:  (out_features, r)             zero init, so B A = 0 at the start

    merge() folds the update into W0 for zero extra inference latency
    (Section 4.1); unmerge() subtracts it again.
    """

    def __init__(self, base: nn.Linear, r: int, alpha: float = 1.0, dropout: float = 0.0):
        super().__init__()
        if r <= 0:
            raise ValueError(f"rank r must be positive, got {r}")
        self.base = base
        self.r = r
        self.alpha = alpha
        self.scaling = alpha / r
        self.dropout = nn.Dropout(dropout) if dropout > 0 else nn.Identity()
        self.merged = False

        for p in self.base.parameters():
            p.requires_grad = False

        self.lora_A = nn.Parameter(torch.empty(r, base.in_features))
        self.lora_B = nn.Parameter(torch.zeros(base.out_features, r))
        nn.init.normal_(self.lora_A, mean=0.0, std=1.0 / math.sqrt(base.in_features))

    @property
    def in_features(self) -> int:
        return self.base.in_features

    @property
    def out_features(self) -> int:
        return self.base.out_features

    def delta_weight(self) -> torch.Tensor:
        """Delta W = (alpha / r) * B A, shape (out_features, in_features)."""
        return self.scaling * (self.lora_B @ self.lora_A)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.merged:
            return self.base(x)
        # Eq. 3. (x A^T) B^T avoids materialising the d x k update.
        lora_out = self.dropout(x) @ self.lora_A.T @ self.lora_B.T
        return self.base(x) + self.scaling * lora_out

    @torch.no_grad()
    def merge(self) -> None:
        """W <- W0 + (alpha / r) B A. Afterwards forward() is one plain matmul."""
        if not self.merged:
            self.base.weight += self.delta_weight()
            self.merged = True

    @torch.no_grad()
    def unmerge(self) -> None:
        """W <- W - (alpha / r) B A, recovering W0 (up to float rounding)."""
        if self.merged:
            self.base.weight -= self.delta_weight()
            self.merged = False

    def extra_repr(self) -> str:
        return (f"in_features={self.in_features}, out_features={self.out_features}, "
                f"r={self.r}, alpha={self.alpha}, merged={self.merged}")


def _matches(name: str, targets: list[str]) -> bool:
    """A module matches a target if its full dotted name equals it or ends with '.target'."""
    return any(name == t or name.endswith("." + t) for t in targets)


def inject_lora(model: nn.Module, target_modules: list[str], r: int,
                alpha: float = 1.0, dropout: float = 0.0) -> nn.Module:
    """Replace each nn.Linear whose name matches target_modules with a LoRALinear.

    Afterwards every parameter of the model is frozen except the LoRA A and B
    matrices, mirroring the paper's setup where only Delta W is trained.
    Modifies the model in place and also returns it.
    """
    to_replace = [(name, module) for name, module in model.named_modules()
                  if isinstance(module, nn.Linear) and _matches(name, target_modules)]
    if not to_replace:
        raise ValueError(f"no nn.Linear modules matched {target_modules}")

    for name, module in to_replace:
        parent_name, _, child_name = name.rpartition(".")
        parent = model.get_submodule(parent_name) if parent_name else model
        setattr(parent, child_name, LoRALinear(module, r=r, alpha=alpha, dropout=dropout))

    mark_only_lora_as_trainable(model)
    return model


def mark_only_lora_as_trainable(model: nn.Module) -> None:
    for name, p in model.named_parameters():
        p.requires_grad = "lora_" in name


def lora_state_dict(model: nn.Module) -> dict[str, torch.Tensor]:
    """Only the LoRA tensors: the small per-task file you would store or ship."""
    return {k: v for k, v in model.state_dict().items() if "lora_" in k}


def merge_lora(model: nn.Module) -> None:
    for m in model.modules():
        if isinstance(m, LoRALinear):
            m.merge()


def unmerge_lora(model: nn.Module) -> None:
    for m in model.modules():
        if isinstance(m, LoRALinear):
            m.unmerge()


def count_parameters(model: nn.Module, trainable_only: bool = False) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad or not trainable_only)
