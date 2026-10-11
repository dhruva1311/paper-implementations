"""Tests for lora.py. Run with `python -m pytest test_lora.py` or `python test_lora.py`."""

import copy

import torch
import torch.nn as nn

from lora import (LoRALinear, count_parameters, inject_lora, lora_state_dict,
                  merge_lora, unmerge_lora)

torch.set_num_threads(2)


class TinyNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(20, 32)
        self.block = nn.Sequential(nn.ReLU(), nn.Linear(32, 16), nn.ReLU())
        self.head = nn.Linear(16, 5)
        self.norm = nn.LayerNorm(5)

    def forward(self, x):
        return self.norm(self.head(self.block(self.fc1(x))))


def _randomize_B(model: nn.Module) -> None:
    """B starts at zero; give it values so merge tests are not trivially true."""
    with torch.no_grad():
        for name, p in model.named_parameters():
            if "lora_B" in name:
                p.normal_(std=0.1)


def test_output_equals_base_at_init():
    torch.manual_seed(0)
    base = nn.Linear(20, 10)
    layer = LoRALinear(copy.deepcopy(base), r=4, alpha=8)
    x = torch.randn(7, 20)
    assert torch.all(layer.lora_B == 0)
    assert torch.equal(layer(x), base(x))


def test_model_output_unchanged_after_injection():
    torch.manual_seed(0)
    model = TinyNet()
    x = torch.randn(7, 20)
    expected = model(x)
    inject_lora(model, ["fc1", "block.1", "head"], r=2, alpha=4)
    assert torch.equal(model(x), expected)


def test_merged_equals_unmerged():
    torch.manual_seed(0)
    layer = LoRALinear(nn.Linear(20, 10), r=4, alpha=8, dropout=0.1)
    _randomize_B(layer)
    layer.eval()  # dropout is only applied in the unmerged training path
    x = torch.randn(7, 20)
    unmerged = layer(x)
    layer.merge()
    assert layer.merged
    assert torch.allclose(layer(x), unmerged, atol=1e-6)


def test_unmerge_restores_base_weight():
    torch.manual_seed(0)
    layer = LoRALinear(nn.Linear(20, 10), r=4, alpha=8)
    _randomize_B(layer)
    w0 = layer.base.weight.clone()
    layer.merge()
    assert not torch.allclose(layer.base.weight, w0)
    layer.unmerge()
    assert torch.allclose(layer.base.weight, w0, atol=1e-6)


def test_model_merge_matches_unmerged():
    torch.manual_seed(0)
    model = inject_lora(TinyNet(), ["fc1", "block.1", "head"], r=3, alpha=6).eval()
    _randomize_B(model)
    x = torch.randn(7, 20)
    unmerged = model(x)
    merge_lora(model)
    assert torch.allclose(model(x), unmerged, atol=1e-5)
    unmerge_lora(model)
    assert torch.allclose(model(x), unmerged, atol=1e-5)


def test_only_lora_params_trainable():
    model = inject_lora(TinyNet(), ["fc1", "head"], r=2)
    trainable = {n for n, p in model.named_parameters() if p.requires_grad}
    assert trainable == {"fc1.lora_A", "fc1.lora_B", "head.lora_A", "head.lora_B"}
    # Non-targeted Linear and LayerNorm stay plain modules and frozen.
    assert isinstance(model.block[1], nn.Linear)
    assert not model.norm.weight.requires_grad


def test_gradients_only_reach_lora_params():
    torch.manual_seed(0)
    model = inject_lora(TinyNet(), ["fc1", "block.1", "head"], r=2)
    model(torch.randn(4, 20)).sum().backward()
    for name, p in model.named_parameters():
        if "lora_" in name:
            assert p.grad is not None, name
        else:
            assert p.grad is None, name


def test_parameter_count_math():
    # Each adapted d x k layer adds r * (d + k) params: A is r x k, B is d x r.
    r = 4
    model = TinyNet()
    total_before = count_parameters(model)
    inject_lora(model, ["fc1", "block.1", "head"], r=r)
    expected_lora = r * (20 + 32) + r * (32 + 16) + r * (16 + 5)
    assert count_parameters(model, trainable_only=True) == expected_lora
    assert count_parameters(model) == total_before + expected_lora
    assert sum(v.numel() for v in lora_state_dict(model).values()) == expected_lora


def test_scaling_is_alpha_over_r():
    layer = LoRALinear(nn.Linear(8, 8), r=4, alpha=16)
    assert layer.scaling == 4.0
    _randomize_B(layer)
    expected = 4.0 * layer.lora_B @ layer.lora_A
    assert torch.allclose(layer.delta_weight(), expected)


def test_no_match_raises():
    try:
        inject_lora(TinyNet(), ["does_not_exist"], r=2)
    except ValueError:
        return
    raise AssertionError("expected ValueError")


if __name__ == "__main__":
    tests = [v for k, v in dict(globals()).items() if k.startswith("test_") and callable(v)]
    for t in tests:
        t()
        print(f"PASS {t.__name__}")
    print(f"{len(tests)} tests passed")
