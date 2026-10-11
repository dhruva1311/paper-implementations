"""Shape and structure tests. Run with: python -m pytest test_model.py  (or: python test_model.py)"""
import torch

from model import BasicBlock, CifarResNet, count_weighted_layers, plainnet, resnet

torch.set_num_threads(2)


def test_basic_block_identity_shortcut_keeps_shape():
    block = BasicBlock(16, 16, stride=1)
    assert isinstance(block.shortcut, torch.nn.Identity)
    assert block(torch.randn(2, 16, 32, 32)).shape == (2, 16, 32, 32)


def test_basic_block_projection_shortcut_downsamples():
    block = BasicBlock(16, 32, stride=2)
    assert isinstance(block.shortcut, torch.nn.Sequential)  # 1x1 conv + BN (option B)
    assert block(torch.randn(2, 16, 32, 32)).shape == (2, 32, 16, 16)


def test_plain_block_has_no_shortcut():
    block = BasicBlock(16, 32, stride=2, residual=False)
    assert isinstance(block.shortcut, torch.nn.Identity)
    assert block(torch.randn(2, 16, 32, 32)).shape == (2, 32, 16, 16)


def test_identity_shortcut_is_used_in_forward():
    """If F(x) is forced to zero, a residual block must output relu(x) and a plain block zeros."""
    x = torch.randn(2, 16, 8, 8)
    for residual, expected in [(True, torch.relu(x)), (False, torch.zeros_like(x))]:
        block = BasicBlock(16, 16, residual=residual).eval()
        torch.nn.init.zeros_(block.bn2.weight)
        assert torch.allclose(block(x), expected)


def test_network_output_shape():
    for build in (resnet, plainnet):
        for depth in (8, 20, 56):
            assert build(depth)(torch.randn(2, 3, 32, 32)).shape == (2, 10)


def test_layer_count_is_6n_plus_2():
    for depth in (8, 20, 32, 56):
        assert count_weighted_layers(resnet(depth)) == depth
        assert count_weighted_layers(plainnet(depth)) == depth


def test_intermediate_feature_map_sizes():
    """Sec. 4.2: feature maps 32x32x16 -> 16x16x32 -> 8x8x64."""
    model = resnet(20)
    x = torch.relu(model.bn1(model.conv1(torch.randn(1, 3, 32, 32))))
    x = model.stage1(x)
    assert x.shape == (1, 16, 32, 32)
    x = model.stage2(x)
    assert x.shape == (1, 32, 16, 16)
    x = model.stage3(x)
    assert x.shape == (1, 64, 8, 8)


def test_resnet20_parameter_count():
    """Paper's Table 6 lists ResNet-20 at 0.27M params (with option A, parameter-free shortcuts).
    Option B adds two 1x1 projection convs + BN, so the count is slightly higher."""
    params = sum(p.numel() for p in resnet(20).parameters())
    assert 0.26e6 < params < 0.28e6


def test_invalid_depth_raises():
    try:
        CifarResNet(depth=21)
    except ValueError:
        return
    raise AssertionError("depth 21 is not 6n+2 and should be rejected")


if __name__ == "__main__":
    for name, fn in list(globals().items()):
        if name.startswith("test_"):
            fn()
            print("ok", name)
