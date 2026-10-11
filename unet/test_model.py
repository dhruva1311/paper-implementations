"""Shape and metric tests. Run with: python -m pytest test_model.py  (or python test_model.py)"""

import pytest
import torch

from model import UNet, center_crop, dice_score, iou_score

torch.set_num_threads(2)


@pytest.mark.parametrize("size", [64, 96, 128])
@pytest.mark.parametrize("depth", [3, 4])
def test_padded_unet_keeps_input_size(size, depth):
    model = UNet(in_channels=3, num_classes=2, base_channels=4, depth=depth, padding=True)
    out = model(torch.randn(2, 3, size, size))
    assert out.shape == (2, 2, size, size)


def test_valid_unet_matches_paper_figure_1():
    """Figure 1: a 572x572 input tile gives a 388x388 output segmentation map.

    Width does not affect spatial shapes, so a narrow model keeps the test fast.
    """
    model = UNet(in_channels=1, num_classes=2, base_channels=2, depth=4, padding=False, batch_norm=False)
    out = model(torch.randn(1, 1, 572, 572))
    assert out.shape == (1, 2, 388, 388)


def test_paper_channel_widths():
    model = UNet(in_channels=1, num_classes=2, base_channels=64, depth=4, padding=False, batch_norm=False)
    assert [d.conv.block[-2].out_channels for d in model.downs] == [128, 256, 512, 1024]
    assert [u.up.out_channels for u in model.ups] == [512, 256, 128, 64]
    assert model.head.in_channels == 64 and model.head.kernel_size == (1, 1)


def test_odd_feature_map_raises():
    model = UNet(base_channels=4, depth=4, padding=True)
    with pytest.raises(ValueError):
        model(torch.randn(1, 3, 72, 72))  # 72 -> 36 -> 18 -> 9 -> cannot pool 9


def test_center_crop():
    x = torch.arange(36.0).reshape(1, 1, 6, 6)
    crop = center_crop(x, 2, 2)
    assert crop.shape == (1, 1, 2, 2)
    assert crop.flatten().tolist() == [14.0, 15.0, 20.0, 21.0]


def test_gradients_reach_first_layer():
    model = UNet(base_channels=4, depth=3)
    model(torch.randn(2, 3, 32, 32)).sum().backward()
    assert model.inc.block[0].weight.grad.abs().sum() > 0


def test_dice_and_iou():
    target = torch.zeros(1, 4, 4, dtype=torch.long)
    target[0, :2] = 1  # 8 foreground pixels
    pred = torch.zeros_like(target)
    pred[0, 1:3] = 1  # 8 predicted, 4 overlap
    assert dice_score(pred, target).item() == pytest.approx(2 * 4 / 16)
    assert iou_score(pred, target).item() == pytest.approx(4 / 12)
    assert dice_score(target, target).item() == pytest.approx(1.0)
    empty = torch.zeros_like(target)
    assert dice_score(empty, empty).item() == pytest.approx(1.0)
    assert iou_score(empty, target).item() == pytest.approx(0.0, abs=1e-6)


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v"]))
