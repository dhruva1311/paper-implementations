"""U-Net (Ronneberger, Fischer, Brox, 2015), implemented from scratch.

Paper: https://arxiv.org/abs/1505.04597

Section 2 ("Network Architecture") describes a contracting path that captures
context and a symmetric expanding path that enables precise localization.
Feature maps from the contracting path are copied (and cropped) into the
expanding path, which is what lets the network output sharp masks.
"""

import torch
import torch.nn as nn


def center_crop(x: torch.Tensor, height: int, width: int) -> torch.Tensor:
    """Crop the spatial center of x (N, C, H, W) to (height, width).

    Section 2: "The cropping is necessary due to the loss of border pixels in
    every convolution." With padded convolutions the sizes already match and
    this is a no-op.
    """
    _, _, h, w = x.shape
    top = (h - height) // 2
    left = (w - width) // 2
    return x[:, :, top:top + height, left:left + width]


class DoubleConv(nn.Module):
    """Two 3x3 convolutions, each followed by a ReLU (Section 2).

    padding=False gives the paper's unpadded ("valid") convs, which lose 1 pixel
    per side per conv. batch_norm is not in the paper.
    """

    def __init__(self, in_channels: int, out_channels: int, padding: bool, batch_norm: bool):
        super().__init__()
        pad = 1 if padding else 0
        layers = []
        for c_in in (in_channels, out_channels):
            layers.append(nn.Conv2d(c_in, out_channels, kernel_size=3, padding=pad, bias=not batch_norm))
            if batch_norm:
                layers.append(nn.BatchNorm2d(out_channels))
            layers.append(nn.ReLU(inplace=True))
        self.block = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.block(x)


class Down(nn.Module):
    """One contracting step: 2x2 max pool (stride 2), then DoubleConv.

    Section 2: "At each downsampling step we double the number of feature
    channels."
    """

    def __init__(self, in_channels: int, out_channels: int, padding: bool, batch_norm: bool):
        super().__init__()
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.conv = DoubleConv(in_channels, out_channels, padding, batch_norm)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.shape[-2] % 2 or x.shape[-1] % 2:
            raise ValueError(
                f"Feature map of size {tuple(x.shape[-2:])} cannot be max-pooled evenly. "
                "Section 2: choose an input size where every 2x2 pooling sees an even size."
            )
        return self.conv(self.pool(x))


class Up(nn.Module):
    """One expanding step: 2x2 up-convolution, concatenate cropped skip, DoubleConv (Section 2).

    The up-convolution is a stride-2 2x2 transposed conv that also halves the channels.
    """

    def __init__(self, in_channels: int, out_channels: int, padding: bool, batch_norm: bool):
        super().__init__()
        self.up = nn.ConvTranspose2d(in_channels, in_channels // 2, kernel_size=2, stride=2)
        # After concatenation: in_channels // 2 (upsampled) + in_channels // 2 (skip).
        self.conv = DoubleConv(in_channels, out_channels, padding, batch_norm)

    def forward(self, x: torch.Tensor, skip: torch.Tensor) -> torch.Tensor:
        x = self.up(x)
        skip = center_crop(skip, x.shape[-2], x.shape[-1])
        x = torch.cat([skip, x], dim=1)
        return self.conv(x)


class UNet(nn.Module):
    """U-Net with configurable width, depth and padding.

    With base_channels=64, depth=4, padding=False, batch_norm=False this is the
    network in Figure 1 of the paper: 572x572 input -> 388x388 output, channels
    64-128-256-512-1024 and back.

    Args:
        in_channels: Image channels (1 for the paper's microscopy, 3 for RGB).
        num_classes: Output channels; the paper uses 2 with a pixel-wise softmax.
        base_channels: Channels at the first level, doubled at every Down step.
        depth: Number of Down (and Up) steps. The paper uses 4.
        padding: True for same-size convs, False for the paper's valid convs.
        batch_norm: Add BatchNorm after each conv (not in the paper).
    """

    def __init__(
        self,
        in_channels: int = 3,
        num_classes: int = 2,
        base_channels: int = 64,
        depth: int = 4,
        padding: bool = True,
        batch_norm: bool = True,
    ):
        super().__init__()
        channels = [base_channels * 2**i for i in range(depth + 1)]

        self.inc = DoubleConv(in_channels, channels[0], padding, batch_norm)
        self.downs = nn.ModuleList(
            Down(channels[i], channels[i + 1], padding, batch_norm) for i in range(depth)
        )
        self.ups = nn.ModuleList(
            Up(channels[i + 1], channels[i], padding, batch_norm) for i in reversed(range(depth))
        )
        # Section 2: "At the final layer a 1x1 convolution is used to map each
        # 64-component feature vector to the desired number of classes."
        self.head = nn.Conv2d(channels[0], num_classes, kernel_size=1)

        self._init_weights()

    def _init_weights(self) -> None:
        """Section 3: weights drawn from a Gaussian with std sqrt(2/N), where N
        is the number of incoming nodes of one neuron (He initialization)."""
        for m in self.modules():
            if isinstance(m, (nn.Conv2d, nn.ConvTranspose2d)):
                nn.init.kaiming_normal_(m.weight, mode="fan_in", nonlinearity="relu")
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.inc(x)
        skips = [x]
        for down in self.downs:
            x = down(x)
            skips.append(x)

        skips.pop()  # the bottleneck output is x itself, not a skip connection
        for up in self.ups:
            x = up(x, skips.pop())
        return self.head(x)


def dice_score(pred: torch.Tensor, target: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    """Per-image Dice coefficient 2|A∩B| / (|A| + |B|) for binary masks (N, H, W).

    Returns a tensor of shape (N,). An image where both masks are empty scores 1.
    """
    pred = pred.flatten(1).float()
    target = target.flatten(1).float()
    intersection = (pred * target).sum(dim=1)
    total = pred.sum(dim=1) + target.sum(dim=1)
    return (2 * intersection + eps) / (total + eps)


def iou_score(pred: torch.Tensor, target: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    """Per-image Intersection over Union |A∩B| / |A∪B| for binary masks (N, H, W).

    The paper reports IoU for the ISBI 2015 cell tracking challenge (Section 4).
    """
    pred = pred.flatten(1).float()
    target = target.flatten(1).float()
    intersection = (pred * target).sum(dim=1)
    union = pred.sum(dim=1) + target.sum(dim=1) - intersection
    return (intersection + eps) / (union + eps)
