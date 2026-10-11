"""ResNet for CIFAR-10, implemented from scratch.

He, Zhang, Ren, Sun. "Deep Residual Learning for Image Recognition", 2015.
https://arxiv.org/abs/1512.03385

The same module builds both networks compared in the paper's Section 4.2:
  - residual=True:  ResNet  (stacked layers learn F(x), block outputs F(x) + x)
  - residual=False: "plain" net (identical layers, no shortcuts, block outputs F(x))
"""
import torch
import torch.nn as nn
import torch.nn.functional as F


class BasicBlock(nn.Module):
    """Two 3x3 conv layers, the building block of Fig. 2 / Fig. 5 (left).

    Residual form, Eq. (1)/(2):  y = relu(F(x) + shortcut(x)),
    where F(x) = BN(conv(relu(BN(conv(x))))).
    BN is applied right after each convolution and before activation (Sec. 3.4).
    """

    def __init__(self, in_channels, out_channels, stride=1, residual=True):
        super().__init__()
        self.residual = residual

        self.conv1 = nn.Conv2d(in_channels, out_channels, 3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, 3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)

        # Shortcut, Sec. 3.3: identity when shapes match (Eq. 1), otherwise a
        # projection W_s x (Eq. 2), implemented as a strided 1x1 conv + BN.
        # This is "option B" from Sec. 4.1.
        if residual and (stride != 1 or in_channels != out_channels):
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, 1, stride=stride, bias=False),
                nn.BatchNorm2d(out_channels),
            )
        else:
            self.shortcut = nn.Identity()

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        if self.residual:
            out = out + self.shortcut(x)
        return F.relu(out)


class CifarResNet(nn.Module):
    """The 6n+2 layer network for CIFAR-10 from Sec. 4.2.

    Layout (input 32x32x3):
      conv3x3, 16 filters                       -> 32x32x16   (1 layer)
      stage 1: n blocks, 16 filters             -> 32x32x16   (2n layers)
      stage 2: n blocks, 32 filters, stride 2   -> 16x16x32   (2n layers)
      stage 3: n blocks, 64 filters, stride 2   ->  8x8x64    (2n layers)
      global average pool, 10-way fully connected            (1 layer)
    Downsampling is done by the first conv of stages 2 and 3 (stride 2).
    """

    def __init__(self, depth=20, num_classes=10, residual=True):
        super().__init__()
        if (depth - 2) % 6 != 0:
            raise ValueError(f"depth must be 6n+2 (e.g. 20, 32, 44, 56), got {depth}")
        n = (depth - 2) // 6

        self.conv1 = nn.Conv2d(3, 16, 3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(16)
        self.stage1 = self._make_stage(16, 16, n, stride=1, residual=residual)
        self.stage2 = self._make_stage(16, 32, n, stride=2, residual=residual)
        self.stage3 = self._make_stage(32, 64, n, stride=2, residual=residual)
        self.fc = nn.Linear(64, num_classes)

        self._init_weights()

    @staticmethod
    def _make_stage(in_channels, out_channels, num_blocks, stride, residual):
        blocks = [BasicBlock(in_channels, out_channels, stride, residual)]
        for _ in range(num_blocks - 1):
            blocks.append(BasicBlock(out_channels, out_channels, 1, residual))
        return nn.Sequential(*blocks)

    def _init_weights(self):
        """He initialization for convs (Sec. 3.4, ref. [13]); BN starts as identity."""
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)

    def forward(self, x):
        x = F.relu(self.bn1(self.conv1(x)))
        x = self.stage1(x)
        x = self.stage2(x)
        x = self.stage3(x)
        x = F.adaptive_avg_pool2d(x, 1).flatten(1)
        return self.fc(x)


def resnet(depth, num_classes=10):
    return CifarResNet(depth, num_classes, residual=True)


def plainnet(depth, num_classes=10):
    return CifarResNet(depth, num_classes, residual=False)


def count_weighted_layers(model):
    """Count conv + fc layers on the main path (shortcut projections excluded, as in the paper)."""
    count = 0
    for name, m in model.named_modules():
        if isinstance(m, (nn.Conv2d, nn.Linear)) and "shortcut" not in name:
            count += 1
    return count
