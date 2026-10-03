"""E2 MultiScaleCNN (PROJECT_SPEC §8, D013, D020): the assignment's complex CNN, trained from scratch."""
import torch
from torch import nn


class MultiScaleBlock(nn.Module):
    """Parallel 1×1, 3×3 and 5×5 convolutions on the same input (1/4, 1/2, 1/4 of the output
    channels), concatenated, then BatchNorm → ReLU."""

    def __init__(self, in_channels, out_channels):
        super().__init__()
        c1 = c5 = out_channels // 4
        c3 = out_channels - c1 - c5
        self.branch1 = nn.Conv2d(in_channels, c1, kernel_size=1, bias=False)
        self.branch3 = nn.Conv2d(in_channels, c3, kernel_size=3, padding=1, bias=False)
        self.branch5 = nn.Conv2d(in_channels, c5, kernel_size=5, padding=2, bias=False)
        self.bn = nn.BatchNorm2d(out_channels)
        self.act = nn.ReLU(inplace=True)

    def forward(self, x):
        return self.act(self.bn(torch.cat([self.branch1(x), self.branch3(x), self.branch5(x)], dim=1)))


class MultiScaleCNN(nn.Module):
    """Stem [Conv3×3 → BN → ReLU → MaxPool] → N × [MultiScaleBlock → MaxPool] → global average pooling
    → Dense(128) → ReLU → Dropout → Dense(K). Returns logits."""

    def __init__(self, num_classes, dropout=0.3, stem_width=32, widths=(64, 128, 256, 256)):
        super().__init__()
        layers = [
            nn.Conv2d(3, stem_width, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(stem_width),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
        ]
        in_channels = stem_width
        for out_channels in widths:
            layers += [MultiScaleBlock(in_channels, out_channels), nn.MaxPool2d(2)]
            in_channels = out_channels
        self.features = nn.Sequential(*layers)
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(in_channels, 128),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(128, num_classes),
        )

    def forward(self, x):
        return self.classifier(self.pool(self.features(x)))
