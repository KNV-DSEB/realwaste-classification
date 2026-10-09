import torch
from torch import nn


class MultiScaleBlock(nn.Module):
    """1x1, 3x3 and 5x5 convolutions side by side (1/4, 1/2, 1/4 of the channels), concatenated."""

    def __init__(self, c_in, c_out):
        super().__init__()
        c1 = c5 = c_out // 4
        self.branch1 = nn.Conv2d(c_in, c1, 1, bias=False)
        self.branch3 = nn.Conv2d(c_in, c_out - c1 - c5, 3, padding=1, bias=False)
        self.branch5 = nn.Conv2d(c_in, c5, 5, padding=2, bias=False)
        self.bn = nn.BatchNorm2d(c_out)
        self.act = nn.ReLU(inplace=True)

    def forward(self, x):
        return self.act(self.bn(torch.cat([self.branch1(x), self.branch3(x), self.branch5(x)], dim=1)))


class MultiScaleCNN(nn.Module):
    """E2: conv stem -> 4 x [multi-scale block -> max-pool] -> global avg pool -> FC 128 -> dropout -> FC K."""

    def __init__(self, num_classes, dropout=0.3, stem_width=32, widths=(64, 128, 256, 256)):
        super().__init__()
        layers = [nn.Conv2d(3, stem_width, 3, padding=1, bias=False), nn.BatchNorm2d(stem_width),
                  nn.ReLU(inplace=True), nn.MaxPool2d(2)]
        c_in = stem_width
        for c_out in widths:
            layers += [MultiScaleBlock(c_in, c_out), nn.MaxPool2d(2)]
            c_in = c_out
        self.features = nn.Sequential(*layers)
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.classifier = nn.Sequential(nn.Flatten(), nn.Linear(c_in, 128), nn.ReLU(inplace=True),
                                        nn.Dropout(dropout), nn.Linear(128, num_classes))

    def forward(self, x):
        return self.classifier(self.pool(self.features(x)))
