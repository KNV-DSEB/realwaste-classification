from torch import nn


class SimpleCNN(nn.Module):
    """E1: 3 x [conv 3x3 -> BN -> ReLU -> max-pool] -> global avg pool -> FC 128 -> dropout -> FC K."""

    def __init__(self, num_classes, dropout=0.3, widths=(32, 64, 128)):
        super().__init__()
        layers, c_in = [], 3
        for c_out in widths:
            layers += [nn.Conv2d(c_in, c_out, 3, padding=1, bias=False), nn.BatchNorm2d(c_out),
                       nn.ReLU(inplace=True), nn.MaxPool2d(2)]
            c_in = c_out
        self.features = nn.Sequential(*layers)
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.classifier = nn.Sequential(nn.Flatten(), nn.Linear(c_in, 128), nn.ReLU(inplace=True),
                                        nn.Dropout(dropout), nn.Linear(128, num_classes))

    def forward(self, x):
        return self.classifier(self.pool(self.features(x)))
