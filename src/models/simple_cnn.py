"""E1 SimpleCNN (PROJECT_SPEC §8): the assignment's simple CNN baseline, trained from scratch."""
from torch import nn


class SimpleCNN(nn.Module):
    """3 × [Conv3×3 → BN → ReLU → MaxPool2] → global average pooling → Dense(128) → ReLU → Dropout → Dense(K).

    Returns logits; softmax is applied only when probabilities are needed (CrossEntropyLoss expects logits).
    """

    def __init__(self, num_classes, dropout=0.3, widths=(32, 64, 128)):
        super().__init__()
        layers = []
        in_channels = 3
        for out_channels in widths:
            layers += [
                nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, bias=False),
                nn.BatchNorm2d(out_channels),
                nn.ReLU(inplace=True),
                nn.MaxPool2d(2),
            ]
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
