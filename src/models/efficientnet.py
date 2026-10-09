from torch import nn
from torchvision.models import EfficientNet_B0_Weights, efficientnet_b0

WEIGHTS = EfficientNet_B0_Weights.IMAGENET1K_V1


class EfficientNetB0Transfer(nn.Module):
    """E3: EfficientNet-B0 with a new 9-class head. pretrained=True loads the ImageNet weights."""

    def __init__(self, num_classes, dropout=0.2, pretrained=False):
        super().__init__()
        self.net = efficientnet_b0(weights=WEIGHTS if pretrained else None)
        self.net.classifier = nn.Sequential(nn.Dropout(dropout), nn.Linear(1280, num_classes))
        self.trainable_blocks = len(self.net.features)

    def set_trainable_blocks(self, n):
        """Freeze the backbone except its last n blocks (n=0: train only the head)."""
        self.trainable_blocks = n
        first = len(self.net.features) - n
        for i, block in enumerate(self.net.features):
            block.requires_grad_(i >= first)
        self.train(self.training)

    def train(self, mode=True):
        # frozen blocks stay in eval mode so their BatchNorm keeps the ImageNet statistics
        super().train(mode)
        for block in self.net.features[:len(self.net.features) - self.trainable_blocks]:
            block.eval()
        return self

    def forward(self, x):
        return self.net(x)
