"""E3 EfficientNet-B0 transfer learning (PROJECT_SPEC §8, D010, D022)."""
from torch import nn
from torchvision.models import EfficientNet_B0_Weights, efficientnet_b0

WEIGHTS = EfficientNet_B0_Weights.IMAGENET1K_V1


class EfficientNetB0Transfer(nn.Module):
    """torchvision EfficientNet-B0 with a new head: Dropout → Linear(1280 → K). Returns logits.

    `pretrained=True` loads the ImageNet weights (training); the default `False` only builds the
    architecture, e.g. to load a saved checkpoint. `set_trainable_blocks(n)` freezes the backbone
    except its last `n` feature blocks (0 = head only, stage A). BatchNorm layers inside frozen
    blocks stay in eval mode, so their ImageNet running statistics are not overwritten in training.
    """

    def __init__(self, num_classes, dropout=0.2, pretrained=False):
        super().__init__()
        self.net = efficientnet_b0(weights=WEIGHTS if pretrained else None)
        in_features = self.net.classifier[1].in_features
        self.net.classifier = nn.Sequential(nn.Dropout(dropout), nn.Linear(in_features, num_classes))
        self.trainable_blocks = len(self.net.features)

    def set_trainable_blocks(self, n):
        blocks = list(self.net.features)
        if not 0 <= n <= len(blocks):
            raise ValueError(f"trainable_blocks must be in 0..{len(blocks)}, got {n}")
        self.trainable_blocks = n
        for i, block in enumerate(blocks):
            for p in block.parameters():
                p.requires_grad = i >= len(blocks) - n
        for p in self.net.classifier.parameters():
            p.requires_grad = True
        self.train(self.training)

    def train(self, mode=True):
        super().train(mode)
        blocks = list(self.net.features)
        for block in blocks[: len(blocks) - self.trainable_blocks]:
            block.eval()
        return self

    def forward(self, x):
        return self.net(x)
