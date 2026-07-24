"""Baseline CNN for wafer map classification.

Deliberately boring: three conv stages, global average pooling, a linear head.
~300k parameters. The baseline's job is to be a standard, unarguable reference —
every later method is measured against this.
"""

import torch
import torch.nn as nn


def conv_block(c_in: int, c_out: int) -> nn.Sequential:
    return nn.Sequential(
        nn.Conv2d(c_in, c_out, 3, padding=1, bias=False),
        nn.BatchNorm2d(c_out),
        nn.ReLU(inplace=True),
        nn.Conv2d(c_out, c_out, 3, padding=1, bias=False),
        nn.BatchNorm2d(c_out),
        nn.ReLU(inplace=True),
        nn.MaxPool2d(2),
    )


class WaferCNN(nn.Module):
    def __init__(self, n_classes: int = 9):
        super().__init__()
        self.features = nn.Sequential(
            conv_block(2, 32),    # 64 -> 32
            conv_block(32, 64),   # 32 -> 16
            conv_block(64, 128),  # 16 -> 8
        )
        self.head = nn.Linear(128, n_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.features(x).mean(dim=(2, 3))  # global average pool -> (B, 128)
        return self.head(h)                    # logits; softmax lives in the loss


if __name__ == "__main__":
    m = WaferCNN()
    n = sum(p.numel() for p in m.parameters())
    out = m(torch.randn(4, 2, 64, 64))
    print(f"params: {n:,}, output shape: {tuple(out.shape)}")
