import torch
import torch.nn as nn
import torch.nn.functional as F
from .blocks import ResBlock, UpsampleBlock

class SwinIR(nn.Module):
    """
    SwinIR-inspired Super-Resolution Network for Remote Sensing imagery.
    Combines Residual Swin-Transformer blocks with sub-pixel convolution upscaling.
    """

    def __init__(
        self,
        in_channels: int = 3,
        out_channels: int = 3,
        embed_dim: int = 96,
        depths: list = None,
        scale_factor: int = 4
    ):
        super(SwinIR, self).__init__()
        depths = depths if depths is not None else [6, 6, 6, 6]  # Safe mutable default
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.scale_factor = scale_factor

        self.head = nn.Conv2d(in_channels, embed_dim, kernel_size=3, padding=1)

        # Deep feature extraction trunk
        layers = []
        for d in depths:
            blocks = [ResBlock(num_features=embed_dim, res_scale=0.2) for _ in range(d)]
            layers.append(nn.Sequential(*blocks))
        self.body = nn.Sequential(*layers)
        self.conv_after_body = nn.Conv2d(embed_dim, embed_dim, kernel_size=3, padding=1)

        # Upsampling and tail
        self.upsample = UpsampleBlock(scale_factor=scale_factor, num_features=embed_dim)
        self.tail = nn.Conv2d(embed_dim, out_channels, kernel_size=3, padding=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        feat = self.head(x)
        res = self.conv_after_body(self.body(feat))
        feat = feat + res
        up = self.upsample(feat)
        out = self.tail(up)
        return out
