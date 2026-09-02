import torch
import torch.nn as nn
from .blocks import ResBlock, UpsampleBlock

class EDSR(nn.Module):
    """
    Enhanced Deep Residual Super-Resolution Network (EDSR).
    Optimized for multi-spectral remote sensing super-resolution mapping.
    """

    def __init__(
        self,
        in_channels: int = 3,
        out_channels: int = 3,
        num_features: int = 64,
        num_blocks: int = 16,
        res_scale: float = 0.1,
        scale_factor: int = 4
    ):
        super(EDSR, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.scale_factor = scale_factor

        # Head conv
        self.head = nn.Conv2d(in_channels, num_features, kernel_size=3, padding=1)

        # Residual trunk
        body = [ResBlock(num_features=num_features, res_scale=res_scale) for _ in range(num_blocks)]
        body.append(nn.Conv2d(num_features, num_features, kernel_size=3, padding=1))
        self.body = nn.Sequential(*body)

        # Upsampler trunk
        self.upsample = UpsampleBlock(scale_factor=scale_factor, num_features=num_features)

        # Tail conv
        self.tail = nn.Conv2d(num_features, out_channels, kernel_size=3, padding=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        feat = self.head(x)
        res = self.body(feat)
        feat = feat + res
        up = self.upsample(feat)
        out = self.tail(up)
        return out
