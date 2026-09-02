import torch
import torch.nn as nn

class ResBlock(nn.Module):
    """
    Enhanced Deep Residual Block without Batch Normalization (EDSR-style).
    Preserves dynamic range and reflectance scaling for remote sensing data.
    """

    def __init__(self, num_features: int = 64, res_scale: float = 0.1):
        super(ResBlock, self).__init__()
        self.res_scale = res_scale
        self.conv1 = nn.Conv2d(num_features, num_features, kernel_size=3, padding=1)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(num_features, num_features, kernel_size=3, padding=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        res = self.conv2(self.relu(self.conv1(x)))
        return x + res * self.res_scale

class UpsampleBlock(nn.Sequential):
    """
    Sub-Pixel Convolutional Upsampling Block (PixelShuffle).
    Supports scale factors of 2, 3, 4.
    """

    def __init__(self, scale_factor: int, num_features: int):
        m = []
        if (scale_factor & (scale_factor - 1)) == 0:
            for _ in range(int(torch.log2(torch.tensor(scale_factor)))):
                m.append(nn.Conv2d(num_features, 4 * num_features, kernel_size=3, padding=1))
                m.append(nn.PixelShuffle(2))
                m.append(nn.ReLU(inplace=True))
        elif scale_factor == 3:
            m.append(nn.Conv2d(num_features, 9 * num_features, kernel_size=3, padding=1))
            m.append(nn.PixelShuffle(3))
            m.append(nn.ReLU(inplace=True))
        else:
            raise ValueError(f"Scale factor {scale_factor} not supported.")

        super(UpsampleBlock, self).__init__(*m)
