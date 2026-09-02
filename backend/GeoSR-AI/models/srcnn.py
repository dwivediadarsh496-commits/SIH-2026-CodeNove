import torch
import torch.nn as nn
import torch.nn.functional as F

class SRCNN(nn.Module):
    """
    Super-Resolution Convolutional Neural Network (SRCNN).
    Baseline 2 implementation supporting multi-channel remote sensing inputs.
    """

    def __init__(
        self,
        in_channels: int = 3,
        out_channels: int = 3,
        num_features: int = 64,
        scale_factor: int = 4
    ):
        super(SRCNN, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.scale_factor = scale_factor

        # Feature Extraction: 9x9 conv
        self.conv1 = nn.Conv2d(in_channels, num_features, kernel_size=9, padding=4)
        # Non-linear Mapping: 5x5 conv
        self.conv2 = nn.Conv2d(num_features, num_features // 2, kernel_size=5, padding=2)
        # Reconstruction: 5x5 conv
        self.conv3 = nn.Conv2d(num_features // 2, out_channels, kernel_size=5, padding=2)

        self.relu = nn.ReLU(inplace=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Input LR tensor of shape (B, C, H, W).

        Returns:
            Super-resolved tensor of shape (B, C, H * scale_factor, W * scale_factor).
        """
        # Bicubic pre-upsampling to target SR resolution
        target_h = x.shape[-2] * self.scale_factor
        target_w = x.shape[-1] * self.scale_factor
        if x.shape[-2] != target_h or x.shape[-1] != target_w:
            x_up = F.interpolate(x, size=(target_h, target_w), mode="bicubic", align_corners=False)
        else:
            x_up = x

        out = self.relu(self.conv1(x_up))
        out = self.relu(self.conv2(out))
        out = self.conv3(out)
        return out
