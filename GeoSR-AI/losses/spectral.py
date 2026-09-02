import torch
import torch.nn as nn

class SpectralAngleLoss(nn.Module):
    """
    Differentiable Spectral Angle Mapper (SAM) loss in PyTorch.
    Penalizes spectral distortion across channels for multi-spectral remote sensing.
    """

    def __init__(self, eps: float = 1e-8):
        super(SpectralAngleLoss, self).__init__()
        self.eps = eps

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """
        Args:
            pred: Super-resolved tensor (B, C, H, W).
            target: High-resolution reference tensor (B, C, H, W).

        Returns:
            Scalar spectral angle loss (radians).
        """
        dot_product = torch.sum(pred * target, dim=1)
        norm_pred = torch.norm(pred, p=2, dim=1)
        norm_target = torch.norm(target, p=2, dim=1)

        denom = norm_pred * norm_target + self.eps
        cos_theta = torch.clamp(dot_product / denom, -1.0 + 1e-7, 1.0 - 1e-7)

        sam_map = torch.acos(cos_theta)
        return torch.mean(sam_map)
