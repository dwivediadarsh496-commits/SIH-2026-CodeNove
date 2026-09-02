import torch
import torch.nn as nn
import torch.nn.functional as F

class StructuralEdgeLoss(nn.Module):
    """
    Structural Edge / Gradient Loss using Sobel filters.
    Preserves fine structural features (roads, buildings, field boundaries)
    without relying on ImageNet/VGG perceptual weights unsuited for satellite data.
    """

    def __init__(self):
        super(StructuralEdgeLoss, self).__init__()
        # Sobel Horizontal and Vertical Kernel
        sobel_x = torch.tensor([[-1., 0., 1.], [-2., 0., 2.], [-1., 0., 1.]], dtype=torch.float32).view(1, 1, 3, 3)
        sobel_y = torch.tensor([[-1., -2., -1.], [0., 0., 0.], [1., 2., 1.]], dtype=torch.float32).view(1, 1, 3, 3)

        self.register_buffer("sobel_x", sobel_x)
        self.register_buffer("sobel_y", sobel_y)

    def _gradient_magnitude(self, x: torch.Tensor) -> torch.Tensor:
        b, c, h, w = x.shape
        x_flat = x.view(b * c, 1, h, w)

        grad_x = F.conv2d(x_flat, self.sobel_x, padding=1)
        grad_y = F.conv2d(x_flat, self.sobel_y, padding=1)

        grad_mag = torch.sqrt(grad_x ** 2 + grad_y ** 2 + 1e-8)
        return grad_mag.view(b, c, h, w)

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        grad_pred = self._gradient_magnitude(pred)
        grad_target = self._gradient_magnitude(target)
        return F.l1_loss(grad_pred, grad_target)
