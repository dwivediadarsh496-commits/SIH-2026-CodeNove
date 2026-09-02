import torch
import torch.nn as nn
from typing import Tuple, Dict
from .reconstruction import ReconstructionLoss
from .spectral import SpectralAngleLoss
from .perceptual import StructuralEdgeLoss

class GeoSRCombinedLoss(nn.Module):
    """
    Compound Loss Function for Remote Sensing Super-Resolution:
      Total Loss = λ1 * L_recon + λ2 * L_spectral + λ3 * L_structural
    All weights are configurable via base configuration YAML.
    """

    def __init__(
        self,
        reconstruction_weight: float = 1.0,
        spectral_weight: float = 0.1,
        structural_weight: float = 0.05
    ):
        super(GeoSRCombinedLoss, self).__init__()
        self.reconstruction_weight = reconstruction_weight
        self.spectral_weight = spectral_weight
        self.structural_weight = structural_weight

        self.recon_loss = ReconstructionLoss(mode="l1")
        self.spectral_loss = SpectralAngleLoss()
        self.structural_loss = StructuralEdgeLoss()

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> Tuple[torch.Tensor, Dict[str, float]]:
        l_recon = self.recon_loss(pred, target)
        l_spec = self.spectral_loss(pred, target) if self.spectral_weight > 0 else torch.tensor(0.0, device=pred.device)
        l_struct = self.structural_loss(pred, target) if self.structural_weight > 0 else torch.tensor(0.0, device=pred.device)

        total_loss = (
            self.reconstruction_weight * l_recon
            + self.spectral_weight * l_spec
            + self.structural_weight * l_struct
        )

        loss_dict = {
            "total_loss": float(total_loss.item()),
            "reconstruction_loss": float(l_recon.item()),
            "spectral_loss": float(l_spec.item()),
            "structural_loss": float(l_struct.item())
        }

        return total_loss, loss_dict
