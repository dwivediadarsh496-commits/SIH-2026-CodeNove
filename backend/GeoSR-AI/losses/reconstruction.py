import torch
import torch.nn as nn

class ReconstructionLoss(nn.Module):
    """
    L1 and L2 Spatial Reconstruction Loss.
    """

    def __init__(self, mode: str = "l1"):
        super(ReconstructionLoss, self).__init__()
        self.mode = mode.lower()
        if self.mode == "l1":
            self.loss_fn = nn.L1Loss()
        elif self.mode == "l2":
            self.loss_fn = nn.MSELoss()
        else:
            raise ValueError(f"Unsupported reconstruction loss mode: {mode}")

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        return self.loss_fn(pred, target)
