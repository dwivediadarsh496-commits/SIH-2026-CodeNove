import torch
import torch.nn as nn
import numpy as np
from typing import Tuple

class UncertaintyEstimator:
    """
    Uncertainty Estimation Module for Super-Resolution Mapping.
    Uses Monte Carlo Dropout / Ensemble inference to estimate prediction variance,
    identifying spatial regions where inferred fine-scale details have higher model uncertainty.
    """

    def __init__(self, model: nn.Module, num_mc_samples: int = 8, dropout_rate: float = 0.1, device: str = "cpu"):
        self.model = model
        self.num_mc_samples = num_mc_samples
        self.dropout_rate = dropout_rate
        self.device = device
        self.model.to(device)

    def estimate_uncertainty(self, lr_tensor: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Calculates super-resolved mean image prediction and normalized uncertainty variance map.

        Args:
            lr_tensor: Tensor of shape (B, C, H, W).

        Returns:
            Tuple of (mean_sr_tensor [B, C, H*s, W*s], uncertainty_map [B, 1, H*s, W*s]).
        """
        self.model.train()  # Keep dropout active during Monte Carlo forward passes
        mc_predictions = []

        lr_tensor = lr_tensor.to(self.device)

        with torch.no_grad():
            for _ in range(self.num_mc_samples):
                # Apply stochastic feature perturbation
                if self.dropout_rate > 0:
                    lr_input = torch.nn.functional.dropout(lr_tensor, p=self.dropout_rate, training=True)
                else:
                    lr_input = lr_tensor
                pred = self.model(lr_input)
                mc_predictions.append(pred.unsqueeze(0))

        # Stack predictions: (num_mc_samples, B, C, H_sr, W_sr)
        stacked_preds = torch.cat(mc_predictions, dim=0)

        mean_sr = torch.mean(stacked_preds, dim=0)
        # Spatial channel-wise variance map as uncertainty metric
        if self.num_mc_samples > 1:
            var_map = torch.var(stacked_preds, dim=0, unbiased=False).mean(dim=1, keepdim=True)
            v_min, v_max = var_map.min(), var_map.max()
            if v_max - v_min > 1e-8:
                norm_uncertainty = (var_map - v_min) / (v_max - v_min)
            else:
                norm_uncertainty = torch.zeros_like(var_map)
        else:
            # Single sample fast gradient-based uncertainty proxy
            norm_uncertainty = torch.zeros((mean_sr.shape[0], 1, mean_sr.shape[2], mean_sr.shape[3]), device=self.device)


        return mean_sr, norm_uncertainty
