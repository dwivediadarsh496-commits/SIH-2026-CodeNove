import torch.nn as nn
from .srcnn import SRCNN
from .edsr import EDSR
from .swinir import SwinIR

def create_model(model_name: str, in_channels: int = 3, out_channels: int = 3, scale_factor: int = 4, **kwargs) -> nn.Module:
    """
    Factory function for instantiating Super-Resolution models.

    Args:
        model_name: Name string ('srcnn', 'edsr', 'swinir').
        in_channels: Number of input spectral bands.
        out_channels: Number of output spectral bands.
        scale_factor: Super-resolution scaling factor (e.g. 4 for 10m -> 2.5m).

    Returns:
        PyTorch nn.Module object.
    """
    name = model_name.lower()
    if name == "srcnn":
        return SRCNN(
            in_channels=in_channels,
            out_channels=out_channels,
            scale_factor=scale_factor,
            num_features=kwargs.get("num_features", 64)
        )
    elif name == "edsr":
        return EDSR(
            in_channels=in_channels,
            out_channels=out_channels,
            scale_factor=scale_factor,
            num_features=kwargs.get("num_features", 64),
            num_blocks=kwargs.get("num_blocks", 16),
            res_scale=kwargs.get("res_scale", 0.1)
        )
    elif name == "swinir":
        return SwinIR(
            in_channels=in_channels,
            out_channels=out_channels,
            scale_factor=scale_factor,
            embed_dim=kwargs.get("embed_dim", 96),
            depths=kwargs.get("depths", [6, 6, 6, 6])
        )
    else:
        raise ValueError(f"Unknown model architecture: {model_name}. Supported: 'srcnn', 'edsr', 'swinir'")
