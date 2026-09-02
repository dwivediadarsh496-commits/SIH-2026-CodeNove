import numpy as np
import torch
from typing import Tuple, Optional, Union, Dict, Any

class SatelliteNormalizer:
    """
    Remote-Sensing-aware normalizer.
    Supports min-max, percentile clipping (1st - 99th), Sentinel-2 L2A reflectance scaling (10000),
    and dataset-level mean/std normalization. Preserves float32 precision for spectral metrics.
    """

    def __init__(
        self,
        method: str = "percentile",
        percentiles: Tuple[float, float] = (1.0, 99.0),
        reflectance_scale: float = 10000.0,
        mean: Optional[Union[list, np.ndarray]] = None,
        std: Optional[Union[list, np.ndarray]] = None
    ):
        self.method = method.lower()
        self.percentiles = percentiles
        self.reflectance_scale = reflectance_scale
        self.mean = np.array(mean) if mean is not None else None
        self.std = np.array(std) if std is not None else None

    def normalize(
        self,
        data: np.ndarray,
        stats_out: Optional[Dict[str, Any]] = None
    ) -> Tuple[np.ndarray, Dict[str, Any]]:
        """
        Normalizes satellite imagery array of shape (C, H, W) or (H, W, C) to range [0, 1] or zero-mean.

        Returns:
            Tuple of (normalized_data, stats_dict) for inverse transformation.
        """
        is_torch = isinstance(data, torch.Tensor)
        if is_torch:
            arr = data.detach().cpu().numpy()
        else:
            arr = data.copy()

        stats = {"method": self.method, "original_dtype": str(arr.dtype)}

        if self.method == "reflectance_scale":
            # Direct Sentinel-2 L2A BOA scaling (0 - 10000 -> 0.0 - 1.0)
            norm_arr = arr / float(self.reflectance_scale)
            stats["scale"] = self.reflectance_scale

        elif self.method == "percentile":
            # Percentile clipping (robust to cloud glints and zero nodata)
            p_min, p_max = self.percentiles
            valid_mask = arr > 0
            if np.any(valid_mask):
                c_min = np.percentile(arr[valid_mask], p_min)
                c_max = np.percentile(arr[valid_mask], p_max)
            else:
                c_min, c_max = arr.min(), arr.max()

            if c_max - c_min < 1e-6:
                c_max = c_min + 1.0

            norm_arr = np.clip((arr - c_min) / (c_max - c_min), 0.0, 1.0)
            stats["min"] = c_min
            stats["max"] = c_max

        elif self.method == "min_max":
            c_min, c_max = arr.min(), arr.max()
            if c_max - c_min < 1e-6:
                c_max = c_min + 1.0
            norm_arr = (arr - c_min) / (c_max - c_min)
            stats["min"] = c_min
            stats["max"] = c_max

        elif self.method == "dataset_stats":
            if self.mean is None or self.std is None:
                raise ValueError("Mean and std must be provided for dataset_stats normalization.")
            # Reshape mean/std for broadcasting
            if arr.ndim == 3 and arr.shape[0] == len(self.mean):
                m = self.mean[:, None, None]
                s = self.std[:, None, None]
            else:
                m, s = self.mean, self.std
            norm_arr = (arr - m) / (s + 1e-8)
            stats["mean"] = self.mean
            stats["std"] = self.std

        else:
            raise ValueError(f"Unknown normalization method: {self.method}")

        if is_torch:
            norm_arr = torch.from_numpy(norm_arr).to(data.device)

        return norm_arr.astype(np.float32) if not is_torch else norm_arr.float(), stats

    def denormalize(
        self,
        data: Union[np.ndarray, torch.Tensor],
        stats: Dict[str, Any]
    ) -> Union[np.ndarray, torch.Tensor]:
        """
        Reverses normalization to restore original spectral reflectance values.
        """
        is_torch = isinstance(data, torch.Tensor)
        arr = data.detach().cpu().numpy() if is_torch else data.copy()

        method = stats.get("method", self.method)

        if method == "reflectance_scale":
            denorm = arr * stats.get("scale", self.reflectance_scale)
        elif method in ["percentile", "min_max"]:
            c_min = stats.get("min", 0.0)
            c_max = stats.get("max", 1.0)
            denorm = arr * (c_max - c_min) + c_min
        elif method == "dataset_stats":
            m = stats.get("mean", self.mean)
            s = stats.get("std", self.std)
            if arr.ndim == 3 and arr.shape[0] == len(m):
                m = m[:, None, None]
                s = s[:, None, None]
            denorm = arr * (s + 1e-8) + m
        else:
            denorm = arr

        if is_torch:
            return torch.from_numpy(denorm).to(data.device)
        return denorm
