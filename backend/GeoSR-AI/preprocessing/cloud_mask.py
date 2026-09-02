import numpy as np
import torch
from typing import Union, Tuple, Optional

class CloudMasker:
    """
    Cloud and nodata pixel handling module.
    Detects invalid pixels, cloud cover, and shadow regions using Sentinel-2 SCL
    or reflectance thresholding to prevent loss calculation on corrupted regions.
    """

    # Sentinel-2 SCL (Scene Classification Layer) class codes
    SCL_SATURATED = 1
    SCL_CAST_SHADOW = 3
    SCL_CLOUD_LOW_PROB = 7
    SCL_CLOUD_MEDIUM_PROB = 8
    SCL_CLOUD_HIGH_PROB = 9
    SCL_THIN_CIRRUS = 10
    SCL_SNOW = 11

    @staticmethod
    def get_scl_cloud_mask(scl_array: np.ndarray) -> np.ndarray:
        """
        Creates binary mask (True = valid pixel, False = cloud/shadow/invalid).
        """
        invalid_codes = [
            CloudMasker.SCL_SATURATED,
            CloudMasker.SCL_CAST_SHADOW,
            CloudMasker.SCL_CLOUD_LOW_PROB,
            CloudMasker.SCL_CLOUD_MEDIUM_PROB,
            CloudMasker.SCL_CLOUD_HIGH_PROB,
            CloudMasker.SCL_THIN_CIRRUS
        ]
        invalid_mask = np.isin(scl_array, invalid_codes)
        valid_mask = ~invalid_mask
        return valid_mask

    @staticmethod
    def get_reflectance_threshold_mask(
        rgb_array: np.ndarray,
        bright_threshold: float = 0.8,
        nodata_val: Optional[float] = 0.0
    ) -> np.ndarray:
        """
        Calculates valid pixel mask for RGB image array (C, H, W) normalized to [0, 1].
        Masks out over-saturated clouds (all channels > bright_threshold) and zero nodata values.

        Returns:
            Boolean mask array (H, W) where True = valid.
        """
        if rgb_array.ndim == 3:
            # Mask out pixels where all channels are 0 (nodata)
            nonzero_mask = np.any(rgb_array > (nodata_val if nodata_val is not None else 0.0), axis=0)
            # Mask out saturated cloud glint across RGB
            not_cloud_mask = ~np.all(rgb_array > bright_threshold, axis=0)
            valid_mask = nonzero_mask & not_cloud_mask
        else:
            valid_mask = (rgb_array > 0.0) & (rgb_array < bright_threshold)

        return valid_mask
