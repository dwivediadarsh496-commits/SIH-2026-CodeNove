import numpy as np
import cv2
import scipy.ndimage
from typing import Tuple, Optional, Union
import torch

class ControlledDegradation:
    """
    Simulates satellite imaging sensor degradation pipeline.
    Generates spatially aligned LR input from high-resolution reference rasters.

    Pipeline:
      HR Reference -> Gaussian Blur (Point Spread Function) -> Sensor Noise -> Bicubic Downsampling -> LR Tensor
    """

    def __init__(
        self,
        scale_factor: int = 4,
        blur_kernel_size: int = 7,
        blur_sigma: float = 1.5,
        noise_type: str = "gaussian",
        noise_sigma: float = 0.005
    ):
        self.scale_factor = scale_factor
        self.blur_kernel_size = blur_kernel_size
        self.blur_sigma = blur_sigma
        self.noise_type = noise_type.lower()
        self.noise_sigma = noise_sigma

    def degrade(self, hr_image: np.ndarray) -> np.ndarray:
        """
        Applies controlled degradation to an HR image array (C, H, W) or (H, W, C).

        Args:
            hr_image: Floating point HR array in range [0, 1] or raw reflectance.

        Returns:
            LR image array of shape (C, H // scale_factor, W // scale_factor).
        """
        is_chw = (hr_image.ndim == 3 and hr_image.shape[0] in [1, 3, 4])
        if is_chw:
            # Convert to (H, W, C) for OpenCV processing
            img = np.transpose(hr_image, (1, 2, 0)).copy()
        else:
            img = hr_image.copy()

        h, w = img.shape[:2]
        lr_h = h // self.scale_factor
        lr_w = w // self.scale_factor

        # 1. Point Spread Function (PSF) simulation via Gaussian Blur
        if self.blur_sigma > 0:
            k_size = self.blur_kernel_size if self.blur_kernel_size % 2 == 1 else self.blur_kernel_size + 1
            if img.ndim == 3:
                blurred = np.zeros_like(img)
                for c in range(img.shape[2]):
                    blurred[:, :, c] = scipy.ndimage.gaussian_filter(img[:, :, c], sigma=self.blur_sigma)
            else:
                blurred = scipy.ndimage.gaussian_filter(img, sigma=self.blur_sigma)
        else:
            blurred = img

        # 2. Additive Sensor Noise (e.g. thermal or atmospheric noise)
        if self.noise_type == "gaussian" and self.noise_sigma > 0:
            noise = np.random.normal(0, self.noise_sigma, blurred.shape).astype(np.float32)
            degraded = blurred + noise
        elif self.noise_type == "poisson":
            # Poisson shot noise simulation
            vals = len(np.unique(blurred))
            vals = 2 ** np.ceil(np.log2(vals))
            noisy = np.random.poisson(blurred * vals) / float(vals)
            degraded = noisy.astype(np.float32)
        else:
            degraded = blurred

        # 3. Controlled Downsampling via Bicubic Interpolation
        if degraded.ndim == 3:
            lr_img = cv2.resize(degraded, (lr_w, lr_h), interpolation=cv2.INTER_CUBIC)
            if lr_img.ndim == 2:
                lr_img = np.expand_dims(lr_img, axis=-1)
        else:
            lr_img = cv2.resize(degraded, (lr_w, lr_h), interpolation=cv2.INTER_CUBIC)

        if is_chw:
            # Convert back to (C, H, W)
            lr_img = np.transpose(lr_img, (2, 0, 1))

        return np.ascontiguousarray(lr_img, dtype=np.float32)
