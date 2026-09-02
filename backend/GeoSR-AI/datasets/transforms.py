import random
import torch
import numpy as np
from typing import Tuple, Union

class SatelliteTransforms:
    """
    Synchronous spatial augmentations for paired LR and HR satellite image tensors/arrays.
    Preserves spectral band order and channel alignment.
    """

    def __init__(self, is_train: bool = True, random_flip: bool = True, random_rotate: bool = True):
        self.is_train = is_train
        self.random_flip = random_flip
        self.random_rotate = random_rotate

    def __call__(
        self,
        lr: Union[np.ndarray, torch.Tensor],
        hr: Union[np.ndarray, torch.Tensor]
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Applies identical spatial transformations to LR and HR patches.

        Args:
            lr: LR patch array/tensor (C, H_lr, W_lr).
            hr: HR patch array/tensor (C, H_hr, W_hr).

        Returns:
            Tuple of (lr_tensor, hr_tensor) of shape (C, H, W).
        """
        is_torch_lr = isinstance(lr, torch.Tensor)
        lr_arr = lr.numpy() if is_torch_lr else lr.copy()
        hr_arr = hr.numpy() if isinstance(hr, torch.Tensor) else hr.copy()

        if self.is_train:
            # Horizontal Flip
            if self.random_flip and random.random() < 0.5:
                lr_arr = np.flip(lr_arr, axis=-1).copy()
                hr_arr = np.flip(hr_arr, axis=-1).copy()

            # Vertical Flip
            if self.random_flip and random.random() < 0.5:
                lr_arr = np.flip(lr_arr, axis=-2).copy()
                hr_arr = np.flip(hr_arr, axis=-2).copy()

            # 90/180/270 degree Rotations
            if self.random_rotate:
                k = random.choice([0, 1, 2, 3])
                if k > 0:
                    lr_arr = np.rot90(lr_arr, k, axes=(-2, -1)).copy()
                    hr_arr = np.rot90(hr_arr, k, axes=(-2, -1)).copy()

        lr_tensor = torch.from_numpy(np.ascontiguousarray(lr_arr)).float()
        hr_tensor = torch.from_numpy(np.ascontiguousarray(hr_arr)).float()

        return lr_tensor, hr_tensor
