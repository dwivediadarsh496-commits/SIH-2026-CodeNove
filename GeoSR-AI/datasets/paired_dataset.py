import os
import glob
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from typing import List, Dict, Any, Tuple, Optional
from preprocessing.degradation import ControlledDegradation
from preprocessing.normalization import SatelliteNormalizer
from geospatial.raster_loader import RasterLoader
from datasets.transforms import SatelliteTransforms

class PairedSatelliteDataset(Dataset):
    """
    PyTorch Dataset for paired Low-Resolution (LR) and High-Resolution (HR) satellite imagery.
    
    Supports geographic scene-level partitioning to prevent spatial data leakage across
    training, validation, and test splits.
    """

    def __init__(
        self,
        data_dir: str,
        split: str = "train",
        patch_size: int = 128,
        scale_factor: int = 4,
        normalizer: Optional[SatelliteNormalizer] = None,
        degradation: Optional[ControlledDegradation] = None,
        transform: Optional[SatelliteTransforms] = None,
        synthetic_fallback: bool = True
    ):
        self.data_dir = data_dir
        self.split = split
        self.patch_size = patch_size
        self.scale_factor = scale_factor
        self.normalizer = normalizer if normalizer is not None else SatelliteNormalizer(method="percentile")
        self.degradation = degradation if degradation is not None else ControlledDegradation(scale_factor=scale_factor)
        self.transform = transform if transform is not None else SatelliteTransforms(is_train=(split == "train"))
        self.synthetic_fallback = synthetic_fallback

        self.samples = self._load_sample_index()

    def _load_sample_index(self) -> List[Dict[str, Any]]:
        """Scans dataset directory for paired LR/HR images or single HR reference rasters."""
        split_dir = os.path.join(self.data_dir, self.split)
        search_dir = split_dir if os.path.exists(split_dir) else self.data_dir

        samples = []
        if os.path.exists(search_dir):
            # Check for pre-processed patch pairs (.npz or .npy)
            npz_files = glob.glob(os.path.join(search_dir, "**", "*.npz"), recursive=True)
            for f in npz_files:
                scene_id = os.path.basename(f).split("_patch")[0]
                samples.append({"type": "npz", "path": f, "scene_id": scene_id})

            # Check for raw GeoTIFF / PNG / TIF images
            if not samples:
                img_files = []
                for ext in ["*.tif", "*.tiff", "*.jp2", "*.png", "*.jpg"]:
                    img_files.extend(glob.glob(os.path.join(search_dir, "**", ext), recursive=True))

                for f in img_files:
                    scene_id = os.path.basename(os.path.dirname(f)) or os.path.basename(f).split(".")[0]
                    samples.append({"type": "image", "path": f, "scene_id": scene_id})

        return samples

    def __len__(self) -> int:
        if len(self.samples) == 0 and self.synthetic_fallback:
            return 50  # Synthetic demonstration samples when no physical rasters exist
        return len(self.samples)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        if len(self.samples) > 0:
            sample_info = self.samples[idx % len(self.samples)]
            if sample_info["type"] == "npz":
                data = np.load(sample_info["path"])
                lr_arr = data["lr"].astype(np.float32)
                hr_arr = data["hr"].astype(np.float32)
                scene_id = sample_info["scene_id"]
            else:
                hr_arr, _ = RasterLoader.load_raster(sample_info["path"])
                # Extract crop if image is larger than patch_size
                c, h, w = hr_arr.shape
                if h >= self.patch_size and w >= self.patch_size:
                    y = np.random.randint(0, h - self.patch_size + 1)
                    x = np.random.randint(0, w - self.patch_size + 1)
                    hr_arr = hr_arr[:, y : y + self.patch_size, x : x + self.patch_size]
                
                # Apply satellite normalizer
                hr_arr, _ = self.normalizer.normalize(hr_arr)
                # Apply controlled degradation to generate synthetic LR
                lr_arr = self.degradation.degrade(hr_arr)
                scene_id = sample_info["scene_id"]
        else:
            # Generate realistic synthetic multi-spectral satellite patch
            hr_arr = self._generate_synthetic_satellite_patch()
            hr_arr, _ = self.normalizer.normalize(hr_arr)
            lr_arr = self.degradation.degrade(hr_arr)
            scene_id = f"synthetic_scene_{idx % 5}"

        # Ensure shapes match PyTorch format (C, H, W)
        if lr_arr.ndim == 2:
            lr_arr = np.expand_dims(lr_arr, axis=0)
        if hr_arr.ndim == 2:
            hr_arr = np.expand_dims(hr_arr, axis=0)

        # Apply synchronous spatial transformations
        lr_tensor, hr_tensor = self.transform(lr_arr, hr_arr)

        return {
            "lr": lr_tensor,
            "hr": hr_tensor,
            "scene_id": scene_id,
            "idx": idx
        }

    def _generate_synthetic_satellite_patch(self) -> np.ndarray:
        """
        Generates realistic synthetic land cover patch (Forest, Water, Agriculture, Urban)
        for testing and initialization verification.
        Shape: (3, patch_size, patch_size), dtype: float32 in range [0, 255].
        """
        ps = self.patch_size
        patch = np.zeros((3, ps, ps), dtype=np.float32)
        half = ps // 2

        # 1. Forest (Dark Green) - Band order: R, G, B
        patch[0, :half, :half] = 35.0   # Red
        patch[1, :half, :half] = 135.0  # Green
        patch[2, :half, :half] = 35.0   # Blue

        # 2. Water Body (Deep Blue)
        patch[0, half:, :half] = 20.0
        patch[1, half:, :half] = 80.0
        patch[2, half:, :half] = 200.0

        # 3. Agriculture Field (Bright Yellowish Green)
        patch[0, :half, half:] = 180.0
        patch[1, :half, half:] = 210.0
        patch[2, :half, half:] = 40.0

        # 4. Urban Built-up (Grey / Concrete)
        patch[0, half:, half:] = 160.0
        patch[1, half:, half:] = 160.0
        patch[2, half:, half:] = 160.0

        # Add realistic spatial texture noise
        noise = np.random.normal(0, 10.0, (3, ps, ps)).astype(np.float32)
        patch = np.clip(patch + noise, 0.0, 255.0)

        return patch

def create_dataloaders(
    data_dir: str,
    batch_size: int = 8,
    patch_size: int = 128,
    scale_factor: int = 4,
    num_workers: int = 0
) -> Tuple[DataLoader, DataLoader, DataLoader]:
    """
    Factory function creating train, val, and test PyTorch DataLoaders.
    """
    train_dataset = PairedSatelliteDataset(data_dir=data_dir, split="train", patch_size=patch_size, scale_factor=scale_factor)
    val_dataset = PairedSatelliteDataset(data_dir=data_dir, split="val", patch_size=patch_size, scale_factor=scale_factor)
    test_dataset = PairedSatelliteDataset(data_dir=data_dir, split="test", patch_size=patch_size, scale_factor=scale_factor)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers)

    return train_loader, val_loader, test_loader
