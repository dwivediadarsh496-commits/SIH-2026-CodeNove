import os
import torch
import numpy as np
from typing import Dict, Any, Union, Optional
from geospatial.raster_loader import RasterLoader
from geospatial.metadata import GeoMetadata
from geospatial.geotiff_writer import GeoTIFFWriter
from preprocessing.normalization import SatelliteNormalizer
from models.factory import create_model
from uncertainty.estimator import UncertaintyEstimator

class GeoSRPredictor:
    """
    Python-only inference interface for GeoSR-AI engine.
    Returns dictionary with super-resolved spatial representation, uncertainty map, and metadata.
    """

    def __init__(self, checkpoint_path: Optional[str] = None, model_name: str = "srcnn", scale_factor: int = 4, device: str = "auto"):
        if device == "auto":
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = torch.device(device)

        self.scale_factor = scale_factor
        self.model = create_model(model_name=model_name, in_channels=3, out_channels=3, scale_factor=scale_factor).to(self.device)

        if checkpoint_path and os.path.exists(checkpoint_path):
            ckpt = torch.load(checkpoint_path, map_location=self.device)
            state_dict = ckpt.get("model_state_dict", ckpt)
            self.model.load_state_dict(state_dict)
            print(f"Loaded GeoSR model checkpoint from: {checkpoint_path}")

        self.model.eval()
        self.normalizer = SatelliteNormalizer(method="percentile")
        self.uncertainty_estimator = UncertaintyEstimator(self.model, num_mc_samples=4, device=str(self.device))

    def predict(self, input_raster: Union[str, np.ndarray], metadata: Optional[GeoMetadata] = None) -> Dict[str, Any]:
        """
        Executes super-resolution mapping inference.

        Returns:
            Dict containing:
              - sr_image: Super-resolved numpy array (C, H_sr, W_sr)
              - uncertainty: Normalized uncertainty map array (1, H_sr, W_sr)
              - metadata: Updated GeoMetadata object with scaled spatial transform.
        """
        if isinstance(input_raster, str):
            data, meta = RasterLoader.load_raster(input_raster)
        else:
            data = input_raster.copy()
            meta = metadata or GeoMetadata(height=data.shape[1], width=data.shape[2], count=data.shape[0], dtype=str(data.dtype))

        # Apply satellite normalization
        norm_data, stats = self.normalizer.normalize(data)
        tensor_lr = torch.from_numpy(norm_data).unsqueeze(0).to(self.device)

        with torch.no_grad():
            sr_tensor, uncertainty_tensor = self.uncertainty_estimator.estimate_uncertainty(tensor_lr)

        sr_norm = sr_tensor.squeeze(0).cpu().numpy()
        uncertainty_map = uncertainty_tensor.squeeze(0).cpu().numpy()

        # Denormalize output to original reflectance scale
        sr_image = self.normalizer.denormalize(sr_norm, stats)

        return {
            "sr_image": sr_image,
            "uncertainty": uncertainty_map,
            "metadata": meta
        }
