import os
import torch
import numpy as np
from typing import Dict, Any, Optional
from geospatial.raster_loader import RasterLoader
from geospatial.geotiff_writer import GeoTIFFWriter
from geospatial.metadata import GeoMetadata
from preprocessing.normalization import SatelliteNormalizer
from preprocessing.tiling import TileStitcher

class TiledInferenceEngine:
    """
    Tiled inference engine for processing large full-scene GeoTIFF rasters safely
    without exceeding GPU VRAM capacity. Uses cosine overlap blending to eliminate tile boundary seams.
    """

    def __init__(self, model: torch.nn.Module, scale_factor: int = 4, tile_size: int = 256, overlap: int = 32, device: str = "cpu"):
        self.model = model.to(device)
        self.model.eval()
        self.scale_factor = scale_factor
        self.tile_size = tile_size
        self.overlap = overlap
        self.device = device
        self.normalizer = SatelliteNormalizer(method="percentile")

    def process_scene(self, input_path: str, output_path: str, uncertainty_output_path: Optional[str] = None) -> str:
        """
        Executes tiled inference across large satellite scene GeoTIFF and writes super-resolved GeoTIFF.
        """
        data, meta = RasterLoader.load_raster(input_path)
        norm_data, stats = self.normalizer.normalize(data)

        c, h_lr, w_lr = norm_data.shape
        h_sr, w_sr = h_lr * self.scale_factor, w_lr * self.scale_factor

        stitcher = TileStitcher(target_shape=(c, h_sr, w_sr), tile_size=self.tile_size * self.scale_factor, overlap=self.overlap * self.scale_factor)

        stride = self.tile_size - self.overlap

        with torch.no_grad():
            for y in range(0, h_lr, stride):
                for x in range(0, w_lr, stride):
                    y_end = min(y + self.tile_size, h_lr)
                    x_end = min(x + self.tile_size, w_lr)

                    tile_lr = norm_data[:, y:y_end, x:x_end]
                    tensor_lr = torch.from_numpy(tile_lr).unsqueeze(0).to(self.device)

                    sr_tile_tensor = self.model(tensor_lr)
                    sr_tile = sr_tile_tensor.squeeze(0).cpu().numpy()

                    # Position in target SR raster
                    y_sr, x_sr = y * self.scale_factor, x * self.scale_factor
                    stitcher.add_tile(sr_tile, y_sr, x_sr)

        sr_norm_full = stitcher.get_stitched_raster()
        sr_full = self.normalizer.denormalize(sr_norm_full, stats)

        # Write output GeoTIFF with updated scale transform (e.g. 10m -> 2.5m spacing)
        saved_path = GeoTIFFWriter.save_geotiff(
            data=sr_full,
            output_path=output_path,
            metadata=meta,
            scale_factor=self.scale_factor
        )

        return saved_path
