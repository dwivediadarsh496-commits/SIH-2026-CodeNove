import os
import numpy as np
from typing import Optional, Union
from geospatial.metadata import GeoMetadata

try:
    import rasterio
    from rasterio.transform import Affine
    RASTERIO_AVAILABLE = True
except ImportError:
    RASTERIO_AVAILABLE = False

from PIL import Image
import cv2

class GeoTIFFWriter:
    """
    Export module for writing super-resolved rasters to GeoTIFF format,
    updating the geotransform and pixel size while preserving spatial projection details.
    """

    @staticmethod
    def save_geotiff(
        data: np.ndarray,
        output_path: str,
        metadata: Optional[GeoMetadata] = None,
        scale_factor: int = 1,
        nodata_val: Optional[float] = None
    ) -> str:
        """
        Saves array (C, H, W) or (H, W, C) to a GeoTIFF raster.

        Args:
            data: Data array of shape (C, H, W) or (H, W, C).
            output_path: Target filename.
            metadata: Associated GeoMetadata object.
            scale_factor: Scale factor used in super-resolution (updates affine pixel spacing).
            nodata_val: Optional nodata value to store in header.

        Returns:
            Absolute path to saved GeoTIFF file.
        """
        os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)

        if data.ndim == 2:
            data = np.expand_dims(data, axis=0) # (1, H, W)
        elif data.ndim == 3 and data.shape[2] in [1, 3, 4] and data.shape[0] not in [1, 3, 4]:
            # Convert (H, W, C) -> (C, H, W)
            data = np.transpose(data, (2, 0, 1))

        count, height, width = data.shape

        if RASTERIO_AVAILABLE and metadata is not None and metadata.crs is not None:
            # Update transform for scale_factor upsampling
            new_transform = metadata.scale_transform(scale_factor)

            profile = {
                "driver": "GTiff",
                "height": height,
                "width": width,
                "count": count,
                "dtype": data.dtype,
                "crs": metadata.crs,
                "transform": new_transform,
                "nodata": nodata_val if nodata_val is not None else metadata.nodata,
                "compress": "lzw"
            }

            with rasterio.open(output_path, "w", **profile) as dst:
                dst.write(data)

            return os.path.abspath(output_path)

        # Fallback for standard rasters (PNG/TIF without CRS)
        # Convert (C, H, W) to (H, W, C) for Image/OpenCV save
        out_img = np.transpose(data, (1, 2, 0))
        if out_img.shape[2] == 1:
            out_img = out_img.squeeze(-1)

        if out_img.dtype != np.uint8 and out_img.max() <= 1.0:
            out_img = (np.clip(out_img, 0, 1) * 255.0).astype(np.uint8)
        else:
            out_img = np.clip(out_img, 0, 255).astype(np.uint8)

        if out_img.ndim == 3 and out_img.shape[2] == 3:
            # Convert RGB -> BGR for OpenCV
            out_img_bgr = cv2.cvtColor(out_img, cv2.COLOR_RGB2BGR)
            cv2.imwrite(output_path, out_img_bgr)
        else:
            Image.fromarray(out_img).save(output_path)

        return os.path.abspath(output_path)
