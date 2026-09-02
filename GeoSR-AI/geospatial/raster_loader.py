import os
import numpy as np
from typing import Tuple, List, Optional, Union
from geospatial.metadata import GeoMetadata

try:
    import rasterio
    RASTERIO_AVAILABLE = True
except ImportError:
    RASTERIO_AVAILABLE = False

from PIL import Image
import cv2

class RasterLoader:
    """
    Geospatial image loader supporting multi-band GeoTIFF, JP2, and standard raster formats.
    Preserves spatial reference metadata, nodata masks, and coordinate transforms.
    """

    @staticmethod
    def load_raster(
        path: str,
        band_indices: Optional[List[int]] = None
    ) -> Tuple[np.ndarray, GeoMetadata]:
        """
        Loads a single multi-band raster file.

        Args:
            path: Absolute or relative file path.
            band_indices: Optional 1-based list of band indices to read (e.g. [1, 2, 3]).

        Returns:
            Tuple of (data_array [C, H, W], GeoMetadata).
        """
        if not os.path.exists(path):
            raise FileNotFoundError(f"Raster file not found: {path}")

        if RASTERIO_AVAILABLE and (path.lower().endswith(('.tif', '.tiff', '.jp2')) or 'SAFE' in path):
            try:
                with rasterio.open(path) as src:
                    if band_indices is None:
                        data = src.read()  # Shape: (C, H, W)
                    else:
                        data = src.read(band_indices)

                    transform = src.transform
                    pixel_size = (abs(transform.a), abs(transform.e)) if transform else (10.0, 10.0)

                    meta = GeoMetadata(
                        height=src.height,
                        width=src.width,
                        count=data.shape[0],
                        dtype=str(data.dtype),
                        crs=src.crs,
                        transform=transform,
                        nodata=src.nodata,
                        pixel_size=pixel_size,
                        file_path=path
                    )
                    return data.astype(np.float32), meta
            except Exception as e:
                # Fallback to OpenCV/Pillow if rasterio fails on non-geospatial image
                pass

        # Fallback loader for standard images (PNG, JPG, BMP, WebP, etc.)
        try:
            pil_img = Image.open(path).convert("RGB")
            img = np.array(pil_img)  # Shape: (H, W, 3)
            data = np.transpose(img, (2, 0, 1)).astype(np.float32)
        except Exception:
            img = cv2.imread(path, cv2.IMREAD_COLOR)
            if img is not None:
                img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                data = np.transpose(img, (2, 0, 1)).astype(np.float32)
            else:
                raise ValueError(f"Unable to decode image from {path}")

        meta = GeoMetadata(
            height=data.shape[1],
            width=data.shape[2],
            count=data.shape[0],
            dtype=str(data.dtype),
            pixel_size=(10.0, 10.0),
            file_path=path
        )
        return data, meta


    @staticmethod
    def load_sentinel2_rgb(
        b04_path: str,
        b03_path: str,
        b02_path: str
    ) -> Tuple[np.ndarray, GeoMetadata]:
        """
        Stack separate Sentinel-2 single-band files (B4, B3, B2) into an RGB tensor (3, H, W).
        """
        r, meta = RasterLoader.load_raster(b04_path)
        g, _ = RasterLoader.load_raster(b03_path)
        b, _ = RasterLoader.load_raster(b02_path)

        rgb = np.concatenate([r[:1], g[:1], b[:1]], axis=0) # (3, H, W)
        meta.count = 3
        meta.band_names = ["B04_Red", "B03_Green", "B02_Blue"]
        return rgb, meta
