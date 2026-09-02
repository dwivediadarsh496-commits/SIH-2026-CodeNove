import dataclasses
from typing import Optional, List, Tuple, Dict, Any

@dataclasses.dataclass
class GeoMetadata:
    """
    Geospatial metadata container preserving coordinate reference system (CRS),
    affine transform, image dimensions, nodata values, and spatial resolution.
    """
    height: int
    width: int
    count: int                                    # Number of channels/bands
    dtype: str                                    # Array datatype string
    crs: Optional[Any] = None                     # Rasterio CRS or EPSG string
    transform: Optional[Any] = None               # Affine transform matrix
    nodata: Optional[float] = None                # Nodata invalid pixel value
    pixel_size: Tuple[float, float] = (10.0, 10.0) # (x_res, y_res) in CRS units
    band_names: Optional[List[str]] = None
    file_path: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        """Convert metadata to a serializable dictionary representation."""
        return {
            "height": self.height,
            "width": self.width,
            "count": self.count,
            "dtype": str(self.dtype),
            "crs": str(self.crs) if self.crs else None,
            "transform": [self.transform[i] for i in range(6)] if self.transform else None,
            "nodata": float(self.nodata) if self.nodata is not None else None,
            "pixel_size": self.pixel_size,
            "band_names": self.band_names,
            "file_path": self.file_path,
        }

    def scale_transform(self, scale_factor: int) -> Any:
        """
        Calculates updated Affine transform matrix scaled for spatial super-resolution.
        Target pixel spacing = Original pixel size / scale_factor.
        """
        if self.transform is None:
            return None
        import affine
        if isinstance(self.transform, affine.Affine):
            # Scale affine transform: a (pixel width) and e (pixel height) scaled by 1/scale_factor
            return affine.Affine(
                self.transform.a / float(scale_factor),
                self.transform.b,
                self.transform.c,
                self.transform.d,
                self.transform.e / float(scale_factor),
                self.transform.f
            )
        return self.transform
