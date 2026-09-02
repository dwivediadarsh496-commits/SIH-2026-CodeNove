import numpy as np
from typing import List, Tuple, Dict, Any, Generator

class RasterTiler:
    """
    Extracts spatial tiles/patches from large satellite rasters with configurable tile size and stride/overlap.
    """

    def __init__(self, tile_size: int = 128, stride: int = 64):
        self.tile_size = tile_size
        self.stride = stride

    def extract_patches(
        self,
        raster: np.ndarray
    ) -> Tuple[List[np.ndarray], List[Dict[str, int]]]:
        """
        Extracts overlapping patches from input raster array of shape (C, H, W) or (H, W, C).

        Returns:
            Tuple of (patches_list, coordinates_list).
        """
        is_chw = (raster.ndim == 3 and raster.shape[0] in [1, 3, 4])
        if is_chw:
            c, h, w = raster.shape
        else:
            h, w, c = raster.shape

        patches = []
        coords = []

        for y in range(0, h - self.tile_size + 1, self.stride):
            for x in range(0, w - self.tile_size + 1, self.stride):
                if is_chw:
                    patch = raster[:, y : y + self.tile_size, x : x + self.tile_size]
                else:
                    patch = raster[y : y + self.tile_size, x : x + self.tile_size, :]

                patches.append(patch)
                coords.append({"y": y, "x": x, "height": self.tile_size, "width": self.tile_size})

        return patches, coords

class TileStitcher:
    """
    Stitches inferenced image tiles back into full large-scene rasters using smooth
    cosine blending weights to eliminate boundary seam artifacts.
    """

    def __init__(self, target_shape: Tuple[int, int, int], tile_size: int, overlap: int):
        """
        Args:
            target_shape: (C, H, W) of the full target raster.
            tile_size: Tile spatial dimension.
            overlap: Overlap pixel width.
        """
        self.c, self.h, self.w = target_shape
        self.tile_size = tile_size
        self.overlap = overlap
        self.output_buffer = np.zeros((self.c, self.h, self.w), dtype=np.float32)
        self.weight_buffer = np.zeros((1, self.h, self.w), dtype=np.float32)

        # Build 2D smooth cosine weight window for seamless tile blending
        self.tile_weight = self._create_cosine_window(tile_size, overlap)

    def _create_cosine_window(self, size: int, overlap: int) -> np.ndarray:
        """Generates a 2D cosine tapering window for seam-free overlap blending."""
        w_1d = np.ones(size, dtype=np.float32)
        if overlap > 0:
            ramp = 0.5 * (1 - np.cos(np.pi * np.arange(overlap) / float(overlap)))
            w_1d[:overlap] = ramp
            w_1d[-overlap:] = ramp[::-1]

        w_2d = np.outer(w_1d, w_1d)[None, :, :]  # Shape: (1, size, size)
        return w_2d

    def add_tile(self, tile: np.ndarray, y: int, x: int):
        """
        Adds predicted tile of shape (C, tile_size, tile_size) at position (y, x).
        """
        th, tw = tile.shape[1], tile.shape[2]
        weight = self.tile_weight[:, :th, :tw]

        self.output_buffer[:, y : y + th, x : x + tw] += tile * weight
        self.weight_buffer[:, y : y + th, x : x + tw] += weight

    def get_stitched_raster(self) -> np.ndarray:
        """Returns normalized full raster after blending all tile predictions."""
        mask = self.weight_buffer > 0
        result = np.where(mask, self.output_buffer / (self.weight_buffer + 1e-8), 0.0)
        return result.astype(np.float32)
