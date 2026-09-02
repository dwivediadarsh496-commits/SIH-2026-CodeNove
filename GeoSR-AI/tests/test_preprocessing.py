import os
import sys
import unittest
import numpy as np

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from preprocessing.normalization import SatelliteNormalizer
from preprocessing.cloud_mask import CloudMasker
from preprocessing.degradation import ControlledDegradation
from preprocessing.tiling import RasterTiler, TileStitcher

class TestPreprocessingSubsystem(unittest.TestCase):

    def test_normalization_and_denormalization(self):
        norm = SatelliteNormalizer(method="percentile", percentiles=(1.0, 99.0))
        arr = np.random.uniform(0, 5000, (3, 100, 100)).astype(np.float32)

        norm_arr, stats = norm.normalize(arr)
        self.assertTrue(norm_arr.min() >= 0.0)
        self.assertTrue(norm_arr.max() <= 1.0)

        denorm_arr = norm.denormalize(norm_arr, stats)
        self.assertEqual(denorm_arr.shape, arr.shape)

    def test_controlled_degradation_shapes(self):
        deg = ControlledDegradation(scale_factor=4, blur_sigma=1.5)
        hr_arr = np.random.uniform(0, 1, (3, 128, 128)).astype(np.float32)
        lr_arr = deg.degrade(hr_arr)

        self.assertEqual(lr_arr.shape[0], 3)
        self.assertEqual(lr_arr.shape[1], 32)
        self.assertEqual(lr_arr.shape[2], 32)

    def test_tiling_and_stitching(self):
        raster = np.random.uniform(0, 1, (3, 256, 256)).astype(np.float32)
        tiler = RasterTiler(tile_size=128, stride=64)
        patches, coords = tiler.extract_patches(raster)

        self.assertGreater(len(patches), 0)
        self.assertEqual(patches[0].shape, (3, 128, 128))

        stitcher = TileStitcher(target_shape=(3, 256, 256), tile_size=128, overlap=64)
        for patch, c in zip(patches, coords):
            stitcher.add_tile(patch, c["y"], c["x"])

        reconstructed = stitcher.get_stitched_raster()
        self.assertEqual(reconstructed.shape, (3, 256, 256))

if __name__ == "__main__":
    unittest.main()
