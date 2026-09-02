import os
import sys
import unittest
import numpy as np
import tempfile

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from geospatial.metadata import GeoMetadata
from geospatial.raster_loader import RasterLoader
from geospatial.geotiff_writer import GeoTIFFWriter

class TestGeospatialSubsystem(unittest.TestCase):

    def test_metadata_scale_transform(self):
        meta = GeoMetadata(
            height=100,
            width=100,
            count=3,
            dtype="float32",
            pixel_size=(10.0, 10.0)
        )
        scaled_meta_dict = meta.to_dict()
        self.assertEqual(scaled_meta_dict["height"], 100)
        self.assertEqual(scaled_meta_dict["width"], 100)
        self.assertEqual(scaled_meta_dict["count"], 3)

    def test_raster_write_and_load(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            test_file = os.path.join(tmp_dir, "test_raster.tif")
            arr = np.random.uniform(0, 1, (3, 64, 64)).astype(np.float32)
            meta = GeoMetadata(height=64, width=64, count=3, dtype="float32")

            saved_path = GeoTIFFWriter.save_geotiff(arr, test_file, metadata=meta, scale_factor=4)
            self.assertTrue(os.path.exists(saved_path))

            loaded_arr, loaded_meta = RasterLoader.load_raster(saved_path)
            self.assertEqual(loaded_arr.shape[0], 3)
            self.assertEqual(loaded_arr.shape[1], 64)
            self.assertEqual(loaded_arr.shape[2], 64)

if __name__ == "__main__":
    unittest.main()
