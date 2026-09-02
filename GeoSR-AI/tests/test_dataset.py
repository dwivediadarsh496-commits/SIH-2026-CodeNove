import os
import sys
import unittest
import torch

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from datasets.paired_dataset import PairedSatelliteDataset, create_dataloaders

class TestDatasetSubsystem(unittest.TestCase):

    def test_paired_dataset_shapes(self):
        dataset = PairedSatelliteDataset(data_dir="data/processed", patch_size=128, scale_factor=4, synthetic_fallback=True)
        self.assertGreater(len(dataset), 0)

        sample = dataset[0]
        self.assertIn("lr", sample)
        self.assertIn("hr", sample)

        lr = sample["lr"]
        hr = sample["hr"]

        self.assertIsInstance(lr, torch.Tensor)
        self.assertIsInstance(hr, torch.Tensor)
        self.assertEqual(lr.ndim, 3)
        self.assertEqual(hr.ndim, 3)

        # LR should be (3, 32, 32) when HR is (3, 128, 128) with scale_factor = 4
        self.assertEqual(lr.shape[0], 3)
        self.assertEqual(hr.shape[0], 3)
        self.assertEqual(hr.shape[1], 128)
        self.assertEqual(hr.shape[2], 128)
        self.assertEqual(lr.shape[1], 32)
        self.assertEqual(lr.shape[2], 32)

    def test_dataloader_batching(self):
        train_loader, val_loader, test_loader = create_dataloaders(
            data_dir="data/processed",
            batch_size=4,
            patch_size=128,
            scale_factor=4,
            num_workers=0
        )
        batch = next(iter(train_loader))
        self.assertEqual(batch["lr"].shape, (4, 3, 32, 32))
        self.assertEqual(batch["hr"].shape, (4, 3, 128, 128))

if __name__ == "__main__":
    unittest.main()
