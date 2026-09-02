import os
import sys
import argparse
import yaml
import torch

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from models.factory import create_model
from datasets.paired_dataset import create_dataloaders
from training.trainer import GeoSRTrainer

def main():
    parser = argparse.ArgumentParser(description="GeoSR-AI Model Training CLI")
    parser.add_argument("--config", type=str, default="configs/srcnn.yaml", help="Path to YAML config file")
    args = parser.parse_args()

    with open(args.config, "r") as f:
        config = yaml.safe_load(f)

    device = torch.device("cuda" if torch.cuda.is_available() and config.get("hardware", {}).get("device") != "cpu" else "cpu")
    print(f"Using device: {device}")

    data_cfg = config["data"]
    train_cfg = config["training"]
    model_cfg = config["model"]

    # Build DataLoaders
    train_loader, val_loader, test_loader = create_dataloaders(
        data_dir=data_cfg.get("processed_data_dir", "data/processed"),
        batch_size=train_cfg.get("batch_size", 8),
        patch_size=data_cfg.get("patch_size", 128),
        scale_factor=data_cfg.get("scale_factor", 4),
        num_workers=train_cfg.get("num_workers", 0)
    )

    # Build PyTorch Model
    model = create_model(
        model_name=model_cfg.get("name", "srcnn"),
        in_channels=model_cfg.get("in_channels", 3),
        out_channels=model_cfg.get("out_channels", 3),
        scale_factor=data_cfg.get("scale_factor", 4),
        num_features=model_cfg.get("num_features", 64),
        num_blocks=model_cfg.get("num_blocks", 16)
    ).to(device)

    # Initialize Trainer and execute fit loop
    trainer = GeoSRTrainer(model=model, config=config, device=device)
    trainer.fit(train_loader=train_loader, val_loader=val_loader, epochs=train_cfg.get("epochs", 5))

if __name__ == "__main__":
    main()
