import os
import sys
import argparse
import yaml
import torch

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from models.factory import create_model
from inference.tiled_inference import TiledInferenceEngine
from inference.predictor import GeoSRPredictor

def main():
    parser = argparse.ArgumentParser(description="GeoSR-AI GeoTIFF Super-Resolution Inference CLI")
    parser.add_argument("--input", type=str, required=True, help="Path to input GeoTIFF satellite image")
    parser.add_argument("--checkpoint", type=str, required=False, default=None, help="Path to trained model checkpoint")
    parser.add_argument("--output", type=str, default="outputs/predictions/sr_output.tif", help="Path to output super-resolved GeoTIFF")
    parser.add_argument("--config", type=str, default="configs/base.yaml", help="Path to YAML config file")
    args = parser.parse_args()

    with open(args.config, "r") as f:
        config = yaml.safe_load(f)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    scale_factor = config.get("data", {}).get("scale_factor", 4)
    tile_size = config.get("inference", {}).get("tile_size", 256)
    overlap = config.get("inference", {}).get("overlap", 32)

    model_name = config.get("model", {}).get("name", "srcnn")
    model = create_model(model_name=model_name, scale_factor=scale_factor)

    if args.checkpoint and os.path.exists(args.checkpoint):
        ckpt = torch.load(args.checkpoint, map_location=device)
        model.load_state_dict(ckpt.get("model_state_dict", ckpt))
        print(f"Loaded checkpoint from {args.checkpoint}")

    print(f"\nRunning Tiled Super-Resolution Inference on: {args.input}")
    print(f"Target: <4m spatial representation (Scale factor: x{scale_factor})")

    engine = TiledInferenceEngine(model=model, scale_factor=scale_factor, tile_size=tile_size, overlap=overlap, device=device)
    saved_path = engine.process_scene(input_path=args.input, output_path=args.output)

    print(f"\nSuper-Resolution GeoTIFF saved to: {saved_path}\n")

if __name__ == "__main__":
    main()
