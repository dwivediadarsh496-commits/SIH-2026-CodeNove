import os
import sys
import argparse
import yaml
import torch
import pandas as pd

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from models.factory import create_model
from datasets.paired_dataset import create_dataloaders
from evaluation.benchmark import evaluate_bicubic_baseline, evaluate_model

def main():
    parser = argparse.ArgumentParser(description="GeoSR-AI Evaluation & Benchmark CLI")
    parser.add_argument("--config", type=str, default="configs/srcnn.yaml", help="Path to config YAML")
    parser.add_argument("--checkpoint", type=str, default=None, help="Path to model checkpoint")
    parser.add_argument("--benchmark", action="store_true", help="Run full model benchmark comparison")
    args = parser.parse_args()

    with open(args.config, "r") as f:
        config = yaml.safe_load(f)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    data_cfg = config["data"]
    scale_factor = data_cfg.get("scale_factor", 4)

    _, _, test_loader = create_dataloaders(
        data_dir=data_cfg.get("processed_data_dir", "data/processed"),
        batch_size=4,
        patch_size=data_cfg.get("patch_size", 128),
        scale_factor=scale_factor,
        num_workers=0
    )

    results = []

    # 1. Bicubic Baseline
    print("\nRunning Bicubic Interpolation Baseline evaluation...")
    bicubic_res = evaluate_bicubic_baseline(test_loader, scale_factor=scale_factor)
    results.append(bicubic_res)

    # 2. Model Checkpoint Evaluation
    if args.checkpoint and os.path.exists(args.checkpoint):
        model_name = config.get("model", {}).get("name", "srcnn")
        print(f"\nEvaluating trained model ({model_name}) from checkpoint: {args.checkpoint}...")
        model = create_model(model_name=model_name, scale_factor=scale_factor)
        ckpt = torch.load(args.checkpoint, map_location=device)
        model.load_state_dict(ckpt.get("model_state_dict", ckpt))

        model_res = evaluate_model(model=model, test_loader=test_loader, model_name=model_name.upper(), device=str(device), scale_factor=scale_factor)
        results.append(model_res)

    df_results = pd.DataFrame(results)
    print("\n" + "=" * 60)
    print("MODEL COMPARISON BENCHMARK TABLE")
    print("=" * 60)
    print(df_results.to_string(index=False))
    print("=" * 60 + "\n")

    os.makedirs("outputs/metrics", exist_ok=True)
    out_csv = "outputs/metrics/model_comparison.csv"
    df_results.to_csv(out_csv, index=False)
    print(f"Benchmark summary saved to: {out_csv}")

if __name__ == "__main__":
    main()
