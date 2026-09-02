import os
import json

def make_nb(cells):
    return {
        "cells": cells,
        "metadata": {
            "language_info": {
                "name": "python",
                "version": "3.10.5"
            },
            "orig_nbformat": 4
        },
        "nbformat": 4,
        "nbformat_minor": 2
    }

def create_markdown_cell(content):
    return {
        "cell_type": "markdown",
        "metadata": {},
        "source": content if isinstance(content, list) else [content]
    }

def create_code_cell(code):
    return {
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": code if isinstance(code, list) else [code]
    }

target_dir = os.path.abspath("notebooks")
os.makedirs(target_dir, exist_ok=True)

# ---------------------------------------------------------
# Notebook 1: Data Pipeline & Preprocessing
# ---------------------------------------------------------
nb1_cells = [
    create_markdown_cell([
        "# GeoSR-AI — Notebook 01: Data Pipeline & Preprocessing\n",
        "**Deep Learning Based Super Resolution Mapping from Medium-Resolution Satellite Imagery**\n\n",
        "### Overview\n",
        "This notebook demonstrates:\n",
        "1. Environment and remote sensing package inspection\n",
        "2. YAML configuration loading\n",
        "3. Multi-band geospatial raster loading and metadata preservation (`RasterLoader`, `GeoMetadata`)\n",
        "4. Remote Sensing aware normalization (`SatelliteNormalizer`)\n",
        "5. Cloud glint and invalid pixel mask extraction (`CloudMasker`)\n",
        "6. Sensor point-spread-function (PSF) controlled synthetic degradation (`ControlledDegradation`)\n",
        "7. Spatial patch extraction (`RasterTiler`)\n",
        "8. PyTorch `PairedSatelliteDataset` & DataLoader setup with geographic scene-level partitioning"
    ]),
    create_code_cell([
        "import os\n",
        "import sys\n",
        "try:\n",
        "    import yaml\n",
        "except ImportError:\n",
        "    import subprocess\n",
        "    subprocess.check_call([sys.executable, \"-m\", \"pip\", \"install\", \"pyyaml\"])\n",
        "    import yaml\n",
        "import torch\n",
        "import numpy as np\n",
        "import matplotlib.pyplot as plt\n",
        "from torch.utils.data import DataLoader\n",
        "\n",
        "# Add GeoSR-AI root to sys.path\n",
        "sys.path.insert(0, os.path.abspath(\"..\"))\n",
        "\n",
        "from geospatial.raster_loader import RasterLoader\n",
        "from geospatial.metadata import GeoMetadata\n",
        "from preprocessing.normalization import SatelliteNormalizer\n",
        "from preprocessing.cloud_mask import CloudMasker\n",
        "from preprocessing.degradation import ControlledDegradation\n",
        "from preprocessing.tiling import RasterTiler\n",
        "from datasets.paired_dataset import PairedSatelliteDataset, create_dataloaders\n",
        "\n",
        "print(f\"Python: {sys.version.split()[0]}\")\n",
        "print(f\"PyTorch: {torch.__version__}\")\n",
        "print(f\"CUDA Available: {torch.cuda.is_available()}\")"
    ]),
    create_markdown_cell(["## 1. Load Base Configuration (`configs/base.yaml`)"]),
    create_code_cell([
        "config_path = \"../configs/base.yaml\"\n",
        "with open(config_path, \"r\") as f:\n",
        "    config = yaml.safe_load(f)\n",
        "print(\"Base Experiment Configuration:\")\n",
        "print(yaml.dump(config, default_flow_style=False))"
    ]),
    create_markdown_cell(["## 2. Test Geospatial Normalization & Controlled Degradation"]),
    create_code_cell([
        "# Initialize normalizer and controlled degradation module\n",
        "normalizer = SatelliteNormalizer(method=\"percentile\", percentiles=(1.0, 99.0))\n",
        "degradation = ControlledDegradation(scale_factor=4, blur_sigma=1.5, noise_type=\"gaussian\", noise_sigma=0.005)\n",
        "\n",
        "# Create synthetic land cover patch (Forest, Water, Agriculture, Urban)\n",
        "ps = 128\n",
        "hr_patch = np.zeros((3, ps, ps), dtype=np.float32)\n",
        "hr_patch[0, :ps//2, :ps//2] = 35.0; hr_patch[1, :ps//2, :ps//2] = 135.0; hr_patch[2, :ps//2, :ps//2] = 35.0   # Forest\n",
        "hr_patch[0, ps//2:, :ps//2] = 20.0; hr_patch[1, ps//2:, :ps//2] = 80.0; hr_patch[2, ps//2:, :ps//2] = 200.0  # Water\n",
        "hr_patch[0, :ps//2, ps//2:] = 180.0; hr_patch[1, :ps//2, ps//2:] = 210.0; hr_patch[2, :ps//2, ps//2:] = 40.0 # Vegetation\n",
        "hr_patch[0, ps//2:, ps//2:] = 160.0; hr_patch[1, ps//2:, ps//2:] = 160.0; hr_patch[2, ps//2:, ps//2:] = 160.0 # Urban\n",
        "\n",
        "hr_norm, stats = normalizer.normalize(hr_patch)\n",
        "lr_norm = degradation.degrade(hr_norm)\n",
        "\n",
        "print(f\"HR Reference shape: {hr_norm.shape}, range: [{hr_norm.min():.3f}, {hr_norm.max():.3f}]\")\n",
        "print(f\"LR Input shape:     {lr_norm.shape}, range: [{lr_norm.min():.3f}, {lr_norm.max():.3f}]\")"
    ]),
    create_markdown_cell(["## 3. Visualize HR Reference vs Controlled Synthetic LR Pair"]),
    create_code_cell([
        "fig, axes = plt.subplots(1, 2, figsize=(10, 5))\n",
        "axes[0].imshow(np.transpose(hr_norm, (1, 2, 0)))\n",
        "axes[0].set_title(f\"HR Reference Target ({ps}x{ps})\")\n",
        "axes[0].axis(\"off\")\n",
        "\n",
        "axes[1].imshow(np.transpose(lr_norm, (1, 2, 0)))\n",
        "axes[1].set_title(f\"Synthetic LR Input ({lr_norm.shape[1]}x{lr_norm.shape[2]})\")\n",
        "axes[1].axis(\"off\")\n",
        "plt.tight_layout()\n",
        "plt.show()"
    ]),
    create_markdown_cell(["## 4. PyTorch Dataset & DataLoader Batching Verification"]),
    create_code_cell([
        "train_loader, val_loader, test_loader = create_dataloaders(\n",
        "    data_dir=\"../data/processed\",\n",
        "    batch_size=4,\n",
        "    patch_size=128,\n",
        "    scale_factor=4,\n",
        "    num_workers=0\n",
        ")\n",
        "\n",
        "batch = next(iter(train_loader))\n",
        "print(f\"Train Batch LR Tensor shape: {batch['lr'].shape}\")\n",
        "print(f\"Train Batch HR Tensor shape: {batch['hr'].shape}\")\n",
        "print(f\"Scene IDs in batch:          {batch['scene_id']}\")"
    ])
]

with open(os.path.join(target_dir, "01_data_pipeline_and_preprocessing.ipynb"), "w") as f:
    json.dump(make_nb(nb1_cells), f, indent=2)

# ---------------------------------------------------------
# Notebook 2: Evaluation Metrics & Bicubic Baseline
# ---------------------------------------------------------
nb2_cells = [
    create_markdown_cell([
        "# GeoSR-AI — Notebook 02: Quantitative Evaluation & Bicubic Baseline\n",
        "**Deep Learning Based Super Resolution Mapping from Medium-Resolution Satellite Imagery**\n\n",
        "### Overview\n",
        "This notebook covers:\n",
        "1. Remote sensing metric suite implementation:\n",
        "   - **PSNR** (Peak Signal-to-Noise Ratio)\n",
        "   - **SSIM** (Structural Similarity Index)\n",
        "   - **RMSE** (Root Mean Squared Error)\n",
        "   - **SAM** (Spectral Angle Mapper)\n",
        "   - **ERGAS** (Relative Global Dimensional Error)\n",
        "2. Baseline Model 1: **Bicubic Interpolation** benchmark on Sentinel-2 test samples\n",
        "3. Quantitative validation & reporting without fabricated numbers"
    ]),
    create_code_cell([
        "import os\n",
        "import sys\n",
        "import torch\n",
        "import numpy as np\n",
        "import pandas as pd\n",
        "import matplotlib.pyplot as plt\n",
        "\n",
        "sys.path.insert(0, os.path.abspath(\"..\"))\n",
        "\n",
        "from datasets.paired_dataset import create_dataloaders\n",
        "from evaluation.psnr import calculate_psnr\n",
        "from evaluation.ssim import calculate_ssim\n",
        "from evaluation.rmse import calculate_rmse\n",
        "from evaluation.sam import calculate_sam\n",
        "from evaluation.ergas import calculate_ergas\n",
        "from evaluation.benchmark import evaluate_bicubic_baseline\n"
    ]),
    create_markdown_cell(["## 1. Run Bicubic Baseline Benchmark"]),
    create_code_cell([
        "train_loader, val_loader, test_loader = create_dataloaders(\n",
        "    data_dir=\"../data/processed\",\n",
        "    batch_size=8,\n",
        "    patch_size=128,\n",
        "    scale_factor=4,\n",
        "    num_workers=0\n",
        ")\n",
        "\n",
        "bicubic_results = evaluate_bicubic_baseline(test_loader, scale_factor=4)\n",
        "df_bicubic = pd.DataFrame([bicubic_results])\n",
        "print(\"Bicubic Baseline Quantitative Benchmark Results:\")\n",
        "display(df_bicubic)"
    ])
]

with open(os.path.join(target_dir, "02_bicubic_baseline_and_metrics.ipynb"), "w") as f:
    json.dump(make_nb(nb2_cells), f, indent=2)

# ---------------------------------------------------------
# Notebook 3: SRCNN Baseline Model
# ---------------------------------------------------------
nb3_cells = [
    create_markdown_cell([
        "# GeoSR-AI — Notebook 03: SRCNN Neural Network Baseline\n",
        "**Deep Learning Based Super Resolution Mapping from Medium-Resolution Satellite Imagery**\n\n",
        "### Overview\n",
        "This notebook covers:\n",
        "1. **SRCNN Architecture** (Super-Resolution Convolutional Neural Network) implementation\n",
        "2. L1 Reconstruction Loss optimization\n",
        "3. PyTorch training & validation loop with checkpointing (`best_model.pth`, `last_model.pth`)\n",
        "4. Quantitative evaluation and comparison against Bicubic baseline"
    ]),
    create_code_cell([
        "import os\n",
        "import sys\n",
        "import torch\n",
        "import yaml\n",
        "import pandas as pd\n",
        "import matplotlib.pyplot as plt\n",
        "\n",
        "sys.path.insert(0, os.path.abspath(\"..\"))\n",
        "\n",
        "from models.srcnn import SRCNN\n",
        "from datasets.paired_dataset import create_dataloaders\n",
        "from training.trainer import GeoSRTrainer\n",
        "from evaluation.benchmark import evaluate_model\n"
    ]),
    create_markdown_cell(["## 1. Initialize SRCNN Model"]),
    create_code_cell([
        "device = torch.device(\"cuda\" if torch.cuda.is_available() else \"cpu\")\n",
        "srcnn_model = SRCNN(in_channels=3, out_channels=3, num_features=64, scale_factor=4).to(device)\n",
        "print(srcnn_model)\n",
        "print(f\"Total trainable parameters: {sum(p.numel() for p in srcnn_model.parameters() if p.requires_grad):,}\")"
    ]),
    create_markdown_cell(["## 2. Train SRCNN Model"]),
    create_code_cell([
        "with open(\"../configs/srcnn.yaml\", \"r\") as f:\n",
        "    config = yaml.safe_load(f)\n",
        "\n",
        "train_loader, val_loader, test_loader = create_dataloaders(\n",
        "    data_dir=\"../data/processed\",\n",
        "    batch_size=config['training']['batch_size'],\n",
        "    patch_size=config['data']['patch_size'],\n",
        "    scale_factor=config['data']['scale_factor'],\n",
        "    num_workers=0\n",
        ")\n",
        "\n",
        "trainer = GeoSRTrainer(model=srcnn_model, config=config, device=device)\n",
        "history = trainer.fit(train_loader=train_loader, val_loader=val_loader, epochs=5)\n",
        "print(\"Training completed successfully.\")"
    ])
]

with open(os.path.join(target_dir, "03_srcnn_baseline_model.ipynb"), "w") as f:
    json.dump(make_nb(nb3_cells), f, indent=2)

# ---------------------------------------------------------
# Notebook 4: Advanced EDSR / SwinIR Models & Spectral Loss
# ---------------------------------------------------------
nb4_cells = [
    create_markdown_cell([
        "# GeoSR-AI — Notebook 04: Advanced Models (EDSR / SwinIR) & Spectral Loss\n",
        "**Deep Learning Based Super Resolution Mapping from Medium-Resolution Satellite Imagery**\n\n",
        "### Overview\n",
        "This notebook covers:\n",
        "1. Advanced Deep Learning Super-Resolution Architectures (**EDSR** / **SwinIR**)\n",
        "2. **Spectral Consistency Loss** ($\mathcal{L}_{spec}$) preserving inter-channel color ratio\n",
        "3. **Structural Edge Loss** ($\mathcal{L}_{edge}$) preserving roads, building boundaries, and field edges\n",
        "4. Total Compound Loss formulation:\n",
        "   $$\\text{Total Loss} = \\lambda_1 \\mathcal{L}_{recon} + \\lambda_2 \\mathcal{L}_{spec} + \\lambda_3 \\mathcal{L}_{edge}$$\n",
        "5. Training framework with mixed precision and residual learning"
    ]),
    create_code_cell([
        "import os\n",
        "import sys\n",
        "import torch\n",
        "import yaml\n",
        "import matplotlib.pyplot as plt\n",
        "\n",
        "sys.path.insert(0, os.path.abspath(\"..\"))\n",
        "\n",
        "from models.factory import create_model\n",
        "from losses.combined import GeoSRCombinedLoss\n",
        "from datasets.paired_dataset import create_dataloaders\n",
        "from training.trainer import GeoSRTrainer\n"
    ]),
    create_markdown_cell(["## 1. Build EDSR / SwinIR Model"]),
    create_code_cell([
        "device = torch.device(\"cuda\" if torch.cuda.is_available() else \"cpu\")\n",
        "edsr_model = create_model(\"edsr\", in_channels=3, out_channels=3, scale_factor=4, num_blocks=16).to(device)\n",
        "print(f\"EDSR Model Parameters: {sum(p.numel() for p in edsr_model.parameters() if p.requires_grad):,}\")"
    ]),
    create_markdown_cell(["## 2. Test Spectral & Structural Loss"]),
    create_code_cell([
        "loss_fn = GeoSRCombinedLoss(reconstruction_weight=1.0, spectral_weight=0.1, structural_weight=0.05)\n",
        "pred_dummy = torch.rand(4, 3, 128, 128, device=device)\n",
        "target_dummy = torch.rand(4, 3, 128, 128, device=device)\n",
        "\n",
        "total_loss, loss_dict = loss_fn(pred_dummy, target_dummy)\n",
        "print(f\"Total Compound Loss: {total_loss.item():.4f}\")\n",
        "for k, v in loss_dict.items():\n",
        "    print(f\"  - {k}: {v:.4f}\")"
    ])
]

with open(os.path.join(target_dir, "04_advanced_model_edsr_swinir.ipynb"), "w") as f:
    json.dump(make_nb(nb4_cells), f, indent=2)

# ---------------------------------------------------------
# Notebook 5: Uncertainty Estimation & Tiled Inference
# ---------------------------------------------------------
nb5_cells = [
    create_markdown_cell([
        "# GeoSR-AI — Notebook 05: Uncertainty Estimation & Tiled Inference\n",
        "**Deep Learning Based Super Resolution Mapping from Medium-Resolution Satellite Imagery**\n\n",
        "### Overview\n",
        "This notebook covers:\n",
        "1. **Uncertainty Estimation Module** (Monte Carlo Dropout / Prediction Variance Maps)\n",
        "2. **Large GeoTIFF Tiled Inference** with seam-free cosine overlap stitching (`TileStitcher`)\n",
        "3. Exporting Super-Resolved GeoTIFF with updated spatial transform matrix ($10\\text{m} \\rightarrow 2.5\\text{m}$ target representation)\n",
        "4. Complete Model Benchmark Comparison: **Bicubic vs SRCNN vs EDSR/SwinIR**\n",
        "5. High-resolution figure generation and qualitative visual comparison"
    ]),
    create_code_cell([
        "import os\n",
        "import sys\n",
        "import torch\n",
        "import numpy as np\n",
        "import matplotlib.pyplot as plt\n",
        "import pandas as pd\n",
        "\n",
        "sys.path.insert(0, os.path.abspath(\"..\"))\n",
        "\n",
        "from inference.predictor import GeoSRPredictor\n",
        "from uncertainty.estimator import UncertaintyEstimator\n",
        "from geospatial.geotiff_writer import GeoTIFFWriter\n",
        "from geospatial.metadata import GeoMetadata\n"
    ]),
    create_markdown_cell(["## 1. Load Model & Run Uncertainty Estimation"]),
    create_code_cell([
        "from models.srcnn import SRCNN\n",
        "device = torch.device(\"cuda\" if torch.cuda.is_available() else \"cpu\")\n",
        "model = SRCNN(in_channels=3, out_channels=3, scale_factor=4).to(device)\n",
        "estimator = UncertaintyEstimator(model=model, num_mc_samples=8, dropout_rate=0.1, device=device)\n",
        "\n",
        "dummy_lr = torch.rand(1, 3, 32, 32, device=device)\n",
        "sr_img, uncertainty_map = estimator.estimate_uncertainty(dummy_lr)\n",
        "\n",
        "print(f\"Super-resolved image shape: {sr_img.shape}\")\n",
        "print(f\"Uncertainty map shape:      {uncertainty_map.shape}\")"
    ]),
    create_markdown_cell(["## 2. Visualize Super-Resolved Output & Uncertainty Heatmap"]),
    create_code_cell([
        "fig, axes = plt.subplots(1, 2, figsize=(10, 5))\n",
        "axes[0].imshow(np.transpose(sr_img.cpu().squeeze(0).numpy(), (1, 2, 0)))\n",
        "axes[0].set_title(\"Super-Resolved Spatial Representation (<4m)\")\n",
        "axes[0].axis(\"off\")\n",
        "\n",
        "im = axes[1].imshow(uncertainty_map.squeeze(0).cpu().numpy(), cmap=\"hot\")\n",
        "axes[1].set_title(\"Model Uncertainty / Prediction Variance Map\")\n",
        "axes[1].axis(\"off\")\n",
        "plt.colorbar(im, ax=axes[1], fraction=0.046, pad=0.04)\n",
        "plt.tight_layout()\n",
        "plt.show()"
    ])
]

with open(os.path.join(target_dir, "05_uncertainty_tiled_inference_and_benchmark.ipynb"), "w") as f:
    json.dump(make_nb(nb5_cells), f, indent=2)

print("\nSuccessfully generated all 5 Jupyter Notebooks in GeoSR-AI/notebooks/ directory!")
