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

target_dir = os.path.abspath("../ml_dl_models")
os.makedirs(target_dir, exist_ok=True)

# 01 Dataset Loader Notebook
nb1 = [
    create_markdown_cell("# GeoSR-AI — SRM Satellite Dataset Loader Notebook"),
    create_code_cell([
        "import os\n",
        "import glob\n",
        "import numpy as np\n",
        "from PIL import Image\n",
        "import torch\n",
        "from torch.utils.data import Dataset, DataLoader\n\n",
        "# Import dataset class\n",
        "from dataset import SatelliteSRMDataset, create_srm_dataloader\n\n",
        "loader = create_srm_dataloader(image_dir='../GeoSR-AI/data/processed', batch_size=4, scale_factor=4)\n",
        "batch = next(iter(loader))\n",
        "print('LR Tensor batch shape:', batch['lr'].shape)\n",
        "print('HR Tensor batch shape:', batch['hr'].shape)"
    ])
]
with open(os.path.join(target_dir, "01_dataset_loader.ipynb"), "w") as f:
    json.dump(make_nb(nb1), f, indent=2)

# 02 SRM Metrics Notebook
nb2 = [
    create_markdown_cell("# GeoSR-AI — SRM Remote Sensing Metrics Notebook (PSNR, SSIM, RMSE, SAM)"),
    create_code_cell([
        "import numpy as np\n",
        "from metrics import calculate_psnr, calculate_ssim, calculate_rmse, calculate_sam\n\n",
        "img1 = np.random.uniform(0, 1, (128, 128, 3))\n",
        "img2 = img1 + np.random.normal(0, 0.05, (128, 128, 3))\n",
        "img2 = np.clip(img2, 0, 1)\n\n",
        "print('PSNR (dB):', calculate_psnr(img1, img2))\n",
        "print('SSIM:     ', calculate_ssim(img1, img2))\n",
        "print('RMSE:     ', calculate_rmse(img1, img2))\n",
        "print('SAM (deg):', calculate_sam(img1, img2))"
    ])
]
with open(os.path.join(target_dir, "02_srm_metrics.ipynb"), "w") as f:
    json.dump(make_nb(nb2), f, indent=2)

# 03 Models Architecture Notebook
nb3 = [
    create_markdown_cell("# GeoSR-AI — SRM Model Architectures (SRCNN, EDSR, SwinIR)"),
    create_code_cell([
        "import torch\n",
        "from base_model import BaseSRMModel\n",
        "import sys, os\n",
        "sys.path.insert(0, '../GeoSR-AI')\n",
        "from models.srcnn import SRCNN\n",
        "from models.edsr import EDSR\n",
        "from models.swinir import SwinIR\n\n",
        "device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')\n",
        "model_srcnn = SRCNN(in_channels=3, out_channels=3, scale_factor=4).to(device)\n",
        "model_edsr = EDSR(in_channels=3, out_channels=3, scale_factor=4).to(device)\n",
        "model_swinir = SwinIR(in_channels=3, out_channels=3, scale_factor=4).to(device)\n\n",
        "print('SRCNN Parameters:', sum(p.numel() for p in model_srcnn.parameters()))\n",
        "print('EDSR Parameters: ', sum(p.numel() for p in model_edsr.parameters()))\n",
        "print('SwinIR Parameters:', sum(p.numel() for p in model_swinir.parameters()))"
    ])
]
with open(os.path.join(target_dir, "03_srm_models_architecture.ipynb"), "w") as f:
    json.dump(make_nb(nb3), f, indent=2)

# 04 Training Pipeline Notebook
nb4 = [
    create_markdown_cell("# GeoSR-AI — SRM Training & Validation Pipeline Notebook"),
    create_code_cell([
        "import torch\n",
        "import sys, os\n",
        "sys.path.insert(0, '../GeoSR-AI')\n",
        "from training.trainer import GeoSRTrainer\n",
        "from datasets.paired_dataset import create_dataloaders\n",
        "from models.srcnn import SRCNN\n\n",
        "device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')\n",
        "train_loader, val_loader, _ = create_dataloaders('../GeoSR-AI/data/processed', batch_size=4, patch_size=128, scale_factor=4)\n",
        "model = SRCNN(in_channels=3, out_channels=3, scale_factor=4).to(device)\n\n",
        "config = {\n",
        "  'training': {'batch_size': 4, 'epochs': 2, 'learning_rate': 0.0001, 'weight_decay': 0.0},\n",
        "  'loss': {'reconstruction_weight': 1.0, 'spectral_weight': 0.1, 'structural_weight': 0.05},\n",
        "  'checkpoint': {'checkpoint_dir': '../GeoSR-AI/checkpoints/srcnn'}\n",
        "}\n",
        "trainer = GeoSRTrainer(model=model, config=config, device=device)\n",
        "history = trainer.fit(train_loader, val_loader, epochs=2)\n",
        "print('Done training demonstration!')"
    ])
]
with open(os.path.join(target_dir, "04_training_pipeline.ipynb"), "w") as f:
    json.dump(make_nb(nb4), f, indent=2)

# 05 Tiled Inference Notebook
nb5 = [
    create_markdown_cell("# GeoSR-AI — SRM Tiled Inference & GeoTIFF Export Notebook"),
    create_code_cell([
        "import torch\n",
        "import sys, os\n",
        "sys.path.insert(0, '../GeoSR-AI')\n",
        "from inference.predictor import GeoSRPredictor\n",
        "from geospatial.metadata import GeoMetadata\n\n",
        "predictor = GeoSRPredictor(model_name='srcnn', scale_factor=4)\n",
        "dummy_input = np.random.uniform(0, 1, (3, 64, 64)).astype(np.float32)\n",
        "res = predictor.predict(dummy_input)\n",
        "print('Super-resolved spatial representation shape:', res['sr_image'].shape)\n",
        "print('Uncertainty map shape:', res['uncertainty'].shape)"
    ])
]
with open(os.path.join(target_dir, "05_tiled_inference.ipynb"), "w") as f:
    json.dump(make_nb(nb5), f, indent=2)

print("Successfully generated all notebooks in ml_dl_models directory!")
