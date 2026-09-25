# CodeNove — GeoSR-AI

> AI-powered satellite image super-resolution and geospatial processing platform, built for Smart India Hackathon 2026.

[![Live Demo](https://img.shields.io/badge/Live%20Demo-Vercel-black?logo=vercel)](https://sih-2026-code-nove.vercel.app/)
[![Python](https://img.shields.io/badge/Python-3.10%2B-3776AB?logo=python&logoColor=white)](https://www.python.org/)
[![FastAPI](https://img.shields.io/badge/API-FastAPI-009688?logo=fastapi&logoColor=white)](https://fastapi.tiangolo.com/)

## Overview

CodeNove helps transform low-resolution multi-spectral satellite imagery—such as Sentinel-2 and Landsat data—into higher-resolution imagery using deep learning. It combines a web interface, a FastAPI inference service, and geospatial tooling designed to preserve useful raster metadata during processing.

**Live application:** [sih-2026-code-nove.vercel.app](https://sih-2026-code-nove.vercel.app/)

## Features

- Super-resolution models: **SRCNN**, **EDSR**, and **SwinIR**
- Multi-band satellite raster support (RGB, NIR, Red Edge, and more)
- GeoTIFF and CRS metadata preservation
- Memory-efficient tiled inference for large satellite scenes
- Seamless tile blending with overlapping patches
- Cloud-masking and atmospheric-correction workflows
- Quality metrics: PSNR, SSIM, RMSE, SAM, and ERGAS
- REST API for uploads, inference jobs, tracking, and downloads

## Architecture

```text
Satellite imagery (GeoTIFF / raster)
              |
              v
     Frontend web application
              |
              v
       FastAPI backend service
              |
              v
Preprocessing → AI super-resolution → Geospatial export
              |
              v
 Enhanced imagery + evaluation metrics
```

## Repository Structure

```text
.
├── frontend/             # Web application
├── backend/              # FastAPI server and inference service
│   ├── main.py           # API routes
│   ├── ml_service.py     # Model loading and inference
│   ├── config.py         # Service configuration
│   └── utils.py          # Image and geospatial utilities
├── GeoSR-AI/             # Core ML and geospatial framework
│   ├── datasets/         # Dataset loaders and transforms
│   ├── evaluation/       # PSNR, SSIM, SAM, ERGAS, and RMSE metrics
│   ├── geospatial/       # Raster and GeoTIFF tools
│   ├── inference/        # Tiled inference pipelines
│   ├── models/           # SRCNN, EDSR, and SwinIR implementations
│   ├── preprocessing/    # Tiling, masking, and normalization
│   ├── training/         # Training and validation workflows
│   └── uncertainty/      # Uncertainty estimation modules
├── ml_dl_models/         # ML/DL research notebooks and experiments
├── requirements.txt      # Project dependencies
└── test_integration.py   # End-to-end integration test
```

## Getting Started

### Prerequisites

- Python 3.10 or later
- `pip`
- Git

### Installation

```bash
git clone https://github.com/dwivediadarsh496-commits/SIH-2026-CodeNove.git
cd SIH-2026-CodeNove

python -m venv .venv
```

Activate the environment:

```bash
# Windows (PowerShell)
.\.venv\Scripts\Activate.ps1

# Linux / macOS
source .venv/bin/activate
```

Install the project dependencies:

```bash
pip install -r requirements.txt
```

If the backend maintains its own dependency list, install it as well:

```bash
pip install -r backend/requirements.txt
```

### Run the API

```bash
uvicorn backend.main:app --host 0.0.0.0 --port 8000 --reload
```

After it starts, open the interactive API documentation at [http://localhost:8000/docs](http://localhost:8000/docs).

### Run the Integration Test

```bash
python test_integration.py
```

## Evaluation Metrics

| Metric | What it measures |
| --- | --- |
| PSNR | Pixel-level reconstruction quality |
| SSIM | Structural similarity between images |
| RMSE | Average prediction error magnitude |
| SAM | Spectral distortion across image bands |
| ERGAS | Relative global error in synthesized imagery |

## Technology Stack

- **Frontend:** Web application deployed on Vercel
- **Backend:** FastAPI
- **Machine learning:** Python deep-learning models (SRCNN, EDSR, SwinIR)
- **Geospatial processing:** GeoTIFF and multi-band raster workflows

## Contributing

1. Fork the repository.
2. Create a feature branch: `git checkout -b feature/your-feature`.
3. Commit your changes with a clear message.
4. Push the branch and open a pull request.

## License

This project was developed for **Smart India Hackathon 2026**. Add a license file to the repository before reusing or distributing the code outside the project terms.

## Team

Built by **CodeNove** for Smart India Hackathon 2026.
