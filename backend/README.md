---
title: GeoSR Backend
emoji: 🛰️
colorFrom: blue
colorTo: purple
sdk: docker
app_port: 7860
---

# GeoSR Backend (FastAPI + PyTorch)

Backend service providing super-resolution inference and geospatial analytics endpoints for satellite imagery.

## 🚀 Endpoints
- `GET /health` : Health check & model status
- `POST /predict` : Run super-resolution inference on uploaded raster / image
- `GET /docs` : Interactive Swagger API documentation

## ⚙️ Configuration
- Runs on Docker with `uvicorn main:app --host 0.0.0.0 --port 7860`
