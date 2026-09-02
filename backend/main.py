"""
GeoSR-AI FastAPI Backend
Serves the GeoSR-AI super-resolution inference pipeline via REST API.
"""
import os
import sys
import logging
from pathlib import Path
from typing import Optional

# Ensure backend directory is in sys.path
BACKEND_DIR = Path(__file__).parent.resolve()
if str(BACKEND_DIR) not in sys.path:
    sys.path.insert(0, str(BACKEND_DIR))

from fastapi import FastAPI, File, UploadFile, Form, HTTPException, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse
import numpy as np

from fastapi.staticfiles import StaticFiles
from config import CORS_ORIGINS, MAX_UPLOAD_BYTES, OUTPUT_DIR, PROJECT_ROOT
from utils import (
    create_session, get_session_dir, cleanup_session,
    sanitize_filename, validate_upload,
    array_to_png_base64, array_to_colormap_base64,
    save_preview_png, get_image_info,
)
from ml_service import inference_service

# ── Logging ────────────────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(name)s — %(message)s",
)
logger = logging.getLogger("geosr.api")

# ── FastAPI App ────────────────────────────────────────────────────────────────
app = FastAPI(
    title="GeoSR-AI API",
    description=(
        "Deep Learning Based Super Resolution Mapping (SRM) "
        "from Medium Resolution Satellite Imageries — "
        "Smart India Hackathon"
    ),
    version="1.0.0",
    docs_url="/api/docs",
    redoc_url="/api/redoc",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
    expose_headers=["*"],
)


# ── Startup ────────────────────────────────────────────────────────────────────
@app.on_event("startup")
async def startup_event():
    """Load GeoSR-AI model once at server startup."""
    try:
        inference_service.initialize()
        logger.info("GeoSR-AI backend started successfully.")
    except Exception as e:
        logger.error(f"Failed to initialize ML service: {e}")
        # Do not crash startup — let /api/health report the error state


# ── Health Check ───────────────────────────────────────────────────────────────
@app.get("/api/health", tags=["System"])
async def health():
    """
    Returns backend health status, model readiness, and device information.
    Safe to call before model loads.
    """
    import torch
    initialized = getattr(inference_service, "_initialized", False)
    return {
        "status": "ok" if initialized else "initializing",
        "model_ready": initialized,
        "device": str(getattr(inference_service, "device", "unknown")),
        "cuda_available": torch.cuda.is_available(),
        "checkpoint_loaded": getattr(inference_service, "checkpoint_loaded", False),
        "service": "GeoSR-AI Super Resolution API",
        "version": "1.0.0",
    }


# ── Model Information ──────────────────────────────────────────────────────────
@app.get("/api/model-info", tags=["System"])
async def model_info():
    """
    Returns model architecture, scale factor, device, and checkpoint metadata.
    All values are live — derived from the actual loaded model.
    """
    if not getattr(inference_service, "_initialized", False):
        raise HTTPException(status_code=503, detail="Model not yet initialized.")
    return inference_service.get_model_info()


# ── Super-Resolution Inference ─────────────────────────────────────────────────
@app.post("/api/super-resolution", tags=["Inference"])
async def super_resolution(
    background_tasks: BackgroundTasks,
    file: UploadFile = File(..., description="LR satellite image (GeoTIFF / PNG / JPG)"),
    reference: Optional[UploadFile] = File(None, description="Optional HR reference for metrics"),
):
    """
    Main inference endpoint.

    - Accepts a low-resolution satellite image (preferably GeoTIFF)
    - Optionally accepts an HR reference for metric computation
    - Returns super-resolved image, uncertainty map, and metrics as base64 PNGs + download URLs
    """
    if not getattr(inference_service, "_initialized", False):
        raise HTTPException(status_code=503, detail="Model not yet initialized. Try again shortly.")

    # ── Validate upload ──────────────────────────────────────────────────────
    raw_bytes = await file.read()
    error = validate_upload(file.filename or "upload", len(raw_bytes), MAX_UPLOAD_BYTES)
    if error:
        raise HTTPException(status_code=400, detail=error)

    # ── Create session ───────────────────────────────────────────────────────
    session_id = create_session()
    session_dir = get_session_dir(session_id)

    safe_name = sanitize_filename(file.filename or "input.tif")
    input_path = session_dir / f"input_{safe_name}"

    with open(str(input_path), "wb") as f:
        f.write(raw_bytes)

    # ── Save optional reference ──────────────────────────────────────────────
    reference_path = None
    if reference and reference.filename:
        ref_bytes = await reference.read()
        ref_error = validate_upload(reference.filename, len(ref_bytes), MAX_UPLOAD_BYTES)
        if not ref_error:
            ref_safe = sanitize_filename(reference.filename)
            ref_p = session_dir / f"reference_{ref_safe}"
            with open(str(ref_p), "wb") as f:
                f.write(ref_bytes)
            reference_path = str(ref_p)

    # ── Get input metadata ───────────────────────────────────────────────────
    input_info = get_image_info(str(input_path))

    # ── Run inference ────────────────────────────────────────────────────────
    try:
        result = inference_service.run_inference(
            input_path=str(input_path),
            reference_path=reference_path,
            use_tiled=True,
        )
    except Exception as e:
        logger.exception("Inference failed")
        background_tasks.add_task(cleanup_session, session_id)
        raise HTTPException(status_code=500, detail=f"Inference error: {str(e)}")

    sr_image      = result["sr_image"]       # np.ndarray (C, H_sr, W_sr)
    uncertainty   = result["uncertainty"]    # np.ndarray (1, H_sr, W_sr)
    metrics       = result["metrics"]

    # ── Save output files ────────────────────────────────────────────────────
    sr_preview_path  = session_dir / "sr_preview.png"
    unc_preview_path = session_dir / "uncertainty_preview.png"
    sr_tif_path      = session_dir / "sr_output.tif"

    save_preview_png(sr_image, sr_preview_path)
    # Uncertainty colormap preview
    try:
        import matplotlib.cm as cm
        from PIL import Image
        unc_2d = uncertainty.squeeze(0) if uncertainty.ndim == 3 else uncertainty
        norm_unc = (unc_2d - unc_2d.min()) / (unc_2d.max() - unc_2d.min() + 1e-8)
        colored = (cm.get_cmap("magma")(norm_unc)[:, :, :3] * 255).astype(np.uint8)
        Image.fromarray(colored).save(str(unc_preview_path))
    except Exception:
        save_preview_png(uncertainty, unc_preview_path)

    # Save SR GeoTIFF using existing GeoTIFFWriter
    try:
        sys.path.insert(0, str(Path(__file__).parent.parent / "GeoSR-AI"))
        from geospatial.geotiff_writer import GeoTIFFWriter
        GeoTIFFWriter.save_geotiff(
            data=sr_image,
            output_path=str(sr_tif_path),
            metadata=result["metadata"],
            scale_factor=inference_service.get_model_info()["scale_factor"],
        )
    except Exception as e:
        logger.warning(f"Could not write SR GeoTIFF: {e}")

    # ── Encode previews as base64 ────────────────────────────────────────────
    sr_b64  = array_to_png_base64(sr_image)
    unc_b64 = array_to_colormap_base64(uncertainty, colormap="magma")

    model_info = inference_service.get_model_info()

    return {
        "success": True,
        "session_id": session_id,

        "input": {
            "filename": safe_name,
            "width":    input_info["width"],
            "height":   input_info["height"],
            "bands":    input_info["bands"],
            "format":   input_info["format"],
            "crs":      input_info["crs"],
            "resolution": "10m (Sentinel-2)",
        },

        "model": {
            "name":             model_info["model_name"],
            "architecture":     model_info["architecture"],
            "scale_factor":     model_info["scale_factor"],
            "device":           model_info["device"],
            "checkpoint_loaded": model_info["checkpoint_loaded"],
            "parameters":       model_info["trainable_parameters"],
        },

        "output": {
            "width":  result["output_shape"][2] if len(result["output_shape"]) > 2 else 0,
            "height": result["output_shape"][1] if len(result["output_shape"]) > 1 else 0,
            "bands":  result["output_shape"][0] if len(result["output_shape"]) > 0 else 0,
            "sr_image_b64":      sr_b64,
            "uncertainty_map_b64": unc_b64,
            "tiled_inference":   result["tiled"],
            "inference_time_s":  result["inference_time_s"],
        },

        "metrics": metrics,

        "downloads": {
            "sr_geotiff":    f"/api/download/{session_id}/sr_geotiff",
            "sr_preview":    f"/api/download/{session_id}/sr_preview",
            "uncertainty":   f"/api/download/{session_id}/uncertainty",
        },
    }


# ── File Download Endpoints ────────────────────────────────────────────────────
@app.get("/api/download/{session_id}/sr_geotiff", tags=["Downloads"])
async def download_sr_geotiff(session_id: str):
    """Download the super-resolved GeoTIFF output."""
    _validate_session_id(session_id)
    path = OUTPUT_DIR / session_id / "sr_output.tif"
    if not path.exists():
        raise HTTPException(status_code=404, detail="SR GeoTIFF not found for this session.")
    return FileResponse(
        str(path),
        media_type="image/tiff",
        filename="geosr_ai_super_resolved.tif",
    )


@app.get("/api/download/{session_id}/sr_preview", tags=["Downloads"])
async def download_sr_preview(session_id: str):
    """Download the SR preview PNG."""
    _validate_session_id(session_id)
    path = OUTPUT_DIR / session_id / "sr_preview.png"
    if not path.exists():
        raise HTTPException(status_code=404, detail="SR preview not found.")
    return FileResponse(str(path), media_type="image/png", filename="geosr_ai_preview.png")


@app.get("/api/download/{session_id}/uncertainty", tags=["Downloads"])
async def download_uncertainty(session_id: str):
    """Download the uncertainty map PNG."""
    _validate_session_id(session_id)
    path = OUTPUT_DIR / session_id / "uncertainty_preview.png"
    if not path.exists():
        raise HTTPException(status_code=404, detail="Uncertainty map not found.")
    return FileResponse(str(path), media_type="image/png", filename="geosr_ai_uncertainty.png")


# ── Session Cleanup ────────────────────────────────────────────────────────────
@app.delete("/api/session/{session_id}", tags=["System"])
async def delete_session(session_id: str):
    """Cleans up temporary inference files for a session."""
    _validate_session_id(session_id)
    deleted = cleanup_session(session_id)
    return {"deleted": deleted, "session_id": session_id}


# ── Helper ─────────────────────────────────────────────────────────────────────
def _validate_session_id(session_id: str) -> None:
    """Prevents directory traversal by validating session ID format."""
    import re
    if not re.fullmatch(r"[0-9a-f\-]{36}", session_id):
        raise HTTPException(status_code=400, detail="Invalid session ID.")


# ── Root / Health Route ────────────────────────────────────────────────────────
@app.get("/", tags=["System"])
async def root():
    return {
        "service": "GeoSR-AI API Backend",
        "status": "online",
        "docs": "/api/docs",
        "health": "/api/health",
        "model_info": "/api/model-info",
    }


frontend_out_dir = PROJECT_ROOT / "frontend" / "out"
if frontend_out_dir.exists():
    app.mount("/static-app", StaticFiles(directory=str(frontend_out_dir), html=True), name="frontend")


# ── Dev entrypoint ─────────────────────────────────────────────────────────────
if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=False, log_level="info")
