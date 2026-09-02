"""
GeoSR-AI Backend Configuration
All settings are environment-variable driven. Copy .env.example to .env and edit.
"""
import os
from pathlib import Path

# ── Root paths ──────────────────────────────────────────────────────────────
BACKEND_DIR   = Path(__file__).parent.resolve()
GEOSR_ROOT    = BACKEND_DIR / "GeoSR-AI"
if not GEOSR_ROOT.exists():
    GEOSR_ROOT = BACKEND_DIR.parent / "GeoSR-AI"
PROJECT_ROOT  = BACKEND_DIR.parent.resolve()

# ── ML Model ─────────────────────────────────────────────────────────────────
# Model architecture: srcnn | edsr | swinir
MODEL_NAME      = os.getenv("GEOSR_MODEL_NAME", "srcnn")
SCALE_FACTOR    = int(os.getenv("GEOSR_SCALE_FACTOR", "4"))
IN_CHANNELS     = int(os.getenv("GEOSR_IN_CHANNELS", "3"))
OUT_CHANNELS    = int(os.getenv("GEOSR_OUT_CHANNELS", "3"))
NUM_FEATURES    = int(os.getenv("GEOSR_NUM_FEATURES", "64"))
NUM_BLOCKS      = int(os.getenv("GEOSR_NUM_BLOCKS", "16"))

# Checkpoint path — leave empty to run with random (untrained) weights
# Example: GEOSR_CHECKPOINT=/path/to/checkpoints/best_model.pth
CHECKPOINT_PATH = os.getenv(
    "GEOSR_CHECKPOINT",
    str(GEOSR_ROOT / "checkpoints" / "best_model.pth")
)

# Compute device: auto | cuda | cpu
DEVICE = os.getenv("GEOSR_DEVICE", "auto")

# ── File Upload ───────────────────────────────────────────────────────────────
MAX_UPLOAD_MB   = int(os.getenv("GEOSR_MAX_UPLOAD_MB", "200"))
MAX_UPLOAD_BYTES = MAX_UPLOAD_MB * 1024 * 1024

ALLOWED_EXTENSIONS = {".tif", ".tiff", ".png", ".jpg", ".jpeg"}

# ── Output / Temp Dirs ────────────────────────────────────────────────────────
# Use /tmp on cloud (Railway/HuggingFace), local outputs/ for dev
_default_output = os.getenv("GEOSR_OUTPUT_DIR") or (
    "/tmp/geosr_outputs" if not Path(str(PROJECT_ROOT / "outputs")).exists()
    else str(PROJECT_ROOT / "outputs" / "inference")
)
OUTPUT_DIR = Path(_default_output)
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# ── Server ────────────────────────────────────────────────────────────────────
CORS_ORIGINS = os.getenv("GEOSR_CORS_ORIGINS", "*").split(",")

# ── MC-Dropout Uncertainty ────────────────────────────────────────────────────
MC_SAMPLES    = int(os.getenv("GEOSR_MC_SAMPLES", "1"))
DROPOUT_RATE  = float(os.getenv("GEOSR_DROPOUT_RATE", "0.1"))

# ── Tiled Inference ───────────────────────────────────────────────────────────
TILE_SIZE   = int(os.getenv("GEOSR_TILE_SIZE", "512"))
TILE_OVERLAP = int(os.getenv("GEOSR_TILE_OVERLAP", "32"))

