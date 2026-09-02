"""
GeoSR-AI Backend Utilities
File handling, image encoding, filename sanitization, session management.
"""
import os
import re
import uuid
import base64
import shutil
import numpy as np
from pathlib import Path
from typing import Optional
from PIL import Image
import io

from config import ALLOWED_EXTENSIONS, OUTPUT_DIR


# ── Session Management ────────────────────────────────────────────────────────

def create_session() -> str:
    """Creates a unique session ID for an inference request."""
    return str(uuid.uuid4())


def get_session_dir(session_id: str) -> Path:
    """Returns and creates the per-session output directory."""
    session_dir = OUTPUT_DIR / session_id
    session_dir.mkdir(parents=True, exist_ok=True)
    return session_dir


def cleanup_session(session_id: str) -> bool:
    """Removes all files associated with a session."""
    session_dir = OUTPUT_DIR / session_id
    if session_dir.exists():
        shutil.rmtree(session_dir, ignore_errors=True)
        return True
    return False


# ── File Validation ────────────────────────────────────────────────────────────

def sanitize_filename(filename: str) -> str:
    """
    Returns a safe filename: alphanumeric, dashes, underscores, and the
    extension only — no path components, no special characters.
    """
    name = Path(filename).name
    stem = Path(name).stem
    suffix = Path(name).suffix.lower()
    safe_stem = re.sub(r"[^\w\-]", "_", stem)[:80]
    return f"{safe_stem}{suffix}"


def validate_upload(filename: str, size_bytes: int, max_bytes: int) -> Optional[str]:
    """
    Validates uploaded file extension and size.
    Returns an error string if invalid, None if valid.
    """
    suffix = Path(filename).suffix.lower()
    if suffix not in ALLOWED_EXTENSIONS:
        return (
            f"Unsupported file type '{suffix}'. "
            f"Allowed: {', '.join(sorted(ALLOWED_EXTENSIONS))}"
        )
    if size_bytes > max_bytes:
        return (
            f"File too large ({size_bytes / 1024 / 1024:.1f} MB). "
            f"Maximum allowed: {max_bytes // 1024 // 1024} MB"
        )
    return None


# ── Image Encoding ────────────────────────────────────────────────────────────

def array_to_png_base64(arr: np.ndarray, max_dim: int = 1600) -> str:
    """
    Converts a (C, H, W) float32 array [0..1] or [0..255] to a base64-encoded PNG string.
    Downscales large previews for responsive web display.
    """
    try:
        if arr.ndim == 3 and arr.shape[0] in [1, 3, 4]:
            arr = np.transpose(arr, (1, 2, 0))  # (H, W, C)

        if arr.ndim == 3 and arr.shape[2] == 1:
            arr = arr.squeeze(-1)

        # Robust normalize to uint8
        if arr.dtype != np.uint8:
            a_min, a_max = float(arr.min()), float(arr.max())
            if a_max <= 1.05 and a_min >= -0.05:
                arr = (np.clip(arr, 0.0, 1.0) * 255.0).astype(np.uint8)
            elif a_max > 255.0:
                p2, p98 = np.percentile(arr, (2, 98))
                if p98 - p2 > 1e-6:
                    arr = (np.clip((arr - p2) / (p98 - p2), 0.0, 1.0) * 255.0).astype(np.uint8)
                else:
                    arr = np.zeros_like(arr, dtype=np.uint8)
            else:
                if a_max - a_min > 1e-6:
                    arr = (np.clip((arr - a_min) / (a_max - a_min), 0.0, 1.0) * 255.0).astype(np.uint8)
                else:
                    arr = np.zeros_like(arr, dtype=np.uint8)

        pil_img = Image.fromarray(arr)
        if max_dim > 0 and (pil_img.width > max_dim or pil_img.height > max_dim):
            pil_img.thumbnail((max_dim, max_dim), Image.Resampling.BILINEAR)

        buffer = io.BytesIO()
        pil_img.save(buffer, format="PNG")
        return base64.b64encode(buffer.getvalue()).decode("utf-8")
    except Exception:
        return ""


def array_to_colormap_base64(arr: np.ndarray, colormap: str = "magma", max_dim: int = 1600) -> str:
    """
    Renders a 2D uncertainty/single-channel array using a matplotlib colormap.
    Returns a base64-encoded PNG string.
    """
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.cm as cm

        if arr.ndim == 3:
            arr = arr.squeeze(0) if arr.shape[0] == 1 else arr[0]

        norm_arr = (arr - arr.min()) / (arr.max() - arr.min() + 1e-8)
        cmap = cm.get_cmap(colormap)
        colored = (cmap(norm_arr)[:, :, :3] * 255).astype(np.uint8)

        pil_img = Image.fromarray(colored)
        if max_dim > 0 and (pil_img.width > max_dim or pil_img.height > max_dim):
            pil_img.thumbnail((max_dim, max_dim), Image.Resampling.BILINEAR)

        buffer = io.BytesIO()
        pil_img.save(buffer, format="PNG")
        return base64.b64encode(buffer.getvalue()).decode("utf-8")
    except Exception:
        return array_to_png_base64(arr, max_dim=max_dim)


def save_preview_png(arr: np.ndarray, path: Path) -> bool:
    """Saves a (C, H, W) array as a preview PNG file."""
    try:
        if arr.ndim == 3 and arr.shape[0] in [1, 3, 4]:
            arr = np.transpose(arr, (1, 2, 0))
        if arr.ndim == 3 and arr.shape[2] == 1:
            arr = arr.squeeze(-1)
        if arr.dtype != np.uint8:
            a_min, a_max = float(arr.min()), float(arr.max())
            if a_max <= 1.05 and a_min >= -0.05:
                arr = (np.clip(arr, 0.0, 1.0) * 255.0).astype(np.uint8)
            elif a_max > 255.0:
                p2, p98 = np.percentile(arr, (2, 98))
                if p98 - p2 > 1e-6:
                    arr = (np.clip((arr - p2) / (p98 - p2), 0.0, 1.0) * 255.0).astype(np.uint8)
                else:
                    arr = np.zeros_like(arr, dtype=np.uint8)
            else:
                if a_max - a_min > 1e-6:
                    arr = (np.clip((arr - a_min) / (a_max - a_min), 0.0, 1.0) * 255.0).astype(np.uint8)
                else:
                    arr = np.zeros_like(arr, dtype=np.uint8)
        Image.fromarray(arr).save(str(path))
        return True
    except Exception:
        return False


def get_image_info(path: str) -> dict:
    """Returns basic image metadata: width, height, bands."""
    try:
        try:
            import rasterio
            with rasterio.open(path) as src:
                return {
                    "width": src.width,
                    "height": src.height,
                    "bands": src.count,
                    "crs": str(src.crs) if src.crs else None,
                    "format": "GeoTIFF",
                }
        except Exception:
            pass
        img = Image.open(path)
        w, h = img.size
        mode_bands = {"RGB": 3, "RGBA": 4, "L": 1, "P": 1}
        return {
            "width": w,
            "height": h,
            "bands": mode_bands.get(img.mode, 3),
            "crs": None,
            "format": img.format or "Image",
        }
    except Exception:
        return {"width": 0, "height": 0, "bands": 0, "crs": None, "format": "Unknown"}
