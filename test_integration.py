import sys, os
sys.path.insert(0, 'GeoSR-AI')
sys.path.insert(0, 'backend')

print("=== GeoSR-AI Integration Test ===")

# 1. Config
try:
    from config import MODEL_NAME, SCALE_FACTOR, CHECKPOINT_PATH, GEOSR_ROOT
    print(f"[OK] Config loaded")
    print(f"     MODEL_NAME:      {MODEL_NAME}")
    print(f"     SCALE_FACTOR:    {SCALE_FACTOR}")
    print(f"     CHECKPOINT_PATH: {CHECKPOINT_PATH}")
    print(f"     GEOSR_ROOT:      {GEOSR_ROOT} (exists={GEOSR_ROOT.exists()})")
except Exception as e:
    print(f"[FAIL] Config: {e}")
    sys.exit(1)

# 2. PyTorch
try:
    import torch
    print(f"[OK] PyTorch {torch.__version__} | CUDA: {torch.cuda.is_available()}")
except ImportError as e:
    print(f"[FAIL] PyTorch: {e}")
    sys.exit(1)

# 3. Model factory
try:
    from models.factory import create_model
    model = create_model(MODEL_NAME, in_channels=3, out_channels=3, scale_factor=4)
    params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"[OK] Model '{MODEL_NAME}' ({model.__class__.__name__}) | Params: {params:,}")
except Exception as e:
    print(f"[FAIL] Model factory: {e}")
    sys.exit(1)

# 4. Normalizer
try:
    from preprocessing.normalization import SatelliteNormalizer
    import numpy as np
    norm = SatelliteNormalizer(method="percentile")
    arr = np.random.uniform(0, 10000, (3, 64, 64)).astype(np.float32)
    out, stats = norm.normalize(arr)
    print(f"[OK] Normalizer: percentile clip -> shape {out.shape}, range [{out.min():.3f}, {out.max():.3f}]")
except Exception as e:
    print(f"[FAIL] Normalizer: {e}")

# 5. Forward pass
try:
    import torch
    model.eval()
    dummy = torch.zeros(1, 3, 32, 32)
    with torch.no_grad():
        out = model(dummy)
    print(f"[OK] Forward pass: {tuple(dummy.shape)} -> {tuple(out.shape)}")
except Exception as e:
    print(f"[FAIL] Forward pass: {e}")
    sys.exit(1)

# 6. Uncertainty estimator
try:
    from uncertainty.estimator import UncertaintyEstimator
    ue = UncertaintyEstimator(model, num_mc_samples=3, dropout_rate=0.1, device="cpu")
    with torch.no_grad():
        mean_sr, unc = ue.estimate_uncertainty(dummy)
    print(f"[OK] Uncertainty estimator: SR {tuple(mean_sr.shape)}, Unc {tuple(unc.shape)}, range [{unc.min():.3f},{unc.max():.3f}]")
except Exception as e:
    print(f"[FAIL] Uncertainty: {e}")

# 7. GeoTIFF writer
try:
    import tempfile, numpy as np
    from geospatial.geotiff_writer import GeoTIFFWriter
    from geospatial.metadata import GeoMetadata
    arr_out = np.random.uniform(0, 1, (3, 128, 128)).astype(np.float32)
    meta = GeoMetadata(height=32, width=32, count=3, dtype='float32')
    with tempfile.NamedTemporaryFile(suffix='.png', delete=False) as f:
        path = f.name
    saved = GeoTIFFWriter.save_geotiff(arr_out, path, metadata=meta, scale_factor=4)
    print(f"[OK] GeoTIFF writer: saved to {saved}")
    os.unlink(path)
except Exception as e:
    print(f"[FAIL] GeoTIFF writer: {e}")

# 8. FastAPI imports
try:
    from fastapi import FastAPI
    from ml_service import inference_service
    print("[OK] FastAPI + ml_service imported")
except Exception as e:
    print(f"[FAIL] FastAPI/ml_service: {e}")

print("\n=== Test Complete ===")
print("To start the backend, run:")
print('  cd backend && python -m uvicorn main:app --host 0.0.0.0 --port 8000')
