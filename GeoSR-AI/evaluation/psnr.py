import numpy as np
import math

def calculate_psnr(img1: np.ndarray, img2: np.ndarray, max_val: float = 1.0) -> float:
    """
    Peak Signal-to-Noise Ratio (PSNR) calculation.
    Handles numeric stability for NaN/Inf values.
    """
    img1 = np.nan_to_num(img1.astype(np.float64))
    img2 = np.nan_to_num(img2.astype(np.float64))

    mse = np.mean((img1 - img2) ** 2)
    if mse < 1e-10:
        return 100.0
    return float(20 * math.log10(max_val / math.sqrt(mse)))
