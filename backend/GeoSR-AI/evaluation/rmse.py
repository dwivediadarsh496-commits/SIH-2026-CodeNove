import numpy as np

def calculate_rmse(img1: np.ndarray, img2: np.ndarray) -> float:
    """Root Mean Squared Error (RMSE)."""
    a1 = np.nan_to_num(img1.astype(np.float64))
    a2 = np.nan_to_num(img2.astype(np.float64))
    mse = np.mean((a1 - a2) ** 2)
    return float(np.sqrt(mse))
