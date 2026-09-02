import numpy as np

def calculate_ssim(img1: np.ndarray, img2: np.ndarray, max_val: float = 1.0) -> float:
    """
    Structural Similarity Index (SSIM) for satellite imagery arrays.
    """
    C1 = (0.01 * max_val) ** 2
    C2 = (0.03 * max_val) ** 2

    a1 = np.nan_to_num(img1.astype(np.float64))
    a2 = np.nan_to_num(img2.astype(np.float64))

    mu1 = np.mean(a1)
    mu2 = np.mean(a2)

    sigma1_sq = np.var(a1)
    sigma2_sq = np.var(a2)

    flat1 = a1.flatten()
    flat2 = a2.flatten()
    if len(flat1) > 1 and np.std(flat1) > 0 and np.std(flat2) > 0:
        sigma12 = np.cov(flat1, flat2)[0, 1]
    else:
        sigma12 = 0.0

    ssim_num = (2 * mu1 * mu2 + C1) * (2 * sigma12 + C2)
    ssim_den = (mu1**2 + mu2**2 + C1) * (sigma1_sq + sigma2_sq + C2)

    return float(np.clip(ssim_num / (ssim_den + 1e-8), -1.0, 1.0))
