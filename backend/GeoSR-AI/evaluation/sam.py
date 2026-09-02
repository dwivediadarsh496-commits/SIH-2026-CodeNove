import numpy as np

def calculate_sam(img1: np.ndarray, img2: np.ndarray) -> float:
    """
    Spectral Angle Mapper (SAM) metric in degrees.
    Numerically stable against zero vectors, NaN, and Inf.
    """
    if img1.ndim == 2:
        img1 = np.expand_dims(img1, axis=0)
        img2 = np.expand_dims(img2, axis=0)

    # Convert (C, H, W) to (N_pixels, C)
    if img1.ndim == 3 and img1.shape[0] in [1, 3, 4]:
        c, h, w = img1.shape
        v1 = img1.reshape(c, -1).T.astype(np.float64)
        v2 = img2.reshape(c, -1).T.astype(np.float64)
    else:
        v1 = img1.reshape(-1, img1.shape[-1]).astype(np.float64)
        v2 = img2.reshape(-1, img2.shape[-1]).astype(np.float64)

    v1 = np.nan_to_num(v1)
    v2 = np.nan_to_num(v2)

    dot_product = np.sum(v1 * v2, axis=1)
    norm_v1 = np.linalg.norm(v1, axis=1)
    norm_v2 = np.linalg.norm(v2, axis=1)

    denom = norm_v1 * norm_v2
    valid_mask = denom > 1e-8

    if not np.any(valid_mask):
        return 0.0

    cos_theta = np.clip(dot_product[valid_mask] / denom[valid_mask], -1.0, 1.0)
    angles_rad = np.arccos(cos_theta)
    sam_deg = np.mean(np.degrees(angles_rad))

    return float(sam_deg)
