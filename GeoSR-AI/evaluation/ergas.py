import numpy as np

def calculate_ergas(img_sr: np.ndarray, img_hr: np.ndarray, scale_factor: int = 4) -> float:
    """
    Relative Global Dimensional Error (ERGAS) metric.
    Measures global dimensional error across multi-spectral bands.
    
    Formula:
      ERGAS = 100 * (h / l) * sqrt( (1 / N) * sum( (RMSE_k / Mean_k)^2 ) )
    """
    if img_sr.ndim == 2:
        img_sr = np.expand_dims(img_sr, axis=0)
        img_hr = np.expand_dims(img_hr, axis=0)

    if img_sr.ndim == 3 and img_sr.shape[2] in [1, 3, 4] and img_sr.shape[0] not in [1, 3, 4]:
        img_sr = np.transpose(img_sr, (2, 0, 1))
        img_hr = np.transpose(img_hr, (2, 0, 1))

    num_bands = img_sr.shape[0]
    sum_ratio_sq = 0.0

    for k in range(num_bands):
        band_sr = np.nan_to_num(img_sr[k].astype(np.float64))
        band_hr = np.nan_to_num(img_hr[k].astype(np.float64))

        rmse_k = np.sqrt(np.mean((band_sr - band_hr) ** 2))
        mean_k = np.mean(band_hr)

        if abs(mean_k) < 1e-6:
            mean_k = 1e-6

        sum_ratio_sq += (rmse_k / mean_k) ** 2

    ergas = 100.0 * (1.0 / float(scale_factor)) * np.sqrt(sum_ratio_sq / float(num_bands))
    return float(ergas)
