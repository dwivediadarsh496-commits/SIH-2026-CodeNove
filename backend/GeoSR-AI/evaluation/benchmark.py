import time
import torch
import cv2
import numpy as np
from typing import Dict, Any
from torch.utils.data import DataLoader
from .psnr import calculate_psnr
from .ssim import calculate_ssim
from .rmse import calculate_rmse
from .sam import calculate_sam
from .ergas import calculate_ergas

def evaluate_bicubic_baseline(test_loader: DataLoader, scale_factor: int = 4) -> Dict[str, Any]:
    """
    Evaluates Bicubic Interpolation baseline on test set.
    """
    psnr_list, ssim_list, rmse_list, sam_list, ergas_list = [], [], [], [], []
    start_time = time.time()

    for batch in test_loader:
        lr = batch["lr"].numpy() # (B, C, H_lr, W_lr)
        hr = batch["hr"].numpy() # (B, C, H_hr, W_hr)

        batch_size, c, h_lr, w_lr = lr.shape
        h_hr, w_hr = hr.shape[2], hr.shape[3]

        for i in range(batch_size):
            lr_img = np.transpose(lr[i], (1, 2, 0))
            hr_img = np.transpose(hr[i], (1, 2, 0))

            # Bicubic upsampling
            sr_bicubic = cv2.resize(lr_img, (w_hr, h_hr), interpolation=cv2.INTER_CUBIC)
            if sr_bicubic.ndim == 2:
                sr_bicubic = np.expand_dims(sr_bicubic, axis=-1)

            sr_chw = np.transpose(sr_bicubic, (2, 0, 1))
            hr_chw = np.transpose(hr_img, (2, 0, 1))

            psnr_list.append(calculate_psnr(sr_chw, hr_chw))
            ssim_list.append(calculate_ssim(sr_chw, hr_chw))
            rmse_list.append(calculate_rmse(sr_chw, hr_chw))
            sam_list.append(calculate_sam(sr_chw, hr_chw))
            ergas_list.append(calculate_ergas(sr_chw, hr_chw, scale_factor=scale_factor))

    elapsed = time.time() - start_time

    return {
        "Model": "Bicubic Interpolation",
        "PSNR": float(np.mean(psnr_list)),
        "SSIM": float(np.mean(ssim_list)),
        "RMSE": float(np.mean(rmse_list)),
        "SAM (deg)": float(np.mean(sam_list)),
        "ERGAS": float(np.mean(ergas_list)),
        "Inference Time (s)": float(elapsed)
    }

def evaluate_model(model: torch.nn.Module, test_loader: DataLoader, model_name: str, device: str = "cpu", scale_factor: int = 4) -> Dict[str, Any]:
    """
    Evaluates trained PyTorch Super-Resolution model on test set.
    """
    model.eval()
    model.to(device)

    psnr_list, ssim_list, rmse_list, sam_list, ergas_list = [], [], [], [], []
    start_time = time.time()

    with torch.no_grad():
        for batch in test_loader:
            lr = batch["lr"].to(device)
            hr = batch["hr"].to(device)

            sr = model(lr)

            lr_np = sr.cpu().numpy()
            hr_np = hr.cpu().numpy()

            for i in range(lr_np.shape[0]):
                sr_chw = lr_np[i]
                hr_chw = hr_np[i]

                psnr_list.append(calculate_psnr(sr_chw, hr_chw))
                ssim_list.append(calculate_ssim(sr_chw, hr_chw))
                rmse_list.append(calculate_rmse(sr_chw, hr_chw))
                sam_list.append(calculate_sam(sr_chw, hr_chw))
                ergas_list.append(calculate_ergas(sr_chw, hr_chw, scale_factor=scale_factor))

    elapsed = time.time() - start_time

    return {
        "Model": model_name,
        "PSNR": float(np.mean(psnr_list)),
        "SSIM": float(np.mean(ssim_list)),
        "RMSE": float(np.mean(rmse_list)),
        "SAM (deg)": float(np.mean(sam_list)),
        "ERGAS": float(np.mean(ergas_list)),
        "Inference Time (s)": float(elapsed)
    }
