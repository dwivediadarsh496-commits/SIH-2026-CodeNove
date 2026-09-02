import torch
from typing import Tuple, Dict
from torch.utils.data import DataLoader
from evaluation.psnr import calculate_psnr
from evaluation.ssim import calculate_ssim

def validate_epoch(
    model: torch.nn.Module,
    val_loader: DataLoader,
    loss_fn: torch.nn.Module,
    device: torch.device
) -> Tuple[float, float, float]:
    """
    Validation loop returning (val_loss, val_psnr, val_ssim).
    """
    model.eval()
    val_loss_sum = 0.0
    psnr_list, ssim_list = [], []

    with torch.no_grad():
        for batch in val_loader:
            lr = batch["lr"].to(device)
            hr = batch["hr"].to(device)

            sr = model(lr)
            total_loss, _ = loss_fn(sr, hr)
            val_loss_sum += total_loss.item() * lr.size(0)

            sr_np = sr.cpu().numpy()
            hr_np = hr.cpu().numpy()

            for i in range(sr_np.shape[0]):
                psnr_list.append(calculate_psnr(sr_np[i], hr_np[i]))
                ssim_list.append(calculate_ssim(sr_np[i], hr_np[i]))

    mean_loss = val_loss_sum / max(len(val_loader.dataset), 1)
    mean_psnr = float(torch.tensor(psnr_list).mean().item()) if psnr_list else 0.0
    mean_ssim = float(torch.tensor(ssim_list).mean().item()) if ssim_list else 0.0

    return mean_loss, mean_psnr, mean_ssim
