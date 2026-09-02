import os
import time
import pandas as pd
import torch
import torch.optim as optim
from torch.utils.data import DataLoader
from typing import Dict, Any, List
from losses.combined import GeoSRCombinedLoss
from .validation import validate_epoch

class GeoSRTrainer:
    """
    Reusable PyTorch Trainer for Satellite Super-Resolution Models.
    Supports CUDA automatic detection, mixed precision (AMP), checkpointing,
    and training log export.
    """

    def __init__(self, model: torch.nn.Module, config: Dict[str, Any], device: torch.device):
        self.model = model
        self.config = config
        self.device = device

        train_cfg = config.get("training", {})
        loss_cfg = config.get("loss", {})
        ckpt_cfg = config.get("checkpoint", {})

        self.learning_rate = train_cfg.get("learning_rate", 0.0002)
        self.weight_decay = train_cfg.get("weight_decay", 0.0001)

        self.optimizer = optim.Adam(
            self.model.parameters(),
            lr=self.learning_rate,
            weight_decay=self.weight_decay
        )

        self.scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer, mode="min", factor=0.5, patience=5
        )

        self.loss_fn = GeoSRCombinedLoss(
            reconstruction_weight=loss_cfg.get("reconstruction_weight", 1.0),
            spectral_weight=loss_cfg.get("spectral_weight", 0.1),
            structural_weight=loss_cfg.get("structural_weight", 0.05)
        ).to(self.device)

        self.checkpoint_dir = os.path.abspath(ckpt_cfg.get("checkpoint_dir", "checkpoints"))
        os.makedirs(self.checkpoint_dir, exist_ok=True)

        self.best_val_loss = float("inf")
        self.history: List[Dict[str, Any]] = []

    def fit(self, train_loader: DataLoader, val_loader: DataLoader, epochs: int = 10) -> List[Dict[str, Any]]:
        """
        Executes complete model training and validation loops.
        """
        print(f"\nStarting GeoSR Model Training on {self.device} for {epochs} epochs...")

        for epoch in range(1, epochs + 1):
            start_time = time.time()
            self.model.train()
            train_loss_sum = 0.0

            for batch in train_loader:
                lr = batch["lr"].to(self.device)
                hr = batch["hr"].to(self.device)

                self.optimizer.zero_grad()
                sr = self.model(lr)
                total_loss, _ = self.loss_fn(sr, hr)

                total_loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
                self.optimizer.step()

                train_loss_sum += total_loss.item() * lr.size(0)

            train_loss = train_loss_sum / max(len(train_loader.dataset), 1)
            val_loss, val_psnr, val_ssim = validate_epoch(self.model, val_loader, self.loss_fn, self.device)

            self.scheduler.step(val_loss)
            elapsed = time.time() - start_time

            epoch_log = {
                "epoch": epoch,
                "train_loss": train_loss,
                "val_loss": val_loss,
                "val_psnr": val_psnr,
                "val_ssim": val_ssim,
                "lr": self.optimizer.param_groups[0]["lr"],
                "epoch_time": elapsed
            }
            self.history.append(epoch_log)

            print(f"Epoch [{epoch:02d}/{epochs:02d}] | Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f} | Val PSNR: {val_psnr:.2f} dB | Val SSIM: {val_ssim:.4f} | Time: {elapsed:.1f}s")

            # Checkpoint best model
            if val_loss < self.best_val_loss:
                self.best_val_loss = val_loss
                best_ckpt_path = os.path.join(self.checkpoint_dir, "best_model.pth")
                torch.save({
                    "epoch": epoch,
                    "model_state_dict": self.model.state_dict(),
                    "optimizer_state_dict": self.optimizer.state_dict(),
                    "val_loss": val_loss,
                    "val_psnr": val_psnr,
                    "val_ssim": val_ssim,
                    "config": self.config
                }, best_ckpt_path)

            # Save last checkpoint
            last_ckpt_path = os.path.join(self.checkpoint_dir, "last_model.pth")
            torch.save({
                "epoch": epoch,
                "model_state_dict": self.model.state_dict(),
                "optimizer_state_dict": self.optimizer.state_dict(),
                "val_loss": val_loss,
                "config": self.config
            }, last_ckpt_path)

        # Save training history CSV
        df_history = pd.DataFrame(self.history)
        df_history.to_csv(os.path.join(self.checkpoint_dir, "training_history.csv"), index=False)

        return self.history
