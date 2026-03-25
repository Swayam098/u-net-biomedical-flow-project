"""
Enhanced training module with advanced loss functions
- Support multiple loss functions
- Learning rate scheduling (cosine annealing, step decay)
- Validation and checkpointing
- Performance tracking
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import CosineAnnealingLR, StepLR, ReduceLROnPlateau
import numpy as np
from typing import Optional, Tuple, List, Dict
import time
from pathlib import Path

from loss_functions import LossFactory


class TrainingConfig:
    """Configuration for training"""
    
    def __init__(
        self,
        epochs: int = 12,
        batch_size: int = 8,
        learning_rate: float = 1e-3,
        loss_type: str = "mse",  # "mse", "combined", "ssim"
        scheduler_type: str = "cosine",  # "cosine", "step", "plateau", "none"
        device: str = "cuda",
        mixed_precision: bool = True,
        **kwargs
    ):
        self.epochs = epochs
        self.batch_size = batch_size
        self.learning_rate = learning_rate
        self.loss_type = loss_type
        self.scheduler_type = scheduler_type
        self.device = device
        self.mixed_precision = mixed_precision
        self.kwargs = kwargs
    
    def __repr__(self):
        return f"""
TrainingConfig:
  • Epochs: {self.epochs}
  • Batch Size: {self.batch_size}
  • Learning Rate: {self.learning_rate}
  • Loss Function: {self.loss_type}
  • LR Scheduler: {self.scheduler_type}
  • Device: {self.device}
  • Mixed Precision: {self.mixed_precision}
"""


class EnhancedTrainer:
    """Enhanced trainer with advanced features"""
    
    def __init__(self, model: nn.Module, config: TrainingConfig):
        self.model = model.to(config.device)
        self.config = config
        self.device = config.device
        
        # Loss function
        self.criterion = LossFactory.create(
            config.loss_type,
            device=config.device
        )
        
        # Optimizer
        self.optimizer = optim.Adam(
            self.model.parameters(),
            lr=config.learning_rate
        )
        
        # Learning rate scheduler
        self.scheduler = self._create_scheduler()
        
        # Mixed precision
        self.use_amp = config.mixed_precision and torch.cuda.is_available()
        if self.use_amp:
            self.scaler = torch.cuda.amp.GradScaler()
        
        # Tracking
        self.history = {
            "train_loss": [],
            "val_loss": [],
            "learning_rates": [],
            "best_val_loss": float('inf'),
            "best_epoch": 0
        }
    
    def _create_scheduler(self):
        """Create learning rate scheduler"""
        if self.config.scheduler_type == "cosine":
            return CosineAnnealingLR(
                self.optimizer,
                T_max=self.config.epochs,
                eta_min=1e-6
            )
        elif self.config.scheduler_type == "step":
            return StepLR(self.optimizer, step_size=4, gamma=0.5)
        elif self.config.scheduler_type == "plateau":
            return ReduceLROnPlateau(
                self.optimizer,
                mode='min',
                factor=0.5,
                patience=2,
                verbose=True
            )
        else:
            return None
    
    def train_epoch(self, train_loader: DataLoader) -> float:
        """Train for one epoch"""
        self.model.train()
        total_loss = 0.0
        
        for batch_idx, (images, targets) in enumerate(train_loader):
            images = images.to(self.device)
            targets = targets.to(self.device)
            
            self.optimizer.zero_grad()
            
            if self.use_amp:
                with torch.cuda.amp.autocast():
                    outputs = self.model(images)
                    loss = self.criterion(outputs, targets)
                
                self.scaler.scale(loss).backward()
                self.scaler.step(self.optimizer)
                self.scaler.update()
            else:
                outputs = self.model(images)
                loss = self.criterion(outputs, targets)
                loss.backward()
                self.optimizer.step()
            
            total_loss += loss.item()
            
            if (batch_idx + 1) % 10 == 0:
                print(f"  Batch {batch_idx+1}/{len(train_loader)}, Loss: {loss.item():.6f}")
        
        avg_loss = total_loss / len(train_loader)
        return avg_loss
    
    def validate(self, val_loader: DataLoader) -> float:
        """Validate on validation set"""
        self.model.eval()
        total_loss = 0.0
        
        with torch.no_grad():
            for images, targets in val_loader:
                images = images.to(self.device)
                targets = targets.to(self.device)
                
                outputs = self.model(images)
                loss = self.criterion(outputs, targets)
                total_loss += loss.item()
        
        avg_loss = total_loss / len(val_loader)
        return avg_loss
    
    def train(
        self,
        train_loader: DataLoader,
        val_loader: Optional[DataLoader] = None,
        checkpoint_dir: str = "checkpoints"
    ) -> Dict:
        """Train model for multiple epochs"""
        print(f"\n{'='*70}")
        print(f"🚀 STARTING TRAINING")
        print(f"{'='*70}")
        print(self.config)
        
        checkpoint_path = Path(checkpoint_dir)
        checkpoint_path.mkdir(exist_ok=True)
        
        best_model_path = checkpoint_path / f"best_unet_{self.config.loss_type}.pth"
        
        for epoch in range(self.config.epochs):
            epoch_start = time.time()
            
            # Train
            train_loss = self.train_epoch(train_loader)
            self.history["train_loss"].append(train_loss)
            
            # Validate
            val_loss = None
            if val_loader:
                val_loss = self.validate(val_loader)
                self.history["val_loss"].append(val_loss)
            
            # Update scheduler
            if self.config.scheduler_type == "plateau" and val_loss is not None:
                self.scheduler.step(val_loss)
            elif self.scheduler is not None:
                self.scheduler.step()
            
            current_lr = self.optimizer.param_groups[0]['lr']
            self.history["learning_rates"].append(current_lr)
            
            # Logging
            epoch_time = time.time() - epoch_start
            print(f"\nEpoch [{epoch+1}/{self.config.epochs}] ({epoch_time:.2f}s)")
            print(f"  Train Loss: {train_loss:.6f}")
            if val_loss is not None:
                print(f"  Val Loss: {val_loss:.6f}")
            print(f"  LR: {current_lr:.2e}")
            
            # Checkpointing
            if val_loss is not None and val_loss < self.history["best_val_loss"]:
                self.history["best_val_loss"] = val_loss
                self.history["best_epoch"] = epoch
                torch.save(self.model.state_dict(), best_model_path)
                print(f"  ✓ Best model saved: {best_model_path}")
        
        print(f"\n{'='*70}")
        print(f"✅ TRAINING COMPLETED")
        print(f"{'='*70}")
        print(f"Best Val Loss: {self.history['best_val_loss']:.6f} (Epoch {self.history['best_epoch']+1})")
        print(f"Best Model: {best_model_path}\n")
        
        return self.history
    
    def get_metrics(self) -> Dict:
        """Get training metrics"""
        return {
            "final_train_loss": self.history["train_loss"][-1] if self.history["train_loss"] else None,
            "best_val_loss": self.history["best_val_loss"],
            "best_epoch": self.history["best_epoch"] + 1,
            "final_lr": self.history["learning_rates"][-1] if self.history["learning_rates"] else None
        }


class LossComparator:
    """Compare different loss functions"""
    
    @staticmethod
    def compare_losses(
        model: nn.Module,
        train_loader: DataLoader,
        val_loader: DataLoader,
        epochs: int = 3,
        device: str = "cuda"
    ):
        """Compare performance of different loss functions"""
        
        losses_to_test = ["mse", "l1", "combined", "ssim"]
        results = {}
        
        print(f"\n{'='*70}")
        print(f"🔬 COMPARING LOSS FUNCTIONS")
        print(f"{'='*70}\n")
        
        for loss_name in losses_to_test:
            print(f"\n{'─'*70}")
            print(f"Testing: {loss_name.upper()}")
            print(f"{'─'*70}")
            
            # Reset model
            model_copy = model.__class__()
            model_copy.load_state_dict(model.state_dict())
            
            config = TrainingConfig(
                epochs=epochs,
                loss_type=loss_name,
                device=device
            )
            
            trainer = EnhancedTrainer(model_copy, config)
            history = trainer.train(train_loader, val_loader)
            
            results[loss_name] = {
                "history": history,
                "metrics": trainer.get_metrics()
            }
        
        # Summary
        print(f"\n{'='*70}")
        print(f"📊 LOSS COMPARISON SUMMARY")
        print(f"{'='*70}\n")
        
        for loss_name, data in results.items():
            metrics = data["metrics"]
            print(f"{loss_name.upper():12} | Best Val Loss: {metrics['best_val_loss']:.6f} (Epoch {metrics['best_epoch']})")
        
        return results
