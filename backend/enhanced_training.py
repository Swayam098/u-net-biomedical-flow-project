from dataclasses import dataclass
from pathlib import Path
from typing import Dict

import torch
import torch.nn as nn
import torch.optim as optim

from backend.inference import UNet
from backend.loss_functions import build_loss


@dataclass
class TrainingConfig:
	epochs: int = 12
	lr: float = 1e-3
	device: str = "cuda" if torch.cuda.is_available() else "cpu"
	loss_mode: str = "mse"  # mse | l1 | frequency
	checkpoint_path: str = "backend/models/unet_model.pth"


class EnhancedTrainer:
	def __init__(self, config: TrainingConfig) -> None:
		self.config = config
		self.device = torch.device(config.device)
		self.model = UNet().to(self.device)
		self.criterion = build_loss(config.loss_mode)
		self.optimizer = optim.Adam(self.model.parameters(), lr=config.lr)

	def train_step(self, x: torch.Tensor, y: torch.Tensor) -> float:
		self.model.train()
		x = x.to(self.device)
		y = y.to(self.device)

		self.optimizer.zero_grad()
		pred = self.model(x)
		loss = self.criterion(pred, y)
		loss.backward()
		self.optimizer.step()
		return float(loss.item())

	def validate_step(self, x: torch.Tensor, y: torch.Tensor) -> float:
		self.model.eval()
		with torch.no_grad():
			x = x.to(self.device)
			y = y.to(self.device)
			pred = self.model(x)
			loss = self.criterion(pred, y)
		return float(loss.item())

	def save_checkpoint(self) -> str:
		path = Path(self.config.checkpoint_path)
		path.parent.mkdir(parents=True, exist_ok=True)
		torch.save(self.model.state_dict(), path)
		return str(path)

	def state(self) -> Dict[str, str | int | float]:
		return {
			"epochs": self.config.epochs,
			"lr": self.config.lr,
			"loss_mode": self.config.loss_mode,
			"device": str(self.device),
		}

