from typing import Dict

import torch
import torch.nn as nn
import torch.nn.functional as F


class FrequencyWeightedLoss(nn.Module):
	"""Blend pixel reconstruction with high-frequency consistency."""

	def __init__(self, alpha: float = 0.7, beta: float = 0.3) -> None:
		super().__init__()
		self.alpha = alpha
		self.beta = beta

	@staticmethod
	def _sobel(x: torch.Tensor) -> torch.Tensor:
		kx = torch.tensor([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], dtype=x.dtype, device=x.device).view(1, 1, 3, 3)
		ky = torch.tensor([[-1, -2, -1], [0, 0, 0], [1, 2, 1]], dtype=x.dtype, device=x.device).view(1, 1, 3, 3)
		gx = F.conv2d(x, kx, padding=1)
		gy = F.conv2d(x, ky, padding=1)
		return torch.sqrt(gx * gx + gy * gy + 1e-8)

	def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
		base = F.mse_loss(pred, target)
		edge_pred = self._sobel(pred)
		edge_target = self._sobel(target)
		freq = F.l1_loss(edge_pred, edge_target)
		return self.alpha * base + self.beta * freq


def build_loss(mode: str = "mse") -> nn.Module:
	mode = mode.lower().strip()
	if mode == "frequency":
		return FrequencyWeightedLoss()
	if mode == "l1":
		return nn.L1Loss()
	return nn.MSELoss()


def summarize_losses(pred: torch.Tensor, target: torch.Tensor) -> Dict[str, float]:
	mse = F.mse_loss(pred, target).item()
	mae = F.l1_loss(pred, target).item()
	freq = FrequencyWeightedLoss()(pred, target).item()
	return {"mse": mse, "mae": mae, "frequency": freq}

