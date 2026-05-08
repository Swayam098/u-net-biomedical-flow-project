from dataclasses import dataclass
from time import perf_counter
from typing import Dict

import numpy as np
import torch
import torch.nn as nn

from backend.inference import run_unet_inference


@dataclass
class OptimizationConfig:
	use_torchscript: bool = True
	use_fp16: bool = True
	use_wavelet: bool = True
	wavelet_name: str = "haar"
	wavelet_threshold: float = 0.035


class OptimizedUNetInference:
	def __init__(self, model: nn.Module, device: torch.device, config: OptimizationConfig | None = None) -> None:
		self.model = model
		self.device = device
		self.config = config or OptimizationConfig()
		self.torchscript_model: nn.Module | None = None

	def compile_torchscript(self, sample_image: np.ndarray | None = None) -> bool:
		if not self.config.use_torchscript:
			return False

		if sample_image is None:
			sample_image = np.random.rand(256, 256).astype(np.float32)

		sample = torch.from_numpy(sample_image).float().unsqueeze(0).unsqueeze(0).to(self.device)
		try:
			self.model.eval()
			self.torchscript_model = torch.jit.trace(self.model, sample)
			return True
		except Exception as exc:
			print(f"[optimization] TorchScript compile failed: {exc}")
			self.torchscript_model = None
			return False

	def predict(self, image_np: np.ndarray, mc_samples: int = 1) -> Dict[str, np.ndarray | float | bool]:
		# Keep stochastic dropout path on eager model; traced eval graphs are deterministic.
		use_stochastic = mc_samples > 1
		active_model = self.model if use_stochastic else (self.torchscript_model if self.torchscript_model is not None else self.model)
		t0 = perf_counter()
		result = run_unet_inference(
			model=active_model,
			image_np=image_np,
			device=self.device,
			use_wavelet=self.config.use_wavelet,
			wavelet_name=self.config.wavelet_name,
			wavelet_threshold=self.config.wavelet_threshold,
			use_fp16=self.config.use_fp16,
			mc_samples=mc_samples,
			enable_dropout=use_stochastic,
		)
		runtime = perf_counter() - t0
		return {
			**result,
			"runtime": runtime,
			"optimized": self.torchscript_model is not None,
			"fp16": bool(self.config.use_fp16 and self.device.type == "cuda"),
		}

