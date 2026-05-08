"""Standalone retraining entry point for the U-Net model.

The script expects a directory of grayscale ultrasound images. It uses the
same autoencoder-style target setup as the training notebook: the input image
is optionally noised, and the clean image is used as the target.

Default checkpoint outputs:
- backend/models/unet_model.pth
- backend/checkpoint/best_unet_model.pt
"""

from __future__ import annotations

import argparse
import random
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Tuple

import numpy as np
import torch
from PIL import Image
from torch import nn
from torch.utils.data import DataLoader, Dataset, random_split

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
	sys.path.insert(0, str(PROJECT_ROOT))

from backend.inference import UNet
from backend.loss_functions import build_loss


@dataclass(frozen=True)
class TrainArgs:
	data_dir: Path
	output_dir: Path
	epochs: int
	batch_size: int
	lr: float
	loss_mode: str
	val_split: float
	seed: int
	device: str
	noise_std: float


class UltrasoundPairDataset(Dataset):
	"""Load grayscale images and return (noisy_input, clean_target) pairs."""

	def __init__(self, image_dir: Path, noise_std: float = 0.03) -> None:
		self.image_dir = image_dir
		self.noise_std = noise_std
		self.files = [
			path
			for path in sorted(image_dir.iterdir())
			if path.suffix.lower() in {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff"}
		]

	def __len__(self) -> int:
		return len(self.files)

	@staticmethod
	def _load_image(path: Path) -> torch.Tensor:
		image = Image.open(path).convert("L").resize((256, 256))
		array = np.asarray(image, dtype=np.float32) / 255.0
		array = np.clip(array, 0.0, 1.0)
		return torch.from_numpy(array).unsqueeze(0)

	def __getitem__(self, index: int) -> Tuple[torch.Tensor, torch.Tensor]:
		clean = self._load_image(self.files[index])
		noise = torch.randn_like(clean) * self.noise_std
		noisy = torch.clamp(clean + noise, 0.0, 1.0)
		return noisy, clean


def _seed_everything(seed: int) -> None:
	random.seed(seed)
	np.random.seed(seed)
	torch.manual_seed(seed)
	if torch.cuda.is_available():
		torch.cuda.manual_seed_all(seed)


def _parse_args() -> TrainArgs:
	parser = argparse.ArgumentParser(description="Retrain the project U-Net on BUSI-style images.")
	parser.add_argument("--data-dir", type=Path, default=Path("data/images"), help="Directory containing training images")
	parser.add_argument("--output-dir", type=Path, default=Path("backend"), help="Directory for checkpoints")
	parser.add_argument("--epochs", type=int, default=12)
	parser.add_argument("--batch-size", type=int, default=8)
	parser.add_argument("--lr", type=float, default=1e-3)
	parser.add_argument("--loss-mode", type=str, default="mse", choices=["mse", "l1", "frequency"])
	parser.add_argument("--val-split", type=float, default=0.2)
	parser.add_argument("--seed", type=int, default=42)
	parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
	parser.add_argument("--noise-std", type=float, default=0.03)
	args = parser.parse_args()
	return TrainArgs(
		data_dir=args.data_dir,
		output_dir=args.output_dir,
		epochs=args.epochs,
		batch_size=args.batch_size,
		lr=args.lr,
		loss_mode=args.loss_mode,
		val_split=args.val_split,
		seed=args.seed,
		device=args.device,
		noise_std=args.noise_std,
	)


def _build_loaders(args: TrainArgs) -> Tuple[DataLoader, DataLoader]:
	dataset = UltrasoundPairDataset(args.data_dir, noise_std=args.noise_std)
	if len(dataset) < 2:
		raise RuntimeError(f"Need at least 2 images in {args.data_dir} to train.")

	val_size = max(1, int(len(dataset) * args.val_split))
	train_size = len(dataset) - val_size
	if train_size < 1:
		raise RuntimeError("Validation split leaves no training data.")

	generator = torch.Generator().manual_seed(args.seed)
	train_ds, val_ds = random_split(dataset, [train_size, val_size], generator=generator)

	train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, num_workers=0, pin_memory=torch.cuda.is_available())
	val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False, num_workers=0, pin_memory=torch.cuda.is_available())
	return train_loader, val_loader


def _save_checkpoint(model: nn.Module, output_dir: Path) -> Tuple[Path, Path]:
	models_dir = output_dir / "models"
	checkpoint_dir = output_dir / "checkpoint"
	models_dir.mkdir(parents=True, exist_ok=True)
	checkpoint_dir.mkdir(parents=True, exist_ok=True)

	state_dict = model.state_dict()
	model_path = models_dir / "unet_model.pth"
	checkpoint_path = checkpoint_dir / "best_unet_model.pt"
	torch.save(state_dict, model_path)
	torch.save(state_dict, checkpoint_path)
	return model_path, checkpoint_path


def train(args: TrainArgs) -> None:
	_seed_everything(args.seed)
	device = torch.device(args.device)
	train_loader, val_loader = _build_loaders(args)

	model = UNet().to(device)
	criterion = build_loss(args.loss_mode)
	optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
	scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=max(1, args.epochs // 2), gamma=0.5)
	use_amp = device.type == "cuda"
	scaler = torch.amp.GradScaler("cuda", enabled=use_amp)

	best_val_loss = float("inf")
	best_paths = None

	for epoch in range(args.epochs):
		model.train()
		train_loss = 0.0
		for noisy, clean in train_loader:
			noisy = noisy.to(device)
			clean = clean.to(device)
			optimizer.zero_grad(set_to_none=True)
			if use_amp:
				with torch.autocast(device_type="cuda", dtype=torch.float16):
					pred = model(noisy)
					loss = criterion(pred, clean)
				scaler.scale(loss).backward()
				scaler.step(optimizer)
				scaler.update()
			else:
				pred = model(noisy)
				loss = criterion(pred, clean)
				loss.backward()
				optimizer.step()

			train_loss += float(loss.item())

		train_loss /= max(1, len(train_loader))

		model.eval()
		val_loss = 0.0
		with torch.no_grad():
			for noisy, clean in val_loader:
				noisy = noisy.to(device)
				clean = clean.to(device)
				pred = model(noisy)
				loss = criterion(pred, clean)
				val_loss += float(loss.item())

		val_loss /= max(1, len(val_loader))
		scheduler.step()

		print(
			f"Epoch {epoch + 1}/{args.epochs} | "
			f"Train Loss: {train_loss:.6f} | "
			f"Val Loss: {val_loss:.6f} | "
			f"LR: {optimizer.param_groups[0]['lr']:.2e}"
		)

		if val_loss < best_val_loss:
			best_val_loss = val_loss
			best_paths = _save_checkpoint(model, args.output_dir)
			print(f"Saved best checkpoint to {best_paths[0]} and {best_paths[1]}")

	if best_paths is None:
		best_paths = _save_checkpoint(model, args.output_dir)

	print(f"Training complete. Best validation loss: {best_val_loss:.6f}")
	print(f"Model checkpoints written to: {best_paths[0]} and {best_paths[1]}")


def main() -> None:
	args = _parse_args()
	if not args.data_dir.exists():
		raise SystemExit(f"Dataset directory not found: {args.data_dir}")
	train(args)


if __name__ == "__main__":
	main()