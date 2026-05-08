from __future__ import annotations

import csv
import json
from pathlib import Path

import numpy as np
import torch
from PIL import Image

from backend.inference import UNet


ROOT = Path(__file__).resolve().parent
RESULTS_DIR = ROOT / "results"
TABLES_DIR = RESULTS_DIR / "tables"
PLOTS_DIR = RESULTS_DIR / "plots"
IMAGES_DIR = RESULTS_DIR / "images"
MODEL_PATH = RESULTS_DIR / "unet_hybrid.pth"
DATA_DIR = ROOT / "data" / "images"


def load_gray_tensor(path: Path, size: int = 320) -> torch.Tensor:
	img = Image.open(path).convert("L").resize((size, size))
	arr = np.asarray(img, dtype=np.float32) / 255.0
	return torch.from_numpy(arr).unsqueeze(0).unsqueeze(0)


def add_speckle_noise(img: torch.Tensor, var: float) -> torch.Tensor:
	noise = torch.randn_like(img) * var
	return torch.clamp(img + img * noise, 0.0, 1.0)


def psnr(pred: torch.Tensor, target: torch.Tensor, eps: float = 1e-12) -> float:
	mse = torch.nn.functional.mse_loss(pred, target).item()
	mse = max(mse, eps)
	return float(10.0 * np.log10(1.0 / mse))


def check_exists(path: Path) -> None:
	if not path.exists():
		raise FileNotFoundError(f"Missing expected artifact: {path}")


def check_csv(path: Path) -> int:
	check_exists(path)
	with path.open(newline="", encoding="utf-8") as handle:
		rows = list(csv.reader(handle))
	if len(rows) < 2:
		raise RuntimeError(f"CSV has no data rows: {path}")
	return len(rows) - 1


def main() -> None:
	print("Running novelty pipeline smoke test...")
	device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
	print("Device:", device)

	# File checks
	baseline_rows = check_csv(TABLES_DIR / "baseline_metrics.csv")
	robustness_rows = check_csv(TABLES_DIR / "robustness.csv")
	comparison_rows = check_csv(TABLES_DIR / "comparison_table.csv")
	check_exists(PLOTS_DIR / "robustness.png")
	check_exists(IMAGES_DIR / "before_after.png")
	check_exists(IMAGES_DIR / "diff_map.png")
	check_exists(MODEL_PATH)
	check_exists(RESULTS_DIR / "run_summary.json")

	with (RESULTS_DIR / "run_summary.json").open(encoding="utf-8") as handle:
		summary = json.load(handle)

	print("Baseline rows:", baseline_rows)
	print("Robustness rows:", robustness_rows)
	print("Comparison rows:", comparison_rows)
	print("Summary device:", summary.get("device"))
	print("Summary image size:", summary.get("image_size"))

	if not DATA_DIR.exists():
		raise FileNotFoundError(f"Missing dataset directory: {DATA_DIR}")

	image_paths = [
		p for p in sorted(DATA_DIR.iterdir())
		if p.suffix.lower() in {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff"} and "mask" not in p.name.lower()
	]
	if not image_paths:
		raise RuntimeError("No valid ultrasound images found for inference check.")

	sample = load_gray_tensor(image_paths[0], size=int(summary.get("image_size", 320)))
	noisy = add_speckle_noise(sample, 0.4)

	model = UNet().to(device)
	state_dict = torch.load(MODEL_PATH, map_location=device)
	model.load_state_dict(state_dict, strict=False)
	model.eval()

	with torch.no_grad():
		pred = model(noisy.to(device))

	if pred.shape != sample.to(device).shape:
		raise RuntimeError(f"Unexpected model output shape: {pred.shape}")

	print("Inference check PSNR:", f"{psnr(pred, sample.to(device)):.4f}")
	print("Smoke test passed.")


if __name__ == "__main__":
	main()
