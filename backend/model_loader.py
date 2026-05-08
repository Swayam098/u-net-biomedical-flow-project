from pathlib import Path
from typing import Optional

import torch

from backend.inference import UNet


def resolve_device(device: Optional[str] = None) -> torch.device:
	if device:
		return torch.device(device)
	return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def load_unet_model(
	model_path: str,
	device: Optional[str] = None,
	dropout_p: float = 0.1,
) -> UNet:
	"""Load a trained checkpoint if available; otherwise return initialized model."""
	target_device = resolve_device(device)
	model = UNet(dropout_p=dropout_p).to(target_device)
	model.checkpoint_loaded = False
	model.checkpoint_path = str(model_path)

	requested_path = Path(model_path)
	candidate_paths = [
		requested_path,
		requested_path.parent.parent / "results" / "unet_hybrid.pth",
		requested_path.parent.parent / "backend" / "models" / "unet_model.pth",
		requested_path.parent.parent / "models" / "unet_model.pth",
		requested_path.parent.parent / "checkpoint" / "best_unet_model.pt",
		requested_path.parent.parent / "checkpoint" / "best_unet_model.pth",
	]

	path = next((candidate for candidate in candidate_paths if candidate.exists()), None)
	if path is None:
		searched = ", ".join(str(candidate) for candidate in candidate_paths)
		print(f"[model_loader] Model file not found. Searched: {searched}. Using initialized weights.")
		model.eval()
		return model

	try:
		state_dict = torch.load(path, map_location=target_device, weights_only=True)
		model.load_state_dict(state_dict, strict=False)
		model.checkpoint_loaded = True
		model.checkpoint_path = str(path)
		print(f"[model_loader] Loaded model checkpoint: {path}")
	except TypeError:
		state_dict = torch.load(path, map_location=target_device)
		model.load_state_dict(state_dict, strict=False)
		model.checkpoint_loaded = True
		model.checkpoint_path = str(path)
		print(f"[model_loader] Loaded model checkpoint (fallback): {path}")
	except Exception as exc:  # pragma: no cover - defensive path
		print(f"[model_loader] Failed to load checkpoint ({exc}). Using initialized weights.")

	model.eval()
	return model

