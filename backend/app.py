import base64
import io
import sys
from pathlib import Path
from typing import Any, Dict

import numpy as np
import torch
from flask import Flask, jsonify, request
from PIL import Image

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
	sys.path.insert(0, str(PROJECT_ROOT))

from backend.inference import estimate_memory_mb, parameter_count
from backend.model_loader import load_unet_model, resolve_device
from backend.optimization import OptimizationConfig, OptimizedUNetInference


app = Flask(__name__)

MODEL_PATH = str(PROJECT_ROOT / "results" / "unet_hybrid.pth")
DEVICE = resolve_device()

OPT_CONFIG = OptimizationConfig(
	use_torchscript=True,
	use_fp16=torch.cuda.is_available(),
	use_wavelet=True,
	wavelet_name="haar",
	wavelet_threshold=0.035,
)

model = load_unet_model(MODEL_PATH, device=str(DEVICE))
optimized = OptimizedUNetInference(model, DEVICE, OPT_CONFIG)
optimized.compile_torchscript()


def _model_loaded() -> bool:
	return bool(getattr(model, "checkpoint_loaded", False))


def _model_checkpoint_path() -> str:
	return str(getattr(model, "checkpoint_path", MODEL_PATH))


def _decode_upload_to_numpy(file_storage) -> np.ndarray:
	image = Image.open(file_storage.stream).convert("L")
	return np.array(image, dtype=np.float32)


def _encode_image(img: np.ndarray) -> str:
	arr = np.clip(img * 255.0, 0, 255).astype(np.uint8)
	pil_img = Image.fromarray(arr)
	buf = io.BytesIO()
	pil_img.save(buf, format="PNG")
	return base64.b64encode(buf.getvalue()).decode("utf-8")


@app.get("/health")
def health() -> Any:
	loaded = _model_loaded()
	return jsonify(
		{
			"status": "ok",
			"model_loaded": loaded,
			"model_path": _model_checkpoint_path(),
			"warning": None if loaded else "Model checkpoint is missing. Predictions use untrained weights.",
		}
	)


@app.get("/stats")
def stats() -> Any:
	total, trainable = parameter_count(model)
	return jsonify(
		{
			"device": str(DEVICE),
			"model_loaded": _model_loaded(),
			"model_path": _model_checkpoint_path(),
			"optimization": {
				"torchscript": bool(optimized.torchscript_model is not None),
				"fp16": bool(OPT_CONFIG.use_fp16 and DEVICE.type == "cuda"),
				"wavelet": OPT_CONFIG.use_wavelet,
				"wavelet_name": OPT_CONFIG.wavelet_name,
				"wavelet_threshold": OPT_CONFIG.wavelet_threshold,
			},
			"model": {
				"total_params": total,
				"trainable_params": trainable,
				"estimated_memory_mb": round(estimate_memory_mb(), 2),
			},
		}
	)


@app.post("/predict")
def predict() -> Any:
	file = request.files.get("file")
	if file is None:
		return jsonify({"error": "Missing file field in multipart form data."}), 400

	image_np = _decode_upload_to_numpy(file)
	result = optimized.predict(image_np=image_np, mc_samples=1)

	response: Dict[str, Any] = {
		"prediction": _encode_image(result["prediction"]),
		"runtime_sec": float(result["runtime"]),
		"optimized": bool(result["optimized"]),
		"fp16": bool(result["fp16"]),
		"model_loaded": _model_loaded(),
		"model_path": _model_checkpoint_path(),
	}
	return jsonify(response)


@app.post("/predict_uncertainty")
def predict_uncertainty() -> Any:
	file = request.files.get("file")
	if file is None:
		return jsonify({"error": "Missing file field in multipart form data."}), 400

	try:
		mc_samples = int(request.form.get("mc_samples", "10"))
	except ValueError:
		return jsonify({"error": "mc_samples must be an integer."}), 400

	mc_samples = max(2, min(mc_samples, 50))
	image_np = _decode_upload_to_numpy(file)
	result = optimized.predict(image_np=image_np, mc_samples=mc_samples)

	uncertainty = result["uncertainty"]
	uncertainty_max = float(np.max(uncertainty))
	if uncertainty_max <= 1e-8:
		uncertainty_norm = np.zeros_like(uncertainty, dtype=np.float32)
	else:
		uncertainty_norm = uncertainty / uncertainty_max

	response: Dict[str, Any] = {
		"prediction": _encode_image(result["prediction"]),
		"uncertainty_map": _encode_image(uncertainty_norm),
		"uncertainty_mean": float(np.mean(uncertainty)),
		"mc_samples": mc_samples,
		"runtime_sec": float(result["runtime"]),
		"model_loaded": _model_loaded(),
		"model_path": _model_checkpoint_path(),
	}
	return jsonify(response)


if __name__ == "__main__":
	app.run(host="0.0.0.0", port=5000, debug=True)

