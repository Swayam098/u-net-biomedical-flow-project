from flask import Flask, request, jsonify
from PIL import Image
import numpy as np
import time
from typing import cast

from utils.preprocessing import preprocess_image
from utils.postprocessing import postprocess_image
from utils.metrics import compute_psnr, compute_ssim
from utils.svd_baseline import svd_filter_image

from backend.model_loader import load_unet_model
from backend.inference import run_unet_inference


app = Flask(__name__)

# -----------------------------
# Load model once at startup
# -----------------------------
model = load_unet_model("backend/models/unet_model.pth")



@app.route("/predict", methods=["POST"])
def predict():
    try:
        start_time = time.time()

        if "file" not in request.files:
            return jsonify({"error": "No file uploaded"}), 400

        file = request.files["file"]
        image = Image.open(file.stream).convert("L")

        # -----------------------------
        # Preprocessing
        # -----------------------------
        input_np: np.ndarray = preprocess_image(image)

        # -----------------------------
        # Inference
        # -----------------------------
        unet_output: np.ndarray = run_unet_inference(model, input_np)
        svd_output: np.ndarray = svd_filter_image(input_np)

        # -----------------------------
        # Metrics (TYPE ASSERTION – FINAL FIX)
        # -----------------------------
        psnr = cast(float, compute_psnr(input_np, unet_output))
        ssim = cast(float, compute_ssim(input_np, unet_output))

        runtime = cast(float, time.time() - start_time)

        return jsonify({
            "unet_output": unet_output.astype(np.float64).tolist(),
            "svd_output": svd_output.astype(np.float64).tolist(),
            "psnr": psnr,
            "ssim": ssim,
            "runtime": runtime
        })

    except Exception as e:
        print("❌ Backend error:", e)
        return jsonify({"error": "Internal server error"}), 500


if __name__ == "__main__":
    app.run(debug=True)
