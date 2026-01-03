from flask import Flask, request, jsonify
from PIL import Image
import numpy as np
import time
import io

from utils.preprocessing import preprocess_image
from utils.postprocessing import postprocess_image
from utils.metrics import compute_psnr, compute_ssim
from utils.svd_baseline import svd_filter_image

from backend.model_loader import load_unet_model
from backend.inference import run_unet_inference


app = Flask(__name__)

# Load model once at startup
model = load_unet_model("models/unet_model.pth")

@app.route("/predict", methods=["POST"])
def predict():
    try:
        start_time = time.time()

        if "file" not in request.files:
            return jsonify({"error": "No file uploaded"}), 400

        file = request.files["file"]
        image = Image.open(file.stream).convert("L")

        input_np = preprocess_image(image)
        unet_output = run_unet_inference(model, input_np)
        svd_output = svd_filter_image(input_np)

        psnr = compute_psnr(input_np, unet_output)
        ssim = compute_ssim(input_np, unet_output)

        runtime = time.time() - start_time

        return jsonify({
            "unet_output": unet_output.astype(float).tolist(),
            "svd_output": svd_output.astype(float).tolist(),
            "psnr": float(psnr),
            "ssim": float(ssim),
            "runtime": float(runtime)
        })

    except Exception as e:
        print("❌ Backend error:", e)
        return jsonify({"error": "Internal server error"}), 500

if __name__ == "__main__":
    app.run(debug=True)
