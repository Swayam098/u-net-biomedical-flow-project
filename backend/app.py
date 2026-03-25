from flask import Flask, request, jsonify
from PIL import Image
import numpy as np
import time
from typing import cast
import torch

from utils.preprocessing import preprocess_image
from utils.postprocessing import postprocess_image
from utils.metrics import compute_psnr, compute_ssim
from utils.svd_baseline import svd_filter_image

from backend.model_loader import load_unet_model
from backend.inference import run_unet_inference
from backend.optimization import OptimizedUNetInference


app = Flask(__name__)

# Configuration
USE_OPTIMIZED = True  # Set to True to use TorchScript + FP16
USE_FP16 = torch.cuda.is_available()  # Only on CUDA
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# Load model once at startup
print(f"\n🚀 Initializing backend...")
print(f"📱 Device: {DEVICE.upper()}")
print(f"⚡ Optimization: {'ENABLED' if USE_OPTIMIZED else 'DISABLED'}")
print(f"🔢 FP16 Support: {'YES' if USE_FP16 else 'NO'}")

model = load_unet_model("backend/models/unet_model.pth")

# Initialize optimization if enabled
optimized_inference = None
if USE_OPTIMIZED:
    optimized_inference = OptimizedUNetInference(model, device=DEVICE)
    # Try to compile TorchScript
    try:
        sample = np.random.rand(256, 256).astype(np.float32)
        optimized_inference.compile_torchscript(sample)
        print("✅ TorchScript optimization enabled\n")
    except Exception as e:
        print(f"⚠️ TorchScript compilation failed: {e}\n")
        optimized_inference = None



@app.route("/predict", methods=["POST"])
def predict():
    try:
        start_time = time.time()

        if "file" not in request.files:
            return jsonify({"error": "No file uploaded"}), 400

        file = request.files["file"]
        image = Image.open(file.stream).convert("L")

        # Preprocessing
        input_np: np.ndarray = preprocess_image(image)

        # Inference - use optimized version if available
        if optimized_inference is not None and USE_FP16:
            unet_output: np.ndarray = optimized_inference.inference_fp16(input_np)
        elif optimized_inference is not None:
            unet_output: np.ndarray = optimized_inference.inference_fp32(input_np)
        else:
            unet_output: np.ndarray = run_unet_inference(model, input_np)
        
        svd_output: np.ndarray = svd_filter_image(input_np)

        # Metrics
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


@app.route("/batch", methods=["POST"])
def batch_predict():
    """Batch prediction endpoint for multiple images"""
    try:
        if "files" not in request.files:
            return jsonify({"error": "No files uploaded"}), 400
        
        files = request.files.getlist("files")
        results = []
        total_time = time.time()
        
        for file in files:
            start_time = time.time()
            image = Image.open(file.stream).convert("L")
            input_np = preprocess_image(image)
            
            # Use optimized inference
            if optimized_inference is not None and USE_FP16:
                unet_output = optimized_inference.inference_fp16(input_np)
            elif optimized_inference is not None:
                unet_output = optimized_inference.inference_fp32(input_np)
            else:
                unet_output = run_unet_inference(model, input_np)
            
            runtime = time.time() - start_time
            
            results.append({
                "filename": file.filename,
                "unet_output": unet_output.astype(np.float64).tolist(),
                "runtime": runtime
            })
        
        total_time = time.time() - total_time
        
        return jsonify({
            "results": results,
            "total_time": total_time,
            "count": len(results)
        })
    
    except Exception as e:
        print("❌ Batch processing error:", e)
        return jsonify({"error": "Batch processing failed"}), 500


@app.route("/stats", methods=["GET"])
def get_stats():
    """Get optimization statistics"""
    return jsonify({
        "device": DEVICE,
        "optimization": "ENABLED" if USE_OPTIMIZED else "DISABLED",
        "fp16_support": USE_FP16,
        "torchscript": optimized_inference is not None and optimized_inference.jit_model is not None
    })


if __name__ == "__main__":
    app.run(debug=True)
