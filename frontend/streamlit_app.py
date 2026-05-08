import base64
import io
import json
from pathlib import Path
from typing import Any, Dict

import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
import numpy as np
import pandas as pd
import requests
import streamlit as st
from PIL import Image


st.set_page_config(page_title="Ultrasound Image Enhancement Using U-Net with Uncertainty Estimation", page_icon="🧬", layout="wide")


PROJECT_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_RESULTS_DIR = PROJECT_ROOT / "results"


def decode_base64_image(encoded: str) -> Image.Image:
	raw = base64.b64decode(encoded)
	return Image.open(io.BytesIO(raw)).convert("L")


def call_health(base_url: str) -> tuple[bool, str]:
	try:
		r = requests.get(f"{base_url}/health", timeout=5)
		if r.status_code == 200:
			payload = r.json()
			if payload.get("model_loaded", True):
				return True, "Backend connected"
			warn = payload.get("warning", "Model checkpoint missing. Running with untrained weights.")
			return True, f"Backend connected with warning: {warn}"
		return False, f"Backend responded with status {r.status_code}"
	except Exception as exc:
		return False, str(exc)


def call_predict(base_url: str, file_bytes: bytes, filename: str) -> Dict[str, Any]:
	files = {"file": (filename, file_bytes, "image/png")}
	r = requests.post(f"{base_url}/predict", files=files, timeout=60)
	r.raise_for_status()
	return r.json()


def call_predict_uncertainty(base_url: str, file_bytes: bytes, filename: str, mc_samples: int) -> Dict[str, Any]:
	files = {"file": (filename, file_bytes, "image/png")}
	data = {"mc_samples": str(mc_samples)}
	r = requests.post(f"{base_url}/predict_uncertainty", files=files, data=data, timeout=120)
	r.raise_for_status()
	return r.json()


def load_csv_if_exists(path: Path) -> pd.DataFrame | None:
	if not path.exists():
		return None
	return pd.read_csv(path)


def load_json_if_exists(path: Path) -> dict[str, Any] | None:
	if not path.exists():
		return None
	with path.open(encoding="utf-8") as handle:
		return json.load(handle)


def image_to_array(image: Image.Image) -> np.ndarray:
	return np.asarray(image.convert("L"), dtype=np.float32) / 255.0


def array_to_png_bytes(image_np: np.ndarray) -> bytes:
	buf = io.BytesIO()
	Image.fromarray(np.clip(image_np * 255.0, 0, 255).astype(np.uint8)).save(buf, format="PNG")
	return buf.getvalue()


def add_speckle_noise(image_np: np.ndarray, noise_level: float, seed: int) -> np.ndarray:
	rng = np.random.default_rng(seed)
	noise = rng.normal(0.0, noise_level, size=image_np.shape).astype(np.float32)
	return np.clip(image_np * (1.0 + noise), 0.0, 1.0)


def compute_psnr(reference: np.ndarray, estimate: np.ndarray) -> float:
	mse = float(np.mean((reference - estimate) ** 2))
	if mse <= 1e-12:
		return float("inf")
	return float(20.0 * np.log10(1.0 / np.sqrt(mse)))


def compute_mae(reference: np.ndarray, estimate: np.ndarray) -> float:
	return float(np.mean(np.abs(reference - estimate)))


def compute_ssim(reference: np.ndarray, estimate: np.ndarray) -> float:
	"""Simple global SSIM for grayscale images in [0, 1]."""
	ref = reference.astype(np.float32)
	est = estimate.astype(np.float32)
	mu_x = float(ref.mean())
	mu_y = float(est.mean())
	sigma_x = float(ref.var())
	sigma_y = float(est.var())
	cov = float(((ref - mu_x) * (est - mu_y)).mean())
	c1 = 0.01 ** 2
	c2 = 0.03 ** 2
	numerator = (2.0 * mu_x * mu_y + c1) * (2.0 * cov + c2)
	denominator = (mu_x**2 + mu_y**2 + c1) * (sigma_x + sigma_y + c2)
	if denominator <= 1e-12:
		return 1.0 if abs(mu_x - mu_y) <= 1e-12 else 0.0
	return float(np.clip(numerator / denominator, 0.0, 1.0))


def create_report_pdf(
	image_name: str,
	original_image: Image.Image,
	noise_level: float,
	selected_noisy_np: np.ndarray,
	deterministic_pred: Image.Image,
	deterministic_result: Dict[str, Any],
	quality_df: pd.DataFrame,
	unc_result: Dict[str, Any] | None = None,
	unc_pred: Image.Image | None = None,
	unc_img: Image.Image | None = None,
) -> bytes:
	buffer = io.BytesIO()
	with PdfPages(buffer) as pdf:
		fig = plt.figure(figsize=(11, 8.5), constrained_layout=True)
		grid = fig.add_gridspec(3, 3, height_ratios=[0.18, 0.52, 0.30])
		fig.suptitle(f"Ultrasound Denoising Report: {image_name}", fontsize=18, fontweight="bold")
		fig.text(0.06, 0.95, f"Noise level: {noise_level:.2f}")
		fig.text(0.34, 0.95, f"Model: {Path(deterministic_result.get('model_path', 'unknown')).name}")
		fig.text(0.68, 0.95, f"Runtime (s): {float(deterministic_result.get('runtime_sec', 0.0)):.4f}")

		ax = fig.add_subplot(grid[1, 0])
		ax.imshow(original_image, cmap="gray")
		ax.set_title("Original")
		ax.axis("off")

		ax = fig.add_subplot(grid[1, 1])
		ax.imshow(Image.fromarray(np.clip(selected_noisy_np * 255.0, 0, 255).astype(np.uint8)), cmap="gray")
		ax.set_title("Noisy Input")
		ax.axis("off")

		ax = fig.add_subplot(grid[1, 2])
		ax.imshow(deterministic_pred, cmap="gray")
		ax.set_title("Enhanced Output")
		ax.axis("off")

		ax = fig.add_subplot(grid[2, :])
		ax.axis("off")
		table = ax.table(cellText=quality_df.round(4).values, colLabels=quality_df.columns, loc="center")
		table.auto_set_font_size(False)
		table.set_fontsize(10)
		table.scale(1.0, 1.45)
		ax.set_title("Quality Metrics Summary", pad=14)
		pdf.savefig(fig)
		plt.close(fig)

		fig, ax = plt.subplots(figsize=(11, 8.5), constrained_layout=True)
		comparison_labels = ["Noisy", "Enhanced"]
		difference_values = [
			compute_mae(image_to_array(original_image.resize((256, 256), Image.BILINEAR)), selected_noisy_np),
			compute_mae(image_to_array(original_image.resize((256, 256), Image.BILINEAR)), image_to_array(deterministic_pred.resize((256, 256), Image.BILINEAR))),
		]
		bars = ax.bar(comparison_labels, difference_values, color=["#d95f02", "#1b9e77"])
		ax.set_title("Original vs Enhanced Difference", pad=18)
		ax.set_ylabel("Mean Absolute Difference")
		ax.text(0.5, 0.96, "Lower values indicate closer agreement with the original image", transform=ax.transAxes, ha="center", fontsize=10)
		for bar, value in zip(bars, difference_values):
			ax.text(bar.get_x() + bar.get_width() / 2, value, f"{value:.4f}", ha="center", va="bottom")
		fig.suptitle("Difference Comparison", fontsize=18, fontweight="bold")
		pdf.savefig(fig, bbox_inches="tight")
		plt.close(fig)

		if unc_result is not None and unc_pred is not None and unc_img is not None:
			fig, axes = plt.subplots(1, 3, figsize=(11, 8.5), constrained_layout=True)
			fig.suptitle("Uncertainty Analysis", fontsize=18, fontweight="bold")
			axes[0].imshow(Image.fromarray(np.clip(selected_noisy_np * 255.0, 0, 255).astype(np.uint8)), cmap="gray")
			axes[0].set_title("Noisy Input")
			axes[1].imshow(unc_pred, cmap="gray")
			axes[1].set_title("Denoised")
			axes[2].imshow(unc_img, cmap="gray")
			axes[2].set_title("Uncertainty")
			for axis in axes:
				axis.axis("off")
			pdf.savefig(fig, bbox_inches="tight")
			plt.close(fig)

	buffer.seek(0)
	return buffer.getvalue()


st.title("Ultrasound Image Enhancement Using U-Net with Uncertainty Estimation")
st.caption("Wavelet-aware denoising, uncertainty visualization, and novelty experiment reporting")

with st.sidebar:
	st.header("Settings")
	backend_url = st.text_input("Backend URL", value="http://127.0.0.1:5000")
	results_dir = st.text_input("Results Dir", value=str(DEFAULT_RESULTS_DIR))
	mc_samples = st.slider("MC Samples (Uncertainty)", min_value=2, max_value=30, value=10, step=1)

	ok, msg = call_health(backend_url)
	if ok:
		st.success(msg)
	else:
		st.error(f"Backend unavailable: {msg}")

tab_analysis = st.tabs(["Image Analysis"])[0]

with tab_analysis:
	uploaded_file = st.file_uploader("Upload ultrasound image", type=["png", "jpg", "jpeg", "bmp"])

	if uploaded_file is None:
		st.info("Upload an image to run denoising and quality analysis.")
	else:
		file_bytes = uploaded_file.read()
		orig_img = Image.open(io.BytesIO(file_bytes)).convert("L")
		analysis_np = image_to_array(orig_img.resize((256, 256), Image.BILINEAR))
		noise_levels = [0.2, 0.4, 0.6]
		analysis_mc_samples = st.slider("MC samples for uncertainty", min_value=2, max_value=30, value=10, step=1)
		selected_noise = st.slider("Synthetic speckle level", min_value=0.05, max_value=0.8, value=0.4, step=0.05)
		robustness_plot = Path(results_dir) / "plots" / "robustness.png"

		col1, col2 = st.columns(2)
		with col1:
			st.subheader("Original")
			st.image(orig_img, use_container_width=True)

		with col2:
			st.subheader("Actions")
			run_basic = st.button("Run Denoising", use_container_width=True)
			run_unc = st.button("Run Denoising + Uncertainty", use_container_width=True)

		if run_basic or run_unc:
			try:
				selected_noisy_np = add_speckle_noise(analysis_np, selected_noise, seed=42)
				selected_noisy_bytes = array_to_png_bytes(selected_noisy_np)

				with st.spinner("Running analysis..."):
					deterministic_result = call_predict(backend_url, file_bytes, uploaded_file.name)
					deterministic_pred = decode_base64_image(deterministic_result["prediction"])
					deterministic_pred_np = image_to_array(deterministic_pred.resize((256, 256), Image.BILINEAR))

					unc_result = None
					unc_pred = None
					unc_img = None
					if run_unc:
						unc_result = call_predict_uncertainty(
							backend_url,
							selected_noisy_bytes,
							f"noisy_{uploaded_file.name}",
							analysis_mc_samples,
						)
						unc_pred = decode_base64_image(unc_result["prediction"])
						unc_img = decode_base64_image(unc_result["uncertainty_map"])

				if not deterministic_result.get("model_loaded", True):
					st.warning(f"Model checkpoint missing at: {deterministic_result.get('model_path', 'unknown path')}")

				st.markdown("### Quality Metrics")
				metrics_rows = [
					{
						"Image": "Original vs Denoised",
						"PSNR (dB)": compute_psnr(analysis_np, deterministic_pred_np),
						"SSIM": compute_ssim(analysis_np, deterministic_pred_np),
						"MAE": compute_mae(analysis_np, deterministic_pred_np),
					},
					{
						"Image": f"Noisy {selected_noise:.2f} vs Original",
						"PSNR (dB)": compute_psnr(analysis_np, selected_noisy_np),
						"SSIM": compute_ssim(analysis_np, selected_noisy_np),
						"MAE": compute_mae(analysis_np, selected_noisy_np),
					},
				]
				st.dataframe(pd.DataFrame(metrics_rows), use_container_width=True)

				if run_basic:
					c1, c2 = st.columns(2)
					with c1:
						st.subheader("Denoised")
						st.image(deterministic_pred, use_container_width=True)
					with c2:
						st.subheader("Inference Stats")
						st.metric("Runtime (s)", f"{deterministic_result.get('runtime_sec', 0):.4f}")
						st.write(f"Optimized: {deterministic_result.get('optimized', False)}")
						st.write(f"FP16: {deterministic_result.get('fp16', False)}")

				if run_unc and unc_result is not None and unc_pred is not None and unc_img is not None:
					st.markdown("### Uncertainty View")
					u1, u2, u3 = st.columns(3)
					with u1:
						st.subheader("Noisy Input")
						st.image(Image.fromarray(np.clip(selected_noisy_np * 255.0, 0, 255).astype(np.uint8)), use_container_width=True)
					with u2:
						st.subheader("Denoised")
						st.image(unc_pred, use_container_width=True)
					with u3:
						st.subheader("Uncertainty Map")
						st.image(unc_img, use_container_width=True)

					st.markdown("### Uncertainty Stats")
					m1, m2, m3 = st.columns(3)
					with m1:
						st.metric("MC Samples", int(unc_result.get("mc_samples", 0)))
					with m2:
						st.metric("Mean Uncertainty", f"{unc_result.get('uncertainty_mean', 0):.6f}")
					with m3:
						st.metric("Runtime (s)", f"{unc_result.get('runtime_sec', 0):.4f}")

				st.markdown("### Per-Image Robustness Sweep")
				rows = []
				for index, noise_level in enumerate(noise_levels):
					noisy_np = add_speckle_noise(analysis_np, noise_level, seed=100 + index)
					noisy_bytes = array_to_png_bytes(noisy_np)
					result = call_predict(backend_url, noisy_bytes, f"sweep_{noise_level}_{uploaded_file.name}")
					pred_np = image_to_array(decode_base64_image(result["prediction"]).resize((256, 256), Image.BILINEAR))
					rows.append(
						{
							"noise_level": noise_level,
							"psnr": compute_psnr(analysis_np, pred_np),
							"ssim": compute_ssim(analysis_np, pred_np),
							"mae": compute_mae(analysis_np, pred_np),
							"runtime_sec": float(result.get("runtime_sec", 0.0)),
						}
					)
				sweep_df = pd.DataFrame(rows)
				st.dataframe(sweep_df, use_container_width=True)

				st.markdown("### Robustness Plot")
				if robustness_plot.exists():
					st.image(str(robustness_plot), use_container_width=True)
				else:
					st.info(f"Missing: {robustness_plot}")

				diff_values = [
					compute_mae(analysis_np, selected_noisy_np),
					compute_mae(analysis_np, deterministic_pred_np),
				]
				st.markdown("### Difference Bar Graph")
				fig, ax = plt.subplots(figsize=(8, 4))
				bars = ax.bar(["Noisy", "Enhanced"], diff_values, color=["#d95f02", "#1b9e77"])
				ax.set_title("Difference Between Original and Enhanced Image")
				ax.set_ylabel("Mean Absolute Difference")
				for bar, value in zip(bars, diff_values):
					ax.text(bar.get_x() + bar.get_width() / 2, value, f"{value:.4f}", ha="center", va="bottom")
				st.pyplot(fig, clear_figure=True)

				pdf_bytes = create_report_pdf(
					image_name=uploaded_file.name,
					original_image=orig_img,
					noise_level=selected_noise,
					selected_noisy_np=selected_noisy_np,
					deterministic_pred=deterministic_pred,
					deterministic_result=deterministic_result,
					quality_df=pd.DataFrame(metrics_rows),
					unc_result=unc_result,
					unc_pred=unc_pred,
					unc_img=unc_img,
				)
				st.download_button(
					label="Download Report as PDF",
					data=pdf_bytes,
					file_name=f"{Path(uploaded_file.name).stem}_report.pdf",
					mime="application/pdf",
					use_container_width=True,
				)

				st.caption(
					"Run Denoising performs one deterministic forward pass. Run Denoising + Uncertainty adds Monte Carlo dropout over a noisy input, so it is slower and also returns an uncertainty map."
				)
			except Exception as exc:
				st.error(f"Analysis failed: {exc}")

