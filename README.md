# Ultrasound Image Enhancement Using U-Net with Uncertainty Estimation

<div align="center">

![Python](https://img.shields.io/badge/Python-3.10%2B-blue?logo=python&logoColor=white) 
![PyTorch](https://img.shields.io/badge/PyTorch-2.0%2B-red?logo=pytorch&logoColor=white) 
![Streamlit](https://img.shields.io/badge/Streamlit-1.0%2B-green?logo=streamlit&logoColor=white) 
![Status](https://img.shields.io/badge/Status-Research%20Demo-brightgreen)

**Speckle-noise suppression and uncertainty-aware denoising for ultrasound imaging using a compact U‑Net.**


</div>

## TL;DR

High-quality denoising for biomedical ultrasound images with fast inference, optional wavelet preprocessing, and Monte Carlo dropout uncertainty estimates. Includes a Flask API, Streamlit demo, evaluation notebooks, and GitHub-ready visual assets.

## Why this project

- Improves visual quality of ultrasound scans while preserving anatomical detail.
- Provides per-pixel uncertainty maps to help interpret model outputs in clinical or research workflows.
- Designed for reproducibility: notebooks, exported figures, and lightweight model checkpoints are included for quick demos.

## Key Features

- Compact U‑Net for real-time-ish inference on CPU/GPU
- Wavelet-aware preprocessing (Haar soft-thresholding) for improved baseline robustness
- Monte Carlo dropout for uncertainty quantification
- Streamlit app for interactive inspection and report export
- End-to-end Flask endpoints for integration into pipelines or demos

## Quick Demo

Run the backend and open the Streamlit app to try the interactive demo and export PDF/PNG reports.

Start the backend:

```powershell
python -m backend.app
```

Start the frontend (Streamlit):

```powershell
streamlit run frontend/streamlit_app.py
```

API examples:

```powershell
curl http://127.0.0.1:5000/health
curl -X POST http://127.0.0.1:5000/predict -F "file=@data/images/sample.png"
curl -X POST http://127.0.0.1:5000/predict_uncertainty -F "file=@data/images/sample.png" -F "mc_samples=12"
```

## Representative Results
Open the `assets/screenshots` folder for publication-ready comparisons: input, SVD baseline, and U‑Net output images, plus metric plots.

<table>
	<tr>
		<td align="center">
			<img src="assets/screenshots/input.jpg" alt="Input ultrasound image" width="260" />
			<br /><strong>Input</strong>
		</td>
		<td align="center">
			<img src="assets/screenshots/svd_output.jpg" alt="SVD baseline output" width="260" />
			<br /><strong>SVD Baseline</strong>
		</td>
		<td align="center">
			<img src="assets/screenshots/unet_output.jpg" alt="U-Net output" width="260" />
			<br /><strong>U-Net Output</strong>
		</td>
	</tr>
	<tr>
		<td colspan="3" align="center">
			<img src="assets/screenshots/metrics.png" alt="Quality metrics" width="780" />
			<br /><strong>Quality Metrics</strong>
		</td>
	</tr>
</table>

## Additional Results 

A selection of exported visuals from recent experiments and run summaries.

<table>
	<tr>
		<td align="center">
			<img src="results/images/preview_20260423T133845Z.png" alt="Preview 1" width="300" />
			<br /><strong>Preview — run 133845</strong>
		</td>
		<td align="center">
			<img src="results/images/preview_20260423T133948Z.png" alt="Preview 2" width="300" />
			<br /><strong>Preview — run 133948</strong>
		</td>
	</tr>
	<tr>
		<td align="center">
			<img src="results/images/before_after (1).png" alt="Before / After" width="300" />
			<br /><strong>Before / After</strong>
		</td>
		<td align="center">
			<img src="results/images/diff_map.png" alt="Difference Map" width="300" />
			<br /><strong>Difference Map</strong>
		</td>
	</tr>
	<tr>
		<td align="center">
			<img src="results/images/robustness (1).png" alt="Robustness Plot" width="300" />
			<br /><strong>Robustness</strong>
		</td>
		<td align="center">
			<img src="results/images/System architecture diagram.png" alt="System Architecture" width="300" />
			<br /><strong>System Architecture</strong>
		</td>
	</tr>
</table>

- Input | SVD baseline | U‑Net output
- Metric plots (PSNR / SSIM / custom metrics)

## Repository Structure

- `backend/` — Flask API, inference, training helpers, and optimization utilities
- `frontend/` — Streamlit demo, report export helpers
- `docs/` — architecture diagrams and supplementary figures
- `notebooks/` — experiments, training, and evaluation notebooks
- `assets/` — screenshots and figures used for README and reports

## Reproducibility & Experiments

See `notebooks/unet_training.ipynb` and `experiments/notebook_run/exports` for training configs, exported results, and run summaries. Check `requirements-dev.txt` and `requirements.txt` for environment specs.

## How to Cite / Acknowledge

If you use this work in research, please acknowledge the repository and include a short description of the preprocessing and uncertainty approach used.

## Contributing

Contributions and issues welcome. Please open a GitHub issue for feature requests or pull requests with focused changes and tests/examples.

## License & Contact

This repository is distributed for research and demonstration. See `LICENSE` for details (if present). For questions or professional inquiries, open an issue or contact the maintainer.


