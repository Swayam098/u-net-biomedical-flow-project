# U-Net Biomedical Flow Project

## Novelty Implementation (Phase 1)

This repository now includes the first implementation stage for research novelty:

- Wavelet-aware inference pre-processing (`haar` + soft threshold)
- Frequency-weighted training loss scaffold
- Monte Carlo dropout uncertainty inference endpoint

### Backend endpoints

- `GET /health` : health check
- `GET /stats` : model/optimization status and deployment metadata
- `POST /predict` : standard denoising inference
- `POST /predict_uncertainty` : denoising + uncertainty map (`mc_samples` form field)

### Quick run

```powershell
python -m backend.app
```

### Quick test

```powershell
curl http://127.0.0.1:5000/health
curl http://127.0.0.1:5000/stats
```

```powershell
curl -X POST http://127.0.0.1:5000/predict \
	-F "file=@data/images/sample.png"
```

```powershell
curl -X POST http://127.0.0.1:5000/predict_uncertainty \
	-F "file=@data/images/sample.png" \
	-F "mc_samples=12"
```

