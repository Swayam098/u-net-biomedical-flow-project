# 📦 Requirements & Installation Guide

## Quick Overview

| File | Purpose | Install Command |
|------|---------|-----------------|
| **requirements.txt** | Production (CPU/GPU auto-detect) | `pip install -r requirements.txt` |
| **requirements-dev.txt** | Development tools + testing | `pip install -r requirements-dev.txt` |

---

## 🚀 Installation (Choose One)

### Option 1: Full Stack (Recommended for Most Users)

**For CPU (works everywhere, slower):**
```bash
pip install -r requirements.txt
```

**For GPU (NVIDIA CUDA 11.8+, 10x faster):**
```bash
# First: Install CUDA 11.8+ from NVIDIA website
# Then: Install GPU PyTorch
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118

# Finally: Install remaining packages
pip install -r requirements.txt
```

**Check GPU availability:**
```bash
python -c "import torch; print('GPU Available:', torch.cuda.is_available())"
```

---



---

## 📊 Package Summary

**Core ML (5 packages):**
- torch, torchvision, numpy, scipy

**Image Processing (4 packages):**
- opencv-python, Pillow, scikit-image, imageio

**Web Frameworks (5 packages):**
- streamlit, streamlit-option-menu, Flask, Werkzeug, requests

**Visualization (6 packages):**
- matplotlib, plotly, reportlab, python-docx, pypdf, pandas

**Utilities (3 packages):**
- scikit-learn, python-dotenv, tqdm

**Total: ~30 production packages**

---

## ⚡ Performance Comparison

| Setup | Speed | Pros | Cons |
|-------|-------|------|------|
| **CPU** | ~3-5s/image | Works everywhere, simple | Slow for production |
| **GPU (CUDA)** | ~0.25s/image | 10x faster, production-ready | Requires NVIDIA GPU |

---

## 🔧 Common Commands

### Verify Installation
```bash
python -c "
import torch
import cv2
import streamlit
import flask
print('✅ All core packages loaded')
print(f'GPU Available: {torch.cuda.is_available()}')
"
```

### Upgrade Packages
```bash
pip install --upgrade -r requirements.txt
```

### Create Environment for Sharing
```bash
pip freeze > requirements-lock.txt
```

### List Installed Packages
```bash
pip list
```

---

## 📝 Notes

- **GPU Installation**: CUDA must be installed separately before running the PyTorch GPU install command
- **CPU Mode**: Autodetected if GPU not available; no special action needed
- **Development**: Use `requirements-dev.txt` for testing, linting, and profiling
- **Single File**: One `requirements.txt` installs everything (frontend + backend)

---

## ❓ Troubleshooting

**GPU not detected?**
```bash
python -c "import torch; print(torch.cuda.is_available())"
# If False, CUDA not installed or PyTorch CUDA version mismatch
```

**Import errors?**
```bash
# Reinstall cleanly
pip cache purge
pip install -r requirements.txt --force-reinstall
```

**Version conflicts?**
```bash
# Start fresh with virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
pip install -r requirements.txt
```

---

## 📚 Related Files

- **README.md** - Project overview and features
- **INSTALL.md** - 30-second quick start
- **OPTIMIZATION_GUIDE.md** - Performance tuning
- **TROUBLESHOOTING.md** - Common errors and fixes
