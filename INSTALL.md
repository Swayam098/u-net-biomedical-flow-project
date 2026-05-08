# ⚡ Quick Installation Guide

## 🚀 5-Minute Setup (GPU)

```bash
# 1. Clone repository
git clone https://github.com/Swayam098/u-net-biomedical-flow-project.git
cd u-net-biomedical-flow-project

# 2. Create virtual environment
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate

# 3. Install GPU-optimized PyTorch (requires CUDA 11.8+)
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118

# 4. Install dependencies
pip install -r requirements.txt

# 5. Run the application
# Terminal 1: Backend
python -m backend.app

# Terminal 2: Frontend
streamlit run frontend/streamlit_app.py

# 6. Open browser
# Navigate to: http://localhost:8501
```

## 💻 CPU-Only Setup (3 minutes)

```bash
# 1-2. Same as above

# 3. Install CPU requirements (skip GPU PyTorch step)
pip install -r requirements.txt

# 4-6. Same as above (but slower: ~3-5s per image)
```

## 📦 What Gets Installed

### Backend (Flask API)
- torch, torchvision - Deep learning
- numpy, scipy - Numerical computing
- opencv-python, Pillow - Image processing
- Flask - Web API
- scikit-image - Image metrics

### Frontend (Streamlit)
- streamlit - Web interface
- matplotlib, plotly - Visualization
- reportlab - PDF generation
- requests - HTTP client

### Shared
- pandas - Data handling
- scikit-learn - ML utilities
- python-dotenv - Configuration

## ✅ Verify Installation

```bash
python -c "
import torch
import streamlit
import cv2
import flask
print('✅ All core packages installed')
print(f'GPU Available: {torch.cuda.is_available()}')
"
```

## 🎯 Next Steps

1. **Read** [README.md](README.md) for full documentation
2. **Upload** an ultrasound image
3. **Explore** all features
4. **Export** results as PDF/PNG
5. **Share** feedback

## 🆘 Issues?

- See [REQUIREMENTS_SIMPLE.md](REQUIREMENTS_SIMPLE.md) for detailed info
- Check [TROUBLESHOOTING.md](TROUBLESHOOTING.md) for common fixes
- GPU not working? Just use `pip install -r requirements.txt` for CPU mode

---

**Status:** Ready to use ✅
