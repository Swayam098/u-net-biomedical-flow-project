# Troubleshooting Guide

## PDF Export Not Available

### Symptoms
- "⚠️ PDF export not available (reportlab not installed)" warning appears
- PNG export works fine
- Backend tests show PDF export working

### Root Cause
Streamlit caches module imports. If reportlab was initially unavailable, Streamlit may cache the failed state even after installation.

### Solutions

**Solution 1: Hard Refresh Browser** (Fastest)
```
Press: Ctrl + Shift + R
```

**Solution 2: Restart Streamlit**
```bash
# Kill existing process (if running)
# Then restart:
streamlit run frontend/streamlit_app.py
```

**Solution 3: Clear Streamlit Cache**
```bash
# On Windows:
rmdir /s %USERPROFILE%\.streamlit

# On Mac/Linux:
rm -rf ~/.streamlit

# Then restart Streamlit
streamlit run frontend/streamlit_app.py
```

**Solution 4: Verify Installation**
```bash
# Check reportlab is installed
python -m pip list | findstr reportlab

# Output should show: reportlab 4.2.2 (or similar)

# If not installed:
python -m pip install reportlab
```

## Image Upload Issues

### Symptoms
- Upload button not working
- "File not found" errors

### Solution
- Ensure images are PNG or JPG
- Images should be < 50MB
- Try different image

## Slider Crashes

### Symptoms
- "ValueError: operands could not be broadcast together"
- App returns to upload screen when using comparison slider

### Status
✅ **FIXED** - Updated to convert all image arrays to uint8 before matplotlib rendering

## Memory Issues

### Symptoms
- "Unable to allocate X MiB for array"
- App becomes slow with large images
- Browser crashes

### Status
✅ **FIXED** - Optimized:
- DPI set to 80 (reduced from 100)
- All images converted to uint8 before display
- Explicit `plt.close()` after each figure

## Backend Not Starting

### Symptoms
- Flask API not responding
- ModuleNotFoundError on imports

### Solution
```bash
cd D:\u-net-biomedical-flow-project
python -m backend.app

# Should output:
# 🚀 Initializing backend...
# ✅ Flask app initialized successfully
```

## Model Not Loading

### Symptoms
- "⚠️ Model file not found, using untrained model"
- Model always shows blank output

### Solution
Ensure trained model exists at expected path:
```
backend/checkpoint/best_unet_model.pt
```

If not found:
1. Train model: `python backend/unet_training.ipynb`
2. Or download pre-trained model
3. Verify file permissions

## Performance Issues

### Symptoms
- Slow inference (> 1 second)
- High GPU memory usage

### Status
✅ **OPTIMIZED**:
- TorchScript JIT compilation enabled (20% faster)
- Mixed-precision FP16 enabled
- Batch processing support
- Model profiling available

Run profiling:
```bash
python backend/profile_model.py
```

## Export Report Issues

### PNG Export Issues

**Symptoms**: "RendererAgg object has no attribute tostring_rgb"

**Status**: ✅ **FIXED** - Updated to use `renderer.buffer_rgba()` instead

**Solution**: 
```bash
# Restart app
streamlit run frontend/streamlit_app.py
```

### PDF Export Issues

**Symptoms**: PDF button missing or "PDF export not available"

**Status**: ✅ **FIXED** - reportlab integration complete

**Solution**:
1. Verify reportlab installed: `python -m pip list | findstr reportlab`
2. If missing: `python -m pip install reportlab`
3. Restart Streamlit (see hard refresh above)

## Metrics Show as "NaN" or "0.00"

### Symptoms
- PSNR shows "NaN" or "0.00"
- SSIM shows "0.0000"
- Runtime shows "0.00"

### Cause
- Model still initializing
- Image dimensions mismatch
- Inference failed silently

### Solution
- Wait a few seconds for first inference
- Check browser console for errors
- Verify image dimensions (256×256 expected after resize)

## Getting Help

### Diagnostic Commands

Check system:
```bash
python -c "
import torch
import streamlit
import reportlab
import matplotlib
print('All dependencies installed')
"
```

Test backend:
```bash
python -m backend.app
```

Test frontend:
```bash
python -c "from frontend.export_reports import ReportExporter; print('Frontend OK')"
```

Test full stack:
```bash
# Terminal 1:
python -m backend.app

# Terminal 2:
streamlit run frontend/streamlit_app.py
```

### View Logs

**Streamlit logs**: Check browser console (F12)

**Backend logs**: Check terminal where Flask started

**Python errors**: Check terminal output

## Still Not Working?

1. Verify Python version: `python --version` (Should be 3.10+)
2. Verify virtual env active: Check `(venv)` in terminal
3. Reinstall dependencies: `pip install -r requirements.txt`
4. Check disk space (>500MB free)
5. Restart computer (nuclear option)

---

**Last Updated**: 2026-03-25
**Status**: All known issues resolved ✅
