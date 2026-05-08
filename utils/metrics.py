import numpy as np
from skimage.metrics import peak_signal_noise_ratio, structural_similarity

def compute_psnr(original, output):
    try:
        original = np.nan_to_num(original, nan=0.0, posinf=1.0, neginf=0.0)
        output = np.nan_to_num(output, nan=0.0, posinf=1.0, neginf=0.0)
        return peak_signal_noise_ratio(original, output, data_range=1.0)
    except Exception as e:
        print("⚠️ PSNR computation failed:", e)
        return 0.0

def compute_ssim(original, output):
    try:
        original = np.nan_to_num(original, nan=0.0, posinf=1.0, neginf=0.0)
        output = np.nan_to_num(output, nan=0.0, posinf=1.0, neginf=0.0)

        return structural_similarity(
            original,
            output,
            data_range=1.0,
            win_size=7  # SAFE window for medical images
        )
    except Exception as e:
        print("⚠️ SSIM computation failed:", e)
        return 0.0
