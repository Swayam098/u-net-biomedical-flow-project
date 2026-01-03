import numpy as np

def svd_filter_image(image_np, k=20):
    try:
        img = np.nan_to_num(image_np, nan=0.0, posinf=1.0, neginf=0.0)
        U, S, Vt = np.linalg.svd(img, full_matrices=False)
        S[k:] = 0
        filtered = U @ np.diag(S) @ Vt
        return np.clip(filtered, 0, 1)
    except Exception as e:
        print("⚠️ SVD failed:", e)
        return image_np
