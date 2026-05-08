import numpy as np
import cv2

def preprocess_image(image, size=256):
    """
    Resize image to a fixed power-of-2 size
    to avoid U-Net tensor mismatch.
    """
    img = np.array(image).astype("float32")

    # Resize to 256x256 (critical for U-Net)
    img = cv2.resize(img, (size, size), interpolation=cv2.INTER_AREA)

    # Normalize
    img = img / 255.0

    return img
