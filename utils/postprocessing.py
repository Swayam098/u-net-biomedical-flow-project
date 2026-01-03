import numpy as np

def postprocess_image(image_np):
    return (image_np * 255).astype("uint8")
