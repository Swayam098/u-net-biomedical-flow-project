"""
Real-time preview effects module
- Intensity adjustment
- Blur effect
- Contrast enhancement
- Saturation adjustment (for future color images)
"""

import numpy as np
import cv2
from typing import Tuple


class PreviewEffects:
    """Apply real-time effects to enhanced image"""
    
    @staticmethod
    def adjust_intensity(image: np.ndarray, factor: float = 1.0) -> np.ndarray:
        """Adjust brightness/intensity
        
        Args:
            image: Input image (0-1 range)
            factor: Intensity multiplier (0.8-1.2 typical)
        
        Returns:
            Adjusted image (clipped to 0-1)
        """
        result = np.clip(image * factor, 0, 1)
        return result
    
    @staticmethod
    def apply_blur(image: np.ndarray, sigma: float = 0.0) -> np.ndarray:
        """Apply Gaussian blur
        
        Args:
            image: Input image (0-1 range, grayscale)
            sigma: Gaussian blur sigma (0-5 typical)
        
        Returns:
            Blurred image
        """
        if sigma < 0.1:
            return image
        
        # Convert to 0-255 for OpenCV
        img_uint8 = (np.clip(image, 0, 1) * 255).astype(np.uint8)
        
        # Kernel size must be odd
        kernel_size = int(sigma * 4) * 2 + 1
        
        # Apply Gaussian blur
        blurred_uint8 = cv2.GaussianBlur(img_uint8, (kernel_size, kernel_size), sigma)
        
        # Convert back to 0-1
        result = blurred_uint8.astype(np.float32) / 255.0
        return result
    
    @staticmethod
    def adjust_contrast(image: np.ndarray, factor: float = 1.0) -> np.ndarray:
        """Adjust contrast
        
        Args:
            image: Input image (0-1 range)
            factor: Contrast multiplier (0.8-1.5 typical)
        
        Returns:
            Image with adjusted contrast
        """
        # Contrast adjustment: (x - 0.5) * factor + 0.5
        center = 0.5
        result = (image - center) * factor + center
        return np.clip(result, 0, 1)
    
    @staticmethod
    def adjust_saturation(image: np.ndarray, factor: float = 0.0) -> np.ndarray:
        """Adjust saturation (placeholder for color images)
        
        For grayscale, this is a no-op since there's no color information
        
        Args:
            image: Input image (grayscale)
            factor: Saturation factor (ignored for grayscale)
        
        Returns:
            Original image
        """
        # For grayscale ultrasound, saturation has no effect
        return image
    
    @staticmethod
    def apply_all_effects(
        image: np.ndarray,
        intensity: float = 1.0,
        blur: float = 0.0,
        contrast: float = 1.0,
        saturation: float = 0.0
    ) -> np.ndarray:
        """Apply all effects in sequence
        
        Args:
            image: Input image (0-1 range)
            intensity: Brightness multiplier (0.8-1.2)
            blur: Gaussian blur sigma (0-5)
            contrast: Contrast multiplier (0.8-1.5)
            saturation: Saturation (ignored for grayscale)
        
        Returns:
            Image with all effects applied
        """
        result = image.copy()
        
        # Apply in order: intensity → blur → contrast
        result = PreviewEffects.adjust_intensity(result, intensity)
        result = PreviewEffects.apply_blur(result, blur)
        result = PreviewEffects.adjust_contrast(result, contrast)
        result = PreviewEffects.adjust_saturation(result, saturation)
        
        return result
    
    @staticmethod
    def create_preview_config(
        intensity: float = 1.0,
        blur: float = 0.0,
        contrast: float = 1.0,
        saturation: float = 0.0
    ) -> dict:
        """Create preview configuration dict"""
        return {
            "intensity": intensity,
            "blur": blur,
            "contrast": contrast,
            "saturation": saturation
        }


class GaussianBlurBaseline:
    """Gaussian blur as baseline comparison"""
    
    @staticmethod
    def apply(image: np.ndarray, sigma: float = 2.0) -> np.ndarray:
        """Apply simple Gaussian blur
        
        Args:
            image: Input image (0-1 range)
            sigma: Blur sigma
        
        Returns:
            Blurred image
        """
        img_uint8 = (np.clip(image, 0, 1) * 255).astype(np.uint8)
        kernel_size = max(3, int(sigma * 4) * 2 + 1)
        blurred_uint8 = cv2.GaussianBlur(img_uint8, (kernel_size, kernel_size), sigma)
        return blurred_uint8.astype(np.float32) / 255.0


def test_preview_effects():
    """Test preview effects"""
    print("Testing preview effects...")
    
    # Create dummy image
    image = np.random.rand(256, 256)
    
    effects = PreviewEffects()
    
    # Test each effect
    tests = [
        ("Intensity", lambda: effects.adjust_intensity(image, 1.1)),
        ("Blur", lambda: effects.apply_blur(image, 2.0)),
        ("Contrast", lambda: effects.adjust_contrast(image, 1.2)),
        ("All Effects", lambda: effects.apply_all_effects(image, 1.1, 1.5, 1.2, 0.0)),
    ]
    
    for name, test_fn in tests:
        try:
            result = test_fn()
            assert result.shape == image.shape
            assert result.min() >= 0 and result.max() <= 1
            print(f"✓ {name}: OK")
        except Exception as e:
            print(f"✗ {name}: {e}")


if __name__ == "__main__":
    test_preview_effects()
