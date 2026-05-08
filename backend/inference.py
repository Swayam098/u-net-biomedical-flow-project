import math
from typing import Dict, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


try:
    import pywt  # type: ignore
except Exception:  # pragma: no cover - optional dependency
    pywt = None


class DoubleConv(nn.Module):
    def __init__(self, in_ch: int, out_ch: int, dropout_p: float = 0.1) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Dropout2d(dropout_p),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class UNet(nn.Module):
    """Compact U-Net aligned with project architecture (1->32->64->128->256)."""

    checkpoint_loaded: bool
    checkpoint_path: str

    def __init__(self, dropout_p: float = 0.1) -> None:
        super().__init__()
        self.enc1 = DoubleConv(1, 32, dropout_p)
        self.enc2 = DoubleConv(32, 64, dropout_p)
        self.enc3 = DoubleConv(64, 128, dropout_p)
        self.pool = nn.MaxPool2d(2)

        self.bottleneck = DoubleConv(128, 256, dropout_p)

        self.up3 = nn.ConvTranspose2d(256, 128, kernel_size=2, stride=2)
        self.dec3 = DoubleConv(256, 128, dropout_p)

        self.up2 = nn.ConvTranspose2d(128, 64, kernel_size=2, stride=2)
        self.dec2 = DoubleConv(128, 64, dropout_p)

        self.up1 = nn.ConvTranspose2d(64, 32, kernel_size=2, stride=2)
        self.dec1 = DoubleConv(64, 32, dropout_p)

        self.final = nn.Conv2d(32, 1, kernel_size=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        e1 = self.enc1(x)
        e2 = self.enc2(self.pool(e1))
        e3 = self.enc3(self.pool(e2))

        b = self.bottleneck(self.pool(e3))

        d3 = self.up3(b)
        d3 = self.dec3(torch.cat([d3, e3], dim=1))

        d2 = self.up2(d3)
        d2 = self.dec2(torch.cat([d2, e2], dim=1))

        d1 = self.up1(d2)
        d1 = self.dec1(torch.cat([d1, e1], dim=1))

        return torch.sigmoid(self.final(d1))


def preprocess_image(image_np: np.ndarray, size: int = 256) -> np.ndarray:
    """Convert to grayscale float image in [0,1] and resize to square input."""
    if image_np.ndim == 3:
        image_np = np.mean(image_np, axis=2)

    img = image_np.astype(np.float32)
    if img.max() > 1.0:
        img /= 255.0

    # Light-weight bilinear resize via torch to avoid hard OpenCV dependency.
    tensor = torch.from_numpy(img).unsqueeze(0).unsqueeze(0)
    tensor = F.interpolate(tensor, size=(size, size), mode="bilinear", align_corners=False)
    return tensor.squeeze(0).squeeze(0).numpy()


def wavelet_preprocess(image_np: np.ndarray, wavelet: str = "haar", threshold: float = 0.035) -> np.ndarray:
    """Frequency-aware denoising prior: soft-threshold detail bands before model inference."""
    if pywt is None:
        return image_np

    coeffs2 = pywt.dwt2(image_np, wavelet)
    ll, (lh, hl, hh) = coeffs2

    def soft_thr(x: np.ndarray, thr: float) -> np.ndarray:
        return np.sign(x) * np.maximum(np.abs(x) - thr, 0.0)

    lh = soft_thr(lh, threshold)
    hl = soft_thr(hl, threshold)
    hh = soft_thr(hh, threshold)

    reconstructed = pywt.idwt2((ll, (lh, hl, hh)), wavelet)
    reconstructed = reconstructed[: image_np.shape[0], : image_np.shape[1]]
    return np.clip(reconstructed, 0.0, 1.0)


def _forward_once(
    model: nn.Module,
    image_np: np.ndarray,
    device: torch.device,
    use_fp16: bool = False,
) -> np.ndarray:
    x = torch.from_numpy(image_np).float().unsqueeze(0).unsqueeze(0).to(device)
    with torch.no_grad():
        if use_fp16 and device.type == "cuda":
            with torch.autocast(device_type="cuda", dtype=torch.float16):
                y = model(x)
        else:
            y = model(x)
    out = y.squeeze(0).squeeze(0).detach().cpu().numpy()
    return np.clip(out, 0.0, 1.0)


def run_unet_inference(
    model: nn.Module,
    image_np: np.ndarray,
    device: torch.device,
    use_wavelet: bool = False,
    wavelet_name: str = "haar",
    wavelet_threshold: float = 0.035,
    use_fp16: bool = False,
    mc_samples: int = 1,
    enable_dropout: bool = False,
) -> Dict[str, np.ndarray]:
    """Run deterministic or MC-dropout inference and return output + uncertainty map."""
    processed = preprocess_image(image_np)
    if use_wavelet:
        processed = wavelet_preprocess(processed, wavelet=wavelet_name, threshold=wavelet_threshold)

    model = model.to(device)
    if enable_dropout and mc_samples > 1:
        model.train()
    else:
        model.eval()

    samples = []
    for _ in range(max(1, mc_samples)):
        out = _forward_once(model, processed, device=device, use_fp16=use_fp16)
        samples.append(out)

    stacked = np.stack(samples, axis=0)
    mean_pred = stacked.mean(axis=0)
    uncertainty = stacked.std(axis=0) if stacked.shape[0] > 1 else np.zeros_like(mean_pred)

    return {
        "input": processed,
        "prediction": mean_pred,
        "uncertainty": uncertainty,
    }


def parameter_count(model: nn.Module) -> Tuple[int, int]:
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return total, trainable


def estimate_memory_mb(image_size: int = 256, batch_size: int = 1) -> float:
    pixels = image_size * image_size * batch_size
    # rough fp32 tensors through model; useful for deployment stats only.
    return (pixels * 4 * 16) / (1024 * 1024)
