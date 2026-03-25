"""
Advanced loss functions for U-Net training
- Perceptual Loss (VGG features)
- Edge Loss (Sobel gradients)
- Combined loss strategies
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models
from typing import Optional, Callable


class PerceptualLoss(nn.Module):
    """Perceptual loss using pre-trained VGG19 features"""
    
    def __init__(self, device: str = "cpu", layer: str = "relu5_1"):
        super().__init__()
        self.device = device
        self.layer = layer
        
        # Load pre-trained VGG19 (frozen)
        vgg = models.vgg19(pretrained=True).to(device)
        self.vgg_features = vgg.features
        
        # Freeze parameters
        for param in self.vgg_features.parameters():
            param.requires_grad = False
        
        self.vgg_features.eval()
        
        # Layer mapping
        self.layer_mapping = {
            "relu1_1": 1,
            "relu2_1": 6,
            "relu3_1": 11,
            "relu4_1": 20,
            "relu5_1": 29
        }
        
        self.target_layer = self.layer_mapping.get(layer, 29)
        
        # Normalization (ImageNet stats)
        self.register_buffer(
            'mean',
            torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
        )
        self.register_buffer(
            'std',
            torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)
        )
    
    def _normalize(self, x: torch.Tensor) -> torch.Tensor:
        """Normalize input to ImageNet statistics"""
        # Convert grayscale to RGB by replicating channels
        if x.shape[1] == 1:
            x = x.repeat(1, 3, 1, 1)
        return (x - self.mean) / self.std
    
    def _extract_features(self, x: torch.Tensor) -> torch.Tensor:
        """Extract VGG features up to target layer"""
        x = self._normalize(x)
        for i, layer in enumerate(self.vgg_features):
            x = layer(x)
            if i == self.target_layer:
                break
        return x
    
    def forward(self, predicted: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """Compute perceptual loss"""
        predicted_features = self._extract_features(predicted)
        target_features = self._extract_features(target)
        
        # Detach target features (don't backprop through target)
        target_features = target_features.detach()
        
        # MSE loss on features
        return F.mse_loss(predicted_features, target_features)


class EdgeLoss(nn.Module):
    """Edge-aware loss using Sobel operators"""
    
    def __init__(self):
        super().__init__()
        
        # Sobel kernels
        sobel_x = torch.tensor([
            [-1, 0, 1],
            [-2, 0, 2],
            [-1, 0, 1]
        ], dtype=torch.float32).unsqueeze(0).unsqueeze(0)
        
        sobel_y = torch.tensor([
            [-1, -2, -1],
            [0, 0, 0],
            [1, 2, 1]
        ], dtype=torch.float32).unsqueeze(0).unsqueeze(0)
        
        self.register_buffer('sobel_x', sobel_x)
        self.register_buffer('sobel_y', sobel_y)
    
    def _compute_edges(self, x: torch.Tensor) -> torch.Tensor:
        """Compute edge maps using Sobel"""
        edges_x = F.conv2d(x, self.sobel_x, padding=1)
        edges_y = F.conv2d(x, self.sobel_y, padding=1)
        edges = torch.sqrt(edges_x**2 + edges_y**2 + 1e-8)
        return edges
    
    def forward(self, predicted: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """Compute edge-aware loss"""
        pred_edges = self._compute_edges(predicted)
        target_edges = self._compute_edges(target)
        
        return F.l1_loss(pred_edges, target_edges)


class CombinedLoss(nn.Module):
    """Combined loss: MSE + Perceptual + Edge"""
    
    def __init__(
        self,
        mse_weight: float = 1.0,
        perceptual_weight: float = 0.1,
        edge_weight: float = 0.1,
        device: str = "cpu"
    ):
        super().__init__()
        self.mse_weight = mse_weight
        self.perceptual_weight = perceptual_weight
        self.edge_weight = edge_weight
        
        self.mse_loss = nn.MSELoss()
        self.perceptual_loss = PerceptualLoss(device=device, layer="relu5_1")
        self.edge_loss = EdgeLoss()
    
    def forward(
        self,
        predicted: torch.Tensor,
        target: torch.Tensor,
        use_perceptual: bool = True,
        use_edge: bool = True
    ) -> torch.Tensor:
        """Compute combined loss"""
        mse = self.mse_loss(predicted, target)
        
        total_loss = self.mse_weight * mse
        
        if use_perceptual:
            try:
                perceptual = self.perceptual_loss(predicted, target)
                total_loss = total_loss + self.perceptual_weight * perceptual
            except Exception as e:
                print(f"⚠️ Perceptual loss failed: {e}")
        
        if use_edge:
            try:
                edge = self.edge_loss(predicted, target)
                total_loss = total_loss + self.edge_weight * edge
            except Exception as e:
                print(f"⚠️ Edge loss failed: {e}")
        
        return total_loss


class SSIM_Loss(nn.Module):
    """Structural Similarity Index Loss (improves SSIM metric)"""
    
    def __init__(self, window_size: int = 11):
        super().__init__()
        self.window_size = window_size
    
    def _gaussian_kernel(self, window_size: int, sigma: float = 1.5) -> torch.Tensor:
        """Create 1D Gaussian kernel"""
        x = torch.arange(window_size).float() - (window_size - 1) / 2
        gauss = torch.exp(-x.pow(2.0) / (2 * sigma ** 2))
        return gauss / gauss.sum()
    
    def _create_window(self, window_size: int, channels: int) -> torch.Tensor:
        """Create 2D Gaussian window"""
        kernel_1d = self._gaussian_kernel(window_size)
        kernel_2d = kernel_1d.unsqueeze(-1) @ kernel_1d.unsqueeze(0)
        window = kernel_2d.unsqueeze(0).unsqueeze(0)
        return window.repeat(channels, 1, 1, 1)
    
    def forward(self, predicted: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """Compute SSIM loss (1 - SSIM)"""
        channels = predicted.shape[1]
        window = self._create_window(self.window_size, channels).to(predicted.device)
        
        mu1 = F.conv2d(predicted, window, padding=self.window_size // 2, groups=channels)
        mu2 = F.conv2d(target, window, padding=self.window_size // 2, groups=channels)
        
        mu1_sq = mu1 ** 2
        mu2_sq = mu2 ** 2
        mu1_mu2 = mu1 * mu2
        
        sigma1_sq = F.conv2d(predicted ** 2, window, padding=self.window_size // 2, groups=channels) - mu1_sq
        sigma2_sq = F.conv2d(target ** 2, window, padding=self.window_size // 2, groups=channels) - mu2_sq
        sigma12 = F.conv2d(predicted * target, window, padding=self.window_size // 2, groups=channels) - mu1_mu2
        
        c1, c2 = 0.01 ** 2, 0.03 ** 2
        
        ssim = (2 * mu1_mu2 + c1) * (2 * sigma12 + c2) / (
            (mu1_sq + mu2_sq + c1) * (sigma1_sq + sigma2_sq + c2)
        )
        
        return 1 - ssim.mean()


class LossFactory:
    """Factory for creating loss functions"""
    
    @staticmethod
    def create(
        loss_type: str = "mse",
        device: str = "cpu",
        **kwargs
    ) -> nn.Module:
        """Create loss function by name"""
        
        loss_types = {
            "mse": nn.MSELoss,
            "l1": nn.L1Loss,
            "perceptual": lambda: PerceptualLoss(device=device),
            "edge": EdgeLoss,
            "ssim": SSIM_Loss,
            "combined": lambda: CombinedLoss(device=device, **kwargs),
        }
        
        if loss_type not in loss_types:
            raise ValueError(f"Unknown loss type: {loss_type}")
        
        loss_fn = loss_types[loss_type]
        return loss_fn() if callable(loss_fn) else loss_fn(**kwargs)
    
    @staticmethod
    def get_available():
        """Get list of available loss functions"""
        return ["mse", "l1", "perceptual", "edge", "ssim", "combined"]


if __name__ == "__main__":
    # Test loss functions
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Testing loss functions on {device.upper()}")
    
    # Create dummy tensors
    pred = torch.randn(2, 1, 256, 256, device=device)
    target = torch.randn(2, 1, 256, 256, device=device)
    
    # Test each loss
    for loss_name in LossFactory.get_available():
        try:
            loss_fn = LossFactory.create(loss_name, device=device)
            loss_value = loss_fn(pred, target)
            print(f"✓ {loss_name:12} - Loss: {loss_value.item():.4f}")
        except Exception as e:
            print(f"✗ {loss_name:12} - Error: {str(e)[:50]}")
