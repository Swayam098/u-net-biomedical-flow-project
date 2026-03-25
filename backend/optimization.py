"""
Model optimization module for U-Net inference
- TorchScript JIT compilation
- Mixed-precision inference (FP16)
- Memory and speed profiling
"""

import torch
import torch.nn as nn
import numpy as np
import time
import psutil
import os
from typing import Tuple, Dict

class OptimizedUNetInference:
    """Optimized inference pipeline with profiling"""
    
    def __init__(self, model: nn.Module, device: str = "cpu"):
        self.device = torch.device(device)
        self.model = model.to(self.device)
        self.model.eval()
        self.jit_model = None
        self.use_fp16 = device == "cuda"
        
    def profile_model(self, sample_input: np.ndarray) -> Dict:
        """Profile baseline model performance"""
        print("\n" + "="*60)
        print("📊 PROFILING BASELINE MODEL")
        print("="*60)
        
        # Memory baseline
        process = psutil.Process(os.getpid())
        mem_before = process.memory_info().rss / 1024 / 1024  # MB
        
        # Inference time (warm up)
        x = torch.tensor(sample_input).unsqueeze(0).unsqueeze(0).float().to(self.device)
        
        with torch.no_grad():
            for _ in range(3):  # Warmup
                _ = self.model(x)
        
        # Actual timing (10 runs)
        times = []
        for _ in range(10):
            start = time.time()
            with torch.no_grad():
                _ = self.model(x)
            times.append(time.time() - start)
        
        mem_after = process.memory_info().rss / 1024 / 1024  # MB
        
        avg_time = np.mean(times)
        std_time = np.std(times)
        mem_used = mem_after - mem_before
        
        results = {
            "avg_time_ms": avg_time * 1000,
            "std_time_ms": std_time * 1000,
            "memory_mb": mem_used,
            "input_shape": sample_input.shape
        }
        
        print(f"✓ Avg Inference Time: {avg_time*1000:.2f}ms ± {std_time*1000:.2f}ms")
        print(f"✓ Memory Used: {mem_used:.2f} MB")
        print(f"✓ Input Shape: {sample_input.shape}")
        
        return results
    
    def compile_torchscript(self, sample_input: np.ndarray, save_path: str = None):
        """Compile model to TorchScript JIT"""
        print("\n" + "="*60)
        print("🔧 CONVERTING TO TORCHSCRIPT JIT")
        print("="*60)
        
        try:
            x = torch.tensor(sample_input).unsqueeze(0).unsqueeze(0).float().to(self.device)
            
            # JIT trace compilation
            self.jit_model = torch.jit.trace(self.model, x)
            
            # Save if path provided
            if save_path:
                self.jit_model.save(save_path)
                print(f"✓ TorchScript model saved to {save_path}")
            
            print("✓ TorchScript JIT compilation successful")
            return True
            
        except Exception as e:
            print(f"❌ TorchScript compilation failed: {e}")
            return False
    
    def profile_torchscript(self, sample_input: np.ndarray) -> Dict:
        """Profile optimized TorchScript model"""
        if self.jit_model is None:
            print("❌ TorchScript model not compiled yet")
            return {}
        
        print("\n" + "="*60)
        print("📊 PROFILING TORCHSCRIPT MODEL")
        print("="*60)
        
        process = psutil.Process(os.getpid())
        mem_before = process.memory_info().rss / 1024 / 1024
        
        x = torch.tensor(sample_input).unsqueeze(0).unsqueeze(0).float().to(self.device)
        
        # Warmup
        with torch.no_grad():
            for _ in range(3):
                _ = self.jit_model(x)
        
        # Actual timing
        times = []
        for _ in range(10):
            start = time.time()
            with torch.no_grad():
                _ = self.jit_model(x)
            times.append(time.time() - start)
        
        mem_after = process.memory_info().rss / 1024 / 1024
        
        avg_time = np.mean(times)
        std_time = np.std(times)
        mem_used = mem_after - mem_before
        
        results = {
            "avg_time_ms": avg_time * 1000,
            "std_time_ms": std_time * 1000,
            "memory_mb": mem_used
        }
        
        print(f"✓ Avg Inference Time: {avg_time*1000:.2f}ms ± {std_time*1000:.2f}ms")
        print(f"✓ Memory Used: {mem_used:.2f} MB")
        
        return results
    
    def inference_fp32(self, image_np: np.ndarray) -> np.ndarray:
        """Standard FP32 inference"""
        with torch.no_grad():
            x = torch.tensor(image_np).unsqueeze(0).unsqueeze(0).float().to(self.device)
            if self.jit_model is not None:
                output = self.jit_model(x)
            else:
                output = self.model(x)
            return output.squeeze().cpu().numpy()
    
    def inference_fp16(self, image_np: np.ndarray) -> np.ndarray:
        """Mixed-precision FP16 inference (GPU only)"""
        if self.device.type != 'cuda':
            print("⚠️ FP16 inference requires CUDA, falling back to FP32")
            return self.inference_fp32(image_np)
        
        with torch.no_grad():
            x = torch.tensor(image_np).unsqueeze(0).unsqueeze(0).float().to(self.device)
            
            # Use autocast for automatic mixed precision
            with torch.cuda.amp.autocast():
                if self.jit_model is not None:
                    output = self.jit_model(x)
                else:
                    output = self.model(x)
            
            return output.squeeze().cpu().numpy()
    
    def compare_optimizations(self, sample_input: np.ndarray) -> Dict:
        """Compare all optimization strategies"""
        print("\n" + "="*60)
        print("📈 OPTIMIZATION COMPARISON")
        print("="*60)
        
        baseline = self.profile_model(sample_input)
        
        # Compile TorchScript
        self.compile_torchscript(sample_input)
        jit_profile = self.profile_torchscript(sample_input)
        
        # Calculate improvements
        jit_speedup = baseline["avg_time_ms"] / jit_profile["avg_time_ms"] if jit_profile else 1
        
        summary = {
            "baseline": baseline,
            "torchscript": jit_profile,
            "speedup_percent": (jit_speedup - 1) * 100
        }
        
        print("\n" + "-"*60)
        print("SUMMARY:")
        print(f"✓ TorchScript Speedup: {jit_speedup:.2f}x ({(jit_speedup-1)*100:.1f}% faster)")
        print(f"✓ FP16 Available: {self.device.type == 'cuda'}")
        print("-"*60 + "\n")
        
        return summary


def batch_inference(model, images: list, batch_size: int = 4, device: str = "cpu"):
    """Process multiple images in batches"""
    results = []
    
    for i in range(0, len(images), batch_size):
        batch = images[i:i+batch_size]
        batch_tensor = torch.stack([
            torch.tensor(img).unsqueeze(0).float() 
            for img in batch
        ]).to(device)
        
        with torch.no_grad():
            batch_output = model(batch_tensor)
        
        for output in batch_output:
            results.append(output.squeeze().cpu().numpy())
    
    return results
