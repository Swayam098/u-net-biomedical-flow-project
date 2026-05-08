#!/usr/bin/env python3
"""
Profiling script to benchmark model optimizations
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import numpy as np
import torch
from inference import UNet
from model_loader import load_unet_model
from optimization import OptimizedUNetInference

def main():
    print("\n" + "="*70)
    print("🔬 U-NET MODEL OPTIMIZATION PROFILING")
    print("="*70)
    
    # Determine device
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"\n📱 Device: {device.upper()}")
    
    # Load model
    print("\n📦 Loading model...")
    model = load_unet_model("backend/models/unet_model.pth")
    
    # Create sample input (256x256 grayscale ultrasound)
    sample_input = np.random.rand(256, 256).astype(np.float32)
    
    # Initialize optimizer
    optimizer = OptimizedUNetInference(model, device=device)
    
    # Run profiling
    print("\n" + "="*70)
    comparison = optimizer.compare_optimizations(sample_input)
    print("="*70)
    
    # Detailed results
    baseline = comparison["baseline"]
    torchscript = comparison["torchscript"]
    speedup = comparison["speedup_percent"]
    
    print("\n📊 DETAILED RESULTS:\n")
    print(f"Baseline (Standard PyTorch):")
    print(f"  • Inference Time: {baseline['avg_time_ms']:.2f}ms ± {baseline['std_time_ms']:.2f}ms")
    print(f"  • Memory Used: {baseline['memory_mb']:.2f} MB")
    
    if torchscript:
        print(f"\nTorchScript JIT Optimized:")
        print(f"  • Inference Time: {torchscript['avg_time_ms']:.2f}ms ± {torchscript['std_time_ms']:.2f}ms")
        print(f"  • Memory Used: {torchscript['memory_mb']:.2f} MB")
        print(f"  • Speedup: {speedup:.1f}% faster")
    
    print(f"\n⚡ Additional Optimizations Available:")
    print(f"  • Mixed-Precision (FP16): Available on {device.upper()}")
    print(f"  • Batch Processing: Implemented")
    print(f"  • Enhanced Loss Functions: Ready for retraining")
    
    print("\n" + "="*70)
    print("✅ PROFILING COMPLETE")
    print("="*70 + "\n")

if __name__ == "__main__":
    main()
