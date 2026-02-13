"""
Benchmark: Triton Kernel vs PyTorch Baseline

Compares the performance of custom Triton gating kernel against
PyTorch's native implementation.
"""

import torch
import time
from volt.kernels.gating import topk_gating_kernel, benchmark_gating_kernel


def main():
    print("=" * 70)
    print("Volt-Infer Kernel Benchmark: Top-K Gating")
    print("=" * 70)
    
    # Test configurations
    configs = [
        {"batch_size": 32, "hidden_dim": 4096, "num_experts": 8, "k": 2},
        {"batch_size": 64, "hidden_dim": 4096, "num_experts": 8, "k": 2},
        {"batch_size": 128, "hidden_dim": 4096, "num_experts": 16, "k": 2},
    ]
    
    print("\nRunning benchmarks...\n")
    
    for config in configs:
        print(f"Configuration: {config}")
        
        results = benchmark_gating_kernel(
            batch_size=config["batch_size"],
            hidden_dim=config["hidden_dim"],
            num_experts=config["num_experts"],
            k=config["k"],
            num_iterations=100,
        )
        
        print(f"  Triton Kernel:  {results['triton_ms']:.3f} ms")
        print(f"  PyTorch Native: {results['pytorch_ms']:.3f} ms")
        print(f"  Speedup:        {results['speedup']:.2f}x")
        print()
    
    print("=" * 70)
    print("✓ Benchmark completed")


if __name__ == "__main__":
    if not torch.cuda.is_available():
        print("✗ CUDA not available. Please run on a GPU-enabled system.")
        exit(1)
    
    main()
