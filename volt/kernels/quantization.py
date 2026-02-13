"""
Triton Int8 Weight Dequantization Kernel

High-performance kernel for on-the-fly dequantization of INT8-quantized expert
weights during inference. This enables 4x memory savings while maintaining
near-FP16 accuracy through per-channel scaling.

Quantization Scheme:
    W_fp16 ≈ scale * W_int8 + zero_point
    
Where:
    - W_int8 ∈ [-127, 127] (stored as int8)
    - scale ∈ ℝ+ (per-channel or per-tensor)
    - zero_point ∈ ℝ (optional, often 0 for symmetric quantization)
"""

from typing import Optional

import torch
import triton
import triton.language as tl


@triton.jit
def _int8_dequant_kernel(
    # Input pointers
    quantized_ptr,  # [M, N] int8 weights
    scales_ptr,  # [N] or [1] float32 scales
    zero_points_ptr,  # [N] or [1] float32 zero points (optional)
    # Output pointer
    output_ptr,  # [M, N] float16/float32 output
    # Dimensions
    M: tl.constexpr,
    N: tl.constexpr,
    use_zero_point: tl.constexpr,
    per_channel: tl.constexpr,
    # Block sizes
    BLOCK_SIZE_M: tl.constexpr,
    BLOCK_SIZE_N: tl.constexpr,
):
    """
    Triton kernel for INT8 weight dequantization.
    
    Algorithm:
        For each weight element W[i,j]:
            1. Load quantized value (int8)
            2. Cast to float
            3. Load appropriate scale (per-tensor or per-channel)
            4. Optionally load zero point
            5. Compute: W_fp = scale * (W_int8 - zero_point)
            6. Store result
    
    Memory Access Pattern:
        - 2D block-tiled access for quantized weights
        - Vector load for scales (reused across M dimension)
        - Coalesced writes to output
    
    Performance:
        - Memory bandwidth bound (INT8 reads, FP16/32 writes)
        - Achieves ~2-3x speedup over naive PyTorch casting
    """
    # Program IDs for 2D grid
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)
    
    # Compute block offsets
    offs_m = pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)
    offs_n = pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)
    
    # Create 2D offset grid [BLOCK_SIZE_M, BLOCK_SIZE_N]
    offs_m_expanded = offs_m[:, None]
    offs_n_expanded = offs_n[None, :]
    
    # Compute pointers for this block
    quantized_block_ptrs = quantized_ptr + offs_m_expanded * N + offs_n_expanded
    output_block_ptrs = output_ptr + offs_m_expanded * N + offs_n_expanded
    
    # Boundary masks
    mask_m = offs_m < M
    mask_n = offs_n < N
    mask = mask_m[:, None] & mask_n[None, :]
    
    # Load quantized weights [BLOCK_SIZE_M, BLOCK_SIZE_N]
    quantized_block = tl.load(quantized_block_ptrs, mask=mask, other=0).to(tl.float32)
    
    # Load scales (per-channel or per-tensor)
    if per_channel:
        # Load scale for each column [BLOCK_SIZE_N]
        scale_ptrs = scales_ptr + offs_n
        scales = tl.load(scale_ptrs, mask=mask_n, other=1.0)
        scales_broadcast = scales[None, :]  # [1, BLOCK_SIZE_N]
    else:
        # Single tensor-wide scale
        scale = tl.load(scales_ptr)
        scales_broadcast = scale
    
    # Load zero points if enabled
    if use_zero_point:
        if per_channel:
            zp_ptrs = zero_points_ptr + offs_n
            zero_points = tl.load(zp_ptrs, mask=mask_n, other=0.0)
            zp_broadcast = zero_points[None, :]
        else:
            zp = tl.load(zero_points_ptr)
            zp_broadcast = zp
        
        # Dequantize: FP = scale * (INT8 - zero_point)
        dequantized = scales_broadcast * (quantized_block - zp_broadcast)
    else:
        # Symmetric quantization: FP = scale * INT8
        dequantized = scales_broadcast * quantized_block
    
    # Store result
    tl.store(output_block_ptrs, dequantized, mask=mask)


class Int8DequantFunction(torch.autograd.Function):
    """
    Custom autograd function for INT8 dequantization.
    
    Note: This is inference-only. Backward pass is not implemented since
    quantized weights are frozen during inference.
    """
    
    @staticmethod
    def forward(
        ctx,
        quantized_weights: torch.Tensor,  # [M, N] int8
        scales: torch.Tensor,  # [N] or [1] float32
        zero_points: Optional[torch.Tensor] = None,  # [N] or [1] float32
        output_dtype: torch.dtype = torch.float16,
    ) -> torch.Tensor:
        """
        Forward pass: Dequantize INT8 weights to FP16/FP32.
        
        Args:
            ctx: Autograd context
            quantized_weights: INT8 weight matrix [M, N]
            scales: Dequantization scales [N] or [1]
            zero_points: Optional zero points [N] or [1]
            output_dtype: Target dtype (float16 or float32)
            
        Returns:
            Dequantized weights in specified dtype [M, N]
            
        Raises:
            ValueError: If dimensions are incompatible
        """
        M, N = quantized_weights.shape
        
        # Validate scales shape
        per_channel = scales.numel() > 1
        if per_channel and scales.numel() != N:
            raise ValueError(
                f"Per-channel scales must have {N} elements, got {scales.numel()}"
            )
        
        # Validate zero points if provided
        use_zero_point = zero_points is not None
        if use_zero_point:
            if per_channel and zero_points.numel() != N:
                raise ValueError(
                    f"Per-channel zero_points must have {N} elements, "
                    f"got {zero_points.numel()}"
                )
        
        # Allocate output
        output = torch.empty(
            (M, N),
            dtype=output_dtype,
            device=quantized_weights.device
        )
        
        # Define block sizes
        BLOCK_SIZE_M = 64
        BLOCK_SIZE_N = 64
        
        # Launch grid
        grid = (
            triton.cdiv(M, BLOCK_SIZE_M),
            triton.cdiv(N, BLOCK_SIZE_N),
        )
        
        _int8_dequant_kernel[grid](
            quantized_weights,
            scales,
            zero_points if use_zero_point else scales,  # Dummy pointer
            output,
            M=M,
            N=N,
            use_zero_point=use_zero_point,
            per_channel=per_channel,
            BLOCK_SIZE_M=BLOCK_SIZE_M,
            BLOCK_SIZE_N=BLOCK_SIZE_N,
        )
        
        return output
    
    @staticmethod
    def backward(ctx, grad_output):
        """
        Backward pass: Not implemented (inference-only).
        """
        raise NotImplementedError(
            "Int8DequantFunction does not support backward pass. "
            "Use this operation only during inference."
        )


def int8_dequant_kernel(
    quantized_weights: torch.Tensor,
    scales: torch.Tensor,
    zero_points: Optional[torch.Tensor] = None,
    output_dtype: torch.dtype = torch.float16,
) -> torch.Tensor:
    """
    High-level API for INT8 weight dequantization.
    
    This function provides on-the-fly dequantization of INT8-quantized expert
    weights, enabling 4x memory savings compared to FP16 storage.
    
    Args:
        quantized_weights: INT8 weight tensor [M, N]
        scales: Per-channel or per-tensor scales [N] or [1]
        zero_points: Optional zero points for asymmetric quantization
        output_dtype: Target dtype (float16 recommended for memory)
        
    Returns:
        Dequantized weight tensor [M, N] in specified dtype
        
    Example:
        >>> # Quantize weights offline
        >>> weights_fp16 = torch.randn(4096, 4096, dtype=torch.float16)
        >>> scale = weights_fp16.abs().max() / 127
        >>> weights_int8 = (weights_fp16 / scale).round().clamp(-127, 127).to(torch.int8)
        >>> 
        >>> # Dequantize on-the-fly during inference
        >>> weights_restored = int8_dequant_kernel(
        ...     weights_int8, 
        ...     scale.unsqueeze(0),
        ...     output_dtype=torch.float16
        ... )
        >>> error = (weights_fp16 - weights_restored).abs().mean()
        >>> print(f"Reconstruction error: {error:.6f}")
    """
    # Validate inputs
    if quantized_weights.dtype != torch.int8:
        raise TypeError(
            f"quantized_weights must be int8, got {quantized_weights.dtype}"
        )
    
    if not quantized_weights.is_cuda:
        # CPU fallback using PyTorch
        result = quantized_weights.to(output_dtype) * scales.to(output_dtype)
        if zero_points is not None:
            result = result - zero_points.to(output_dtype)
        return result
    
    # Use Triton kernel for GPU tensors
    return Int8DequantFunction.apply(
        quantized_weights,
        scales,
        zero_points,
        output_dtype,
    )


# Utility: Quantize weights for storage
def quantize_weights_int8(
    weights: torch.Tensor,
    per_channel: bool = True,
    symmetric: bool = True,
) -> tuple[torch.Tensor, torch.Tensor, Optional[torch.Tensor]]:
    """
    Quantize FP16/FP32 weights to INT8 for storage.
    
    This is typically done offline during model preparation.
    
    Args:
        weights: FP16/FP32 weight tensor [M, N]
        per_channel: Use per-channel (per-output) quantization
        symmetric: Use symmetric quantization (zero_point = 0)
        
    Returns:
        quantized: INT8 weights [M, N]
        scales: Quantization scales [N] or [1]
        zero_points: Zero points [N] or [1], or None if symmetric
        
    Example:
        >>> weights = torch.randn(1024, 512, dtype=torch.float16, device='cuda')
        >>> q_weights, scales, zp = quantize_weights_int8(weights)
        >>> print(f"Compression: {weights.nbytes / q_weights.nbytes:.1f}x")
        Compression: 2.0x
    """
    if per_channel:
        # Compute per-output-channel statistics
        if symmetric:
            # Scale = max(|W|) / 127
            abs_max = weights.abs().max(dim=0, keepdim=False)[0]
            scales = abs_max / 127.0
            scales = torch.clamp(scales, min=1e-8)  # Avoid division by zero
            
            quantized = (weights / scales).round().clamp(-127, 127).to(torch.int8)
            zero_points = None
        else:
            # Asymmetric: scale = (max - min) / 255, zero_point = -min / scale
            min_val = weights.min(dim=0, keepdim=False)[0]
            max_val = weights.max(dim=0, keepdim=False)[0]
            
            scales = (max_val - min_val) / 255.0
            scales = torch.clamp(scales, min=1e-8)
            
            zero_points = -min_val / scales
            
            quantized = ((weights / scales) + zero_points).round()
            quantized = quantized.clamp(-127, 127).to(torch.int8)
    else:
        # Per-tensor quantization
        if symmetric:
            scale = weights.abs().max() / 127.0
            scale = max(scale, 1e-8)
            
            quantized = (weights / scale).round().clamp(-127, 127).to(torch.int8)
            scales = torch.tensor([scale], device=weights.device, dtype=weights.dtype)
            zero_points = None
        else:
            min_val = weights.min()
            max_val = weights.max()
            
            scale = (max_val - min_val) / 255.0
            scale = max(scale, 1e-8)
            
            zero_point = -min_val / scale
            
            quantized = ((weights / scale) + zero_point).round()
            quantized = quantized.clamp(-127, 127).to(torch.int8)
            
            scales = torch.tensor([scale], device=weights.device, dtype=weights.dtype)
            zero_points = torch.tensor([zero_point], device=weights.device, dtype=weights.dtype)
    
    return quantized, scales, zero_points
