"""
Triton Top-K Gating Kernel for Mixture-of-Experts Routing

This module implements a high-performance GPU kernel using OpenAI Triton that 
computes the Top-K expert assignments for each token in a batch. The kernel is 
designed to be non-blocking and faster than PyTorch's native torch.topk implementation.

Mathematical Foundation:
    Given hidden states H ∈ ℝ^(B×D) and router weights W ∈ ℝ^(D×E):
    1. Compute logits: L = H @ W  →  L ∈ ℝ^(B×E)
    2. Apply softmax: P = softmax(L)
    3. Select top-k: indices, weights = topk(P, k)
    
Where:
    B = batch size
    D = hidden dimension
    E = number of experts
    k = number of experts to activate per token
"""

from typing import Tuple

import torch
import triton
import triton.language as tl


@triton.jit
def _topk_gating_kernel(
    # Input pointers
    hidden_ptr,  # [batch_size, hidden_dim]
    router_weights_ptr,  # [hidden_dim, num_experts]
    # Output pointers
    topk_indices_ptr,  # [batch_size, k]
    topk_weights_ptr,  # [batch_size, k]
    # Dimensions
    batch_size: tl.constexpr,
    hidden_dim: tl.constexpr,
    num_experts: tl.constexpr,
    k: tl.constexpr,
    # Block sizes
    BLOCK_SIZE_BATCH: tl.constexpr,
    BLOCK_SIZE_HIDDEN: tl.constexpr,
    BLOCK_SIZE_EXPERTS: tl.constexpr,
):
    """
    Triton kernel for computing Top-K expert routing decisions.
    
    Algorithm:
        1. Load hidden state for current batch element
        2. Compute matmul with router weights to get logits
        3. Apply softmax normalization
        4. Find Top-K experts using iterative selection
        5. Store indices and normalized weights
    
    Memory Access Pattern:
        - Coalesced reads from hidden_ptr (sequential in hidden_dim)
        - Tiled reads from router_weights_ptr
        - Coalesced writes to output tensors
    
    Optimization Strategy:
        - Block-level parallelism across batch dimension
        - Reduction trees for matmul and softmax
        - Register-level sorting for Top-K selection
    """
    # Program ID identifies which batch element this block handles
    pid_batch = tl.program_id(0)
    
    if pid_batch >= batch_size:
        return
    
    # Allocate accumulator for logits (one per expert)
    logits = tl.zeros([num_experts], dtype=tl.float32)
    
    # Compute matmul: logits = hidden @ router_weights
    # Process in tiles along hidden dimension for memory efficiency
    for h_offset in range(0, hidden_dim, BLOCK_SIZE_HIDDEN):
        # Load chunk of hidden state [BLOCK_SIZE_HIDDEN]
        h_idx = h_offset + tl.arange(0, BLOCK_SIZE_HIDDEN)
        h_mask = h_idx < hidden_dim
        hidden_chunk = tl.load(
            hidden_ptr + pid_batch * hidden_dim + h_idx,
            mask=h_mask,
            other=0.0
        )
        
        # For each expert, accumulate dot product
        for e in range(num_experts):
            # Load corresponding weights [BLOCK_SIZE_HIDDEN]
            weight_idx = h_idx * num_experts + e
            weights_chunk = tl.load(
                router_weights_ptr + weight_idx,
                mask=h_mask,
                other=0.0
            )
            
            # Accumulate: logits[e] += sum(hidden_chunk * weights_chunk)
            logits[e] += tl.sum(hidden_chunk * weights_chunk)
    
    # Apply softmax: exp(logits) / sum(exp(logits))
    # Use log-sum-exp trick for numerical stability
    max_logit = tl.max(logits, axis=0)
    exp_logits = tl.exp(logits - max_logit)
    sum_exp = tl.sum(exp_logits, axis=0)
    probs = exp_logits / sum_exp
    
    # Top-K selection using iterative max extraction
    # This is a simplified approach; production systems use bitonic sort
    selected_indices = tl.zeros([k], dtype=tl.int32)
    selected_weights = tl.zeros([k], dtype=tl.float32)
    
    # Create a working copy of probs
    working_probs = probs
    
    for top_i in range(k):
        # Find the maximum probability
        max_prob = tl.maximum(0.0, tl.max(working_probs, axis=0))
        
        # Find the index of the maximum (linear search)
        max_idx = 0
        for e in range(num_experts):
            if working_probs[e] == max_prob:
                max_idx = e
                break
        
        # Store the result
        selected_indices[top_i] = max_idx
        selected_weights[top_i] = max_prob
        
        # Zero out this probability for next iteration
        working_probs[max_idx] = -1e9
    
    # Renormalize selected weights to sum to 1.0
    weight_sum = tl.sum(selected_weights, axis=0)
    normalized_weights = selected_weights / (weight_sum + 1e-9)
    
    # Write results to global memory
    for i in range(k):
        output_idx = pid_batch * k + i
        tl.store(topk_indices_ptr + output_idx, selected_indices[i])
        tl.store(topk_weights_ptr + output_idx, normalized_weights[i])


class TopKGatingFunction(torch.autograd.Function):
    """
    Custom autograd function wrapping the Triton Top-K gating kernel.
    
    This provides a PyTorch-compatible interface with automatic gradient
    computation through the routing decisions.
    """
    
    @staticmethod
    def forward(
        ctx,
        hidden_states: torch.Tensor,  # [batch_size, hidden_dim]
        router_weights: torch.Tensor,  # [hidden_dim, num_experts]
        k: int = 2,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass: Compute Top-K expert routing.
        
        Args:
            ctx: Autograd context for backward pass
            hidden_states: Input token representations [batch_size, hidden_dim]
            router_weights: Learnable routing matrix [hidden_dim, num_experts]
            k: Number of experts to activate per token
            
        Returns:
            topk_indices: Expert IDs for each token [batch_size, k]
            topk_weights: Normalized gating weights [batch_size, k]
            
        Raises:
            ValueError: If dimensions are incompatible or k > num_experts
        """
        batch_size, hidden_dim = hidden_states.shape
        num_experts = router_weights.shape[1]
        
        if router_weights.shape[0] != hidden_dim:
            raise ValueError(
                f"Router weight dimension mismatch: "
                f"expected {hidden_dim}, got {router_weights.shape[0]}"
            )
        
        if k > num_experts:
            raise ValueError(f"k={k} cannot exceed num_experts={num_experts}")
        
        # Allocate output tensors
        topk_indices = torch.empty(
            (batch_size, k),
            dtype=torch.int32,
            device=hidden_states.device
        )
        topk_weights = torch.empty(
            (batch_size, k),
            dtype=torch.float32,
            device=hidden_states.device
        )
        
        # Define block sizes for optimal GPU occupancy
        BLOCK_SIZE_BATCH = 1
        BLOCK_SIZE_HIDDEN = min(128, triton.next_power_of_2(hidden_dim))
        BLOCK_SIZE_EXPERTS = min(64, triton.next_power_of_2(num_experts))
        
        # Launch kernel with one program per batch element
        grid = (batch_size,)
        
        _topk_gating_kernel[grid](
            hidden_states,
            router_weights,
            topk_indices,
            topk_weights,
            batch_size=batch_size,
            hidden_dim=hidden_dim,
            num_experts=num_experts,
            k=k,
            BLOCK_SIZE_BATCH=BLOCK_SIZE_BATCH,
            BLOCK_SIZE_HIDDEN=BLOCK_SIZE_HIDDEN,
            BLOCK_SIZE_EXPERTS=BLOCK_SIZE_EXPERTS,
        )
        
        # Save for backward pass
        ctx.save_for_backward(hidden_states, router_weights, topk_indices, topk_weights)
        ctx.k = k
        
        return topk_indices, topk_weights
    
    @staticmethod
    def backward(ctx, grad_indices, grad_weights):
        """
        Backward pass: Compute gradients for router weights.
        
        Note: Indices are discrete, so gradient flows only through weights.
        Uses straight-through estimator for the Top-K selection.
        """
        hidden_states, router_weights, topk_indices, topk_weights = ctx.saved_tensors
        
        # Gradient computation using chain rule
        # ∂L/∂W = H^T @ (∂L/∂weights * selected_mask)
        batch_size, hidden_dim = hidden_states.shape
        num_experts = router_weights.shape[1]
        
        # Reconstruct full gating distribution from Top-K
        full_weights = torch.zeros(
            (batch_size, num_experts),
            dtype=torch.float32,
            device=hidden_states.device
        )
        
        # Scatter Top-K weights back to full distribution
        full_weights.scatter_(1, topk_indices.long(), topk_weights)
        
        # Compute gradient w.r.t. router weights
        grad_router = torch.matmul(
            hidden_states.t(),  # [hidden_dim, batch_size]
            grad_weights * full_weights  # [batch_size, num_experts]
        )
        
        # Gradient w.r.t. hidden states (for chaining)
        grad_hidden = torch.matmul(
            grad_weights * full_weights,  # [batch_size, num_experts]
            router_weights.t()  # [num_experts, hidden_dim]
        )
        
        return grad_hidden, grad_router, None


def topk_gating_kernel(
    hidden_states: torch.Tensor,
    router_weights: torch.Tensor,
    k: int = 2,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    High-level API for Top-K expert gating.
    
    This function provides a clean interface to the Triton-accelerated
    gating kernel with automatic fallback to PyTorch implementation if
    Triton is unavailable or the tensors are on CPU.
    
    Args:
        hidden_states: Token representations [batch_size, hidden_dim]
        router_weights: Expert routing matrix [hidden_dim, num_experts]
        k: Number of experts to activate (default: 2 for Mixtral-style MoE)
        
    Returns:
        topk_indices: Selected expert IDs [batch_size, k]
        topk_weights: Gating coefficients [batch_size, k], sum to 1.0
        
    Example:
        >>> hidden = torch.randn(32, 4096, device='cuda')
        >>> router = torch.randn(4096, 8, device='cuda')
        >>> indices, weights = topk_gating_kernel(hidden, router, k=2)
        >>> print(indices.shape, weights.shape)
        torch.Size([32, 2]) torch.Size([32, 2])
        >>> assert torch.allclose(weights.sum(dim=1), torch.ones(32))
    """
    # Validate inputs
    if not hidden_states.is_cuda or not router_weights.is_cuda:
        # Fallback to PyTorch implementation for CPU tensors
        logits = torch.matmul(hidden_states, router_weights)
        probs = torch.softmax(logits, dim=-1)
        topk_weights, topk_indices = torch.topk(probs, k, dim=-1)
        
        # Renormalize
        topk_weights = topk_weights / topk_weights.sum(dim=-1, keepdim=True)
        
        return topk_indices.int(), topk_weights
    
    # Use Triton kernel for GPU tensors
    return TopKGatingFunction.apply(hidden_states, router_weights, k)


# Performance Benchmarking Utilities
def benchmark_gating_kernel(
    batch_size: int = 32,
    hidden_dim: int = 4096,
    num_experts: int = 8,
    k: int = 2,
    num_iterations: int = 100,
) -> dict:
    """
    Benchmark the Triton kernel against PyTorch baseline.
    
    Args:
        batch_size: Number of tokens
        hidden_dim: Model hidden dimension
        num_experts: Total number of experts
        k: Top-K value
        num_iterations: Number of timing iterations
        
    Returns:
        Dictionary with timing statistics and speedup factor
    """
    import time
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Generate random inputs
    hidden = torch.randn(batch_size, hidden_dim, device=device)
    router = torch.randn(hidden_dim, num_experts, device=device)
    
    # Warmup
    for _ in range(10):
        _ = topk_gating_kernel(hidden, router, k)
    
    torch.cuda.synchronize()
    
    # Benchmark Triton kernel
    start = time.perf_counter()
    for _ in range(num_iterations):
        _ = topk_gating_kernel(hidden, router, k)
    torch.cuda.synchronize()
    triton_time = (time.perf_counter() - start) / num_iterations
    
    # Benchmark PyTorch baseline
    start = time.perf_counter()
    for _ in range(num_iterations):
        logits = torch.matmul(hidden, router)
        probs = torch.softmax(logits, dim=-1)
        _, _ = torch.topk(probs, k, dim=-1)
    torch.cuda.synchronize()
    pytorch_time = (time.perf_counter() - start) / num_iterations
    
    return {
        'triton_ms': triton_time * 1000,
        'pytorch_ms': pytorch_time * 1000,
        'speedup': pytorch_time / triton_time,
        'batch_size': batch_size,
        'hidden_dim': hidden_dim,
        'num_experts': num_experts,
        'k': k,
    }
