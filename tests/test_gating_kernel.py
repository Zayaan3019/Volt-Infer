"""
Tests for Triton Gating Kernel

Validates correctness and performance of the Top-K gating implementation.
"""

import pytest
import torch
from volt.kernels.gating import topk_gating_kernel, TopKGatingFunction


class TestTopKGating:
    """Test suite for Top-K gating kernel."""
    
    @pytest.fixture
    def sample_inputs(self):
        """Create sample inputs for testing."""
        batch_size = 32
        hidden_dim = 128
        num_experts = 8
        
        hidden_states = torch.randn(batch_size, hidden_dim, device='cuda')
        router_weights = torch.randn(hidden_dim, num_experts, device='cuda')
        
        return hidden_states, router_weights
    
    def test_output_shapes(self, sample_inputs):
        """Test that output shapes are correct."""
        hidden_states, router_weights = sample_inputs
        batch_size = hidden_states.shape[0]
        k = 2
        
        indices, weights = topk_gating_kernel(hidden_states, router_weights, k=k)
        
        assert indices.shape == (batch_size, k)
        assert weights.shape == (batch_size, k)
        assert indices.dtype == torch.int32
        assert weights.dtype == torch.float32
    
    def test_weights_sum_to_one(self, sample_inputs):
        """Test that gating weights are normalized."""
        hidden_states, router_weights = sample_inputs
        k = 2
        
        _, weights = topk_gating_kernel(hidden_states, router_weights, k=k)
        
        # Check that weights sum to 1.0 for each token
        weight_sums = weights.sum(dim=1)
        torch.testing.assert_close(
            weight_sums,
            torch.ones(hidden_states.shape[0], device='cuda'),
            rtol=1e-4,
            atol=1e-4,
        )
    
    def test_expert_indices_in_range(self, sample_inputs):
        """Test that expert indices are valid."""
        hidden_states, router_weights = sample_inputs
        num_experts = router_weights.shape[1]
        k = 2
        
        indices, _ = topk_gating_kernel(hidden_states, router_weights, k=k)
        
        assert (indices >= 0).all()
        assert (indices < num_experts).all()
    
    def test_consistency_with_pytorch(self, sample_inputs):
        """Test that kernel produces similar results to PyTorch."""
        hidden_states, router_weights = sample_inputs
        k = 2
        
        # Triton kernel
        triton_indices, triton_weights = topk_gating_kernel(
            hidden_states, router_weights, k=k
        )
        
        # PyTorch baseline
        logits = torch.matmul(hidden_states, router_weights)
        probs = torch.softmax(logits, dim=-1)
        pytorch_weights, pytorch_indices = torch.topk(probs, k, dim=-1)
        pytorch_weights = pytorch_weights / pytorch_weights.sum(dim=1, keepdim=True)
        
        # Compare (allow small numerical differences)
        torch.testing.assert_close(
            triton_indices.long(),
            pytorch_indices,
            rtol=0.0,
            atol=0.0,  # Indices should match exactly
        )
        
        torch.testing.assert_close(
            triton_weights,
            pytorch_weights,
            rtol=1e-3,
            atol=1e-3,
        )
    
    def test_different_k_values(self, sample_inputs):
        """Test with different Top-K values."""
        hidden_states, router_weights = sample_inputs
        
        for k in [1, 2, 4]:
            indices, weights = topk_gating_kernel(hidden_states, router_weights, k=k)
            
            assert indices.shape[1] == k
            assert weights.shape[1] == k
    
    def test_cpu_fallback(self):
        """Test CPU fallback when CUDA unavailable."""
        batch_size = 16
        hidden_dim = 64
        num_experts = 4
        k = 2
        
        hidden_states = torch.randn(batch_size, hidden_dim)
        router_weights = torch.randn(hidden_dim, num_experts)
        
        indices, weights = topk_gating_kernel(hidden_states, router_weights, k=k)
        
        assert indices.shape == (batch_size, k)
        assert weights.shape == (batch_size, k)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
