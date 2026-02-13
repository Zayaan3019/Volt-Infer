"""
Pytest configuration and fixtures.
"""

import pytest
import torch


@pytest.fixture(scope="session")
def cuda_available():
    """Check if CUDA is available."""
    return torch.cuda.is_available()


@pytest.fixture(scope="session")
def device(cuda_available):
    """Get test device."""
    return "cuda" if cuda_available else "cpu"


@pytest.fixture
def sample_tensor(device):
    """Create a sample tensor for testing."""
    return torch.randn(32, 128, device=device)
