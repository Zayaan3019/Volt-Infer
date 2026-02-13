# Volt-Infer Examples

This directory contains example scripts demonstrating various Volt-Infer features.

## Available Examples

### 1. Basic Inference (`basic_inference.py`)
Demonstrates setting up a minimal cluster with router and workers.

```bash
python basic_inference.py
```

### 2. Kernel Benchmark (`../benchmarks/kernel_benchmark.py`)
Compares Triton kernel performance against PyTorch baseline.

```bash
cd ../benchmarks
python kernel_benchmark.py
```

## Custom Examples

You can create custom examples by:

1. Importing Volt-Infer modules:
```python
from volt.core.config import VoltConfig, NodeType
from volt.runtime.router import RouterNode
from volt.runtime.worker import WorkerNode
```

2. Configuring nodes:
```python
config = VoltConfig.from_env(NodeType.ROUTER, "my-router")
```

3. Running inference:
```python
router = RouterNode(config)
await router.start()

output = await router.forward(hidden_states)
```

## Requirements

- Redis running on localhost:6379
- CUDA-capable GPU (for GPU examples)
- All dependencies from requirements.txt installed

## Support

For questions or issues with examples, please open an issue on GitHub.
