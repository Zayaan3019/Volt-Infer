# Quick Start Guide

Get Volt-Infer running in 5 minutes.

## Prerequisites

1. **Python 3.11+** installed
2. **CUDA 11.8+** (for GPU support)
3. **Redis** running locally or accessible via network

## Step 1: Installation

```bash
# Clone repository
git clone https://github.com/yourusername/volt-infer.git
cd volt-infer

# Create virtual environment
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Install package
pip install -e .
```

## Step 2: Start Redis

### Option A: Docker (Recommended)
```bash
docker run -d --name volt-redis -p 6379:6379 redis:7-alpine
```

### Option B: Local Installation
```bash
# Ubuntu/Debian
sudo apt-get install redis-server
sudo systemctl start redis

# macOS
brew install redis
brew services start redis

# Windows
# Download from https://redis.io/download
```

Verify Redis is running:
```bash
redis-cli ping  # Should return "PONG"
```

## Step 3: Launch Worker Nodes

Open **Terminal 1**:
```bash
volt-worker \
    --node-id worker-1 \
    --expert-ids 0,1,2,3 \
    --port 50100 \
    --device cuda:0
```

Open **Terminal 2**:
```bash
volt-worker \
    --node-id worker-2 \
    --expert-ids 4,5,6,7 \
    --port 50101 \
    --device cuda:0
```

You should see:
```
==============================================================
Volt-Infer Worker Node
==============================================================
Node ID:      worker-1
Expert IDs:   [0, 1, 2, 3]
Device:       cuda:0
Port:         50100
Redis:        redis://localhost:6379/0
Metrics:      http://0.0.0.0:9091/metrics

✓ Worker node started
```

## Step 4: Launch Router Node

Open **Terminal 3**:
```bash
volt-router \
    --node-id router-1 \
    --port 50051
```

You should see:
```
==============================================================
Volt-Infer Router Node
==============================================================
Node ID:      router-1
Port:         50051
Redis:        redis://localhost:6379/0
Metrics:      http://0.0.0.0:9090/metrics

✓ Router node started
✓ Discovered Expert 0 at localhost:50100
✓ Discovered Expert 1 at localhost:50100
...
```

## Step 5: Run Inference

Open **Terminal 4** or Python REPL:

```python
import asyncio
import torch
from volt.core.config import VoltConfig, NodeType
from volt.runtime.router import RouterNode

async def main():
    # Create router config
    config = VoltConfig.from_env(NodeType.ROUTER, "test-router")
    router = RouterNode(config)
    
    # Start router
    await router.start()
    
    # Create sample input (batch of 32 tokens with 4096-dim embeddings)
    hidden_states = torch.randn(32, 4096, device='cuda', dtype=torch.float16)
    
    print(f"Input shape: {hidden_states.shape}")
    
    # Run inference
    output = await router.forward(hidden_states)
    
    print(f"Output shape: {output.shape}")
    print(f"Output mean: {output.mean().item():.4f}")
    print(f"Output std: {output.std().item():.4f}")
    
    # Cleanup
    await router.shutdown()

# Run
asyncio.run(main())
```

Expected output:
```
Input shape: torch.Size([32, 4096])
Output shape: torch.Size([32, 4096])
Output mean: 0.0023
Output std: 0.9847
```

## Step 6: Monitor Metrics

Open browser to view Prometheus metrics:

- **Router Metrics**: http://localhost:9090/metrics
- **Worker 1 Metrics**: http://localhost:9091/metrics
- **Worker 2 Metrics**: http://localhost:9092/metrics

Key metrics to watch:
```prometheus
voltinfer_request_latency_seconds_bucket{le="0.05"}
voltinfer_queue_depth
voltinfer_expert_requests_total
```

## Troubleshooting

### Issue: "Connection refused to Redis"
**Solution**: Ensure Redis is running
```bash
redis-cli ping
```

### Issue: "CUDA out of memory"
**Solution**: Reduce batch size or enable quantization
```bash
export VOLT_MODEL_USE_QUANTIZATION=true
```

### Issue: "Expert not found in registry"
**Solution**: Wait 5 seconds for worker registration
```python
await asyncio.sleep(5)  # After starting workers
```

### Issue: "ImportError: No module named 'triton'"
**Solution**: Install Triton
```bash
pip install triton>=2.1.0
```

## Next Steps

1. **Read Documentation**: See [README.md](README.md) for detailed architecture
2. **Run Benchmarks**: `python benchmarks/kernel_benchmark.py`
3. **Run Tests**: `pytest tests/ -v`
4. **Explore Examples**: See [examples/](examples/) directory
5. **Configure Autoscaling**: See [ARCHITECTURE.md](ARCHITECTURE.md)

## Production Deployment

For production use:

1. **Replace mock checkpoints** with real model weights
2. **Configure Redis cluster** for high availability
3. **Set up Prometheus + Grafana** for monitoring
4. **Enable autoscaling** with webhook to orchestration platform
5. **Use uvloop** for faster async I/O:
   ```bash
   pip install uvloop
   ```

## Getting Help

- **Issues**: https://github.com/yourusername/volt-infer/issues
- **Discussions**: https://github.com/yourusername/volt-infer/discussions

---

**You're ready to scale MoE inference! 🚀**
