"""
Example: Basic Router and Worker Setup

This script demonstrates how to set up and run a minimal Volt-Infer
cluster with one router and multiple workers.
"""

import asyncio
import torch
from volt.core.config import VoltConfig, NodeType, WorkerConfig, RouterConfig
from volt.runtime.router import RouterNode
from volt.runtime.worker import WorkerNode


async def run_worker(expert_ids, port):
    """Run a worker node."""
    config = VoltConfig(
        node_type=NodeType.WORKER,
        node_id=f"worker-{port}",
        worker=WorkerConfig(
            expert_ids=expert_ids,
            device="cuda:0",
        ),
    )
    config.network.port = port
    
    worker = WorkerNode(config)
    await worker.start()
    
    print(f"✓ Worker started on port {port}, hosting experts {expert_ids}")
    
    # Keep running
    await asyncio.Event().wait()


async def run_router():
    """Run router node."""
    config = VoltConfig(
        node_type=NodeType.ROUTER,
        node_id="router-1",
        router=RouterConfig(),
    )
    
    router = RouterNode(config)
    await router.start()
    
    print("✓ Router started")
    
    # Simulate inference request
    hidden_states = torch.randn(32, 4096, device='cuda')
    
    print(f"Sending batch of {hidden_states.shape[0]} tokens...")
    output = await router.forward(hidden_states)
    
    print(f"✓ Received output: {output.shape}")
    print(f"  Mean: {output.mean().item():.4f}")
    print(f"  Std: {output.std().item():.4f}")


async def main():
    """Run example cluster."""
    print("=" * 60)
    print("Volt-Infer Example: Router + Workers")
    print("=" * 60)
    
    # Start workers in background
    worker_tasks = [
        asyncio.create_task(run_worker([0, 1], 50100)),
        asyncio.create_task(run_worker([2, 3], 50101)),
    ]
    
    # Wait for workers to initialize
    await asyncio.sleep(2.0)
    
    # Run router
    await run_router()
    
    print("\n✓ Example completed successfully!")


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        print("\n\n✗ Interrupted by user")
