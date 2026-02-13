"""
CLI entry points for Volt-Infer.

Provides command-line interfaces for starting router and worker nodes.
"""

import asyncio
import logging
import sys
import click

from volt.core.config import VoltConfig, NodeType
from volt.runtime.router import RouterNode
from volt.runtime.worker import WorkerNode
from volt.mlops.metrics import VoltMetrics, MetricsServer


logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


@click.group()
@click.version_option(version="0.1.0")
def cli():
    """Volt-Infer: Distributed MoE Inference Engine."""
    pass


@cli.command()
@click.option('--node-id', default='router-1', help='Node identifier')
@click.option('--port', default=50051, type=int, help='Network port')
@click.option('--redis-url', default='redis://localhost:6379/0', help='Redis URL')
@click.option('--metrics-port', default=9090, type=int, help='Metrics port')
def router(node_id, port, redis_url, metrics_port):
    """Start a Router node."""
    click.echo("=" * 60)
    click.echo("Volt-Infer Router Node")
    click.echo("=" * 60)
    
    config = VoltConfig.from_env(NodeType.ROUTER, node_id)
    config.network.port = port
    config.redis.url = redis_url
    config.observability.metrics_port = metrics_port
    
    click.echo(f"Node ID:      {node_id}")
    click.echo(f"Port:         {port}")
    click.echo(f"Redis:        {redis_url}")
    click.echo(f"Metrics:      http://0.0.0.0:{metrics_port}/metrics")
    click.echo()
    
    async def run():
        # Initialize metrics
        metrics = VoltMetrics(node_type="router", node_id=node_id)
        metrics_server = MetricsServer(metrics, port=metrics_port)
        metrics_server.start()
        
        # Start router
        router_node = RouterNode(config)
        await router_node.start()
        
        click.echo("✓ Router node started")
        
        # Keep running
        try:
            await asyncio.Event().wait()
        except KeyboardInterrupt:
            click.echo("\n\nShutting down...")
            await router_node.shutdown()
    
    try:
        asyncio.run(run())
    except KeyboardInterrupt:
        sys.exit(0)


@cli.command()
@click.option('--node-id', required=True, help='Node identifier')
@click.option('--expert-ids', required=True, help='Comma-separated expert IDs (e.g., 0,1,2)')
@click.option('--port', default=50100, type=int, help='Network port')
@click.option('--device', default='cuda:0', help='Device (cuda:0, cuda:1, cpu)')
@click.option('--redis-url', default='redis://localhost:6379/0', help='Redis URL')
@click.option('--metrics-port', default=9091, type=int, help='Metrics port')
def worker(node_id, expert_ids, port, device, redis_url, metrics_port):
    """Start a Worker node."""
    click.echo("=" * 60)
    click.echo("Volt-Infer Worker Node")
    click.echo("=" * 60)
    
    # Parse expert IDs
    expert_id_list = [int(x.strip()) for x in expert_ids.split(',')]
    
    config = VoltConfig.from_env(NodeType.WORKER, node_id)
    config.network.port = port
    config.redis.url = redis_url
    config.worker.expert_ids = expert_id_list
    config.worker.device = device
    config.observability.metrics_port = metrics_port
    
    click.echo(f"Node ID:      {node_id}")
    click.echo(f"Expert IDs:   {expert_id_list}")
    click.echo(f"Device:       {device}")
    click.echo(f"Port:         {port}")
    click.echo(f"Redis:        {redis_url}")
    click.echo(f"Metrics:      http://0.0.0.0:{metrics_port}/metrics")
    click.echo()
    
    async def run():
        # Initialize metrics
        metrics = VoltMetrics(node_type="worker", node_id=node_id)
        metrics_server = MetricsServer(metrics, port=metrics_port)
        metrics_server.start()
        
        # Start worker
        worker_node = WorkerNode(config)
        await worker_node.start()
        
        click.echo("✓ Worker node started")
        
        # Keep running
        try:
            await asyncio.Event().wait()
        except KeyboardInterrupt:
            click.echo("\n\nShutting down...")
            await worker_node.shutdown()
    
    try:
        asyncio.run(run())
    except KeyboardInterrupt:
        sys.exit(0)


def router_main():
    """Entry point for router CLI."""
    sys.argv = [sys.argv[0], 'router'] + sys.argv[1:]
    cli()


def worker_main():
    """Entry point for worker CLI."""
    sys.argv = [sys.argv[0], 'worker'] + sys.argv[1:]
    cli()


if __name__ == '__main__':
    cli()
