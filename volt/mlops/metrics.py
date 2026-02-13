"""
Prometheus Metrics Exporter for Volt-Infer

Provides comprehensive observability for distributed MoE inference:
- Request latency (P50, P90, P99)
- Expert utilization rates
- Queue depths for autoscaling triggers
- Network bandwidth usage
- GPU memory consumption
"""

import time
from typing import Dict, Optional
import logging
from prometheus_client import (
    Counter,
    Histogram,
    Gauge,
    Info,
    start_http_server,
    CollectorRegistry,
)

logger = logging.getLogger(__name__)


class VoltMetrics:
    """
    Centralized metrics collection for Volt-Infer nodes.
    
    All metrics follow Prometheus naming conventions:
        - snake_case
        - suffix with unit (_seconds, _bytes, _total)
        - namespace prefix (voltinfer_)
    """
    
    def __init__(
        self,
        node_type: str,
        node_id: str,
        registry: Optional[CollectorRegistry] = None,
    ):
        """
        Initialize metrics collector.
        
        Args:
            node_type: "router" or "worker"
            node_id: Unique node identifier
            registry: Optional custom registry (default: global)
        """
        self.node_type = node_type
        self.node_id = node_id
        self.registry = registry
        
        # Node info
        self.node_info = Info(
            'voltinfer_node',
            'Node information',
            registry=self.registry,
        )
        self.node_info.info({
            'node_type': node_type,
            'node_id': node_id,
            'version': '0.1.0',
        })
        
        # === Router Metrics ===
        
        # Request throughput
        self.requests_total = Counter(
            'voltinfer_requests_total',
            'Total number of inference requests',
            labelnames=['status'],  # success, failure, timeout
            registry=self.registry,
        )
        
        # Request latency distribution
        self.request_latency_seconds = Histogram(
            'voltinfer_request_latency_seconds',
            'Request processing latency',
            buckets=[0.001, 0.005, 0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1.0, 2.5],
            registry=self.registry,
        )
        
        # Per-expert latency
        self.expert_latency_seconds = Histogram(
            'voltinfer_expert_latency_seconds',
            'Expert node processing latency',
            labelnames=['expert_id'],
            buckets=[0.001, 0.005, 0.01, 0.025, 0.05, 0.1, 0.25],
            registry=self.registry,
        )
        
        # Expert utilization
        self.expert_requests_total = Counter(
            'voltinfer_expert_requests_total',
            'Requests routed to each expert',
            labelnames=['expert_id'],
            registry=self.registry,
        )
        
        # Expert failures
        self.expert_failures_total = Counter(
            'voltinfer_expert_failures_total',
            'Expert node failures',
            labelnames=['expert_id', 'reason'],  # timeout, connection_error, etc.
            registry=self.registry,
        )
        
        # === Worker Metrics ===
        
        # Queue depth (critical for autoscaling)
        self.queue_depth = Gauge(
            'voltinfer_queue_depth',
            'Current request queue depth',
            registry=self.registry,
        )
        
        # Max queue depth
        self.queue_depth_max = Gauge(
            'voltinfer_queue_depth_max',
            'Maximum queue capacity',
            registry=self.registry,
        )
        
        # Expert forward pass time
        self.expert_forward_seconds = Histogram(
            'voltinfer_expert_forward_seconds',
            'Expert forward pass duration',
            labelnames=['expert_id'],
            buckets=[0.001, 0.002, 0.005, 0.01, 0.02, 0.05, 0.1],
            registry=self.registry,
        )
        
        # GPU memory usage
        self.gpu_memory_bytes = Gauge(
            'voltinfer_gpu_memory_bytes',
            'GPU memory usage',
            labelnames=['device', 'type'],  # type: allocated, reserved
            registry=self.registry,
        )
        
        # === Prefetch Metrics ===
        
        self.prefetch_requests_total = Counter(
            'voltinfer_prefetch_requests_total',
            'Total prefetch requests',
            registry=self.registry,
        )
        
        self.prefetch_hits_total = Counter(
            'voltinfer_prefetch_hits_total',
            'Prefetch requests that completed in time',
            registry=self.registry,
        )
        
        self.prefetch_accuracy = Gauge(
            'voltinfer_prefetch_accuracy',
            'Prefetch prediction accuracy',
            registry=self.registry,
        )
        
        # === Network Metrics ===
        
        self.network_bytes_sent = Counter(
            'voltinfer_network_bytes_sent_total',
            'Total bytes sent over network',
            labelnames=['destination'],
            registry=self.registry,
        )
        
        self.network_bytes_received = Counter(
            'voltinfer_network_bytes_received_total',
            'Total bytes received',
            labelnames=['source'],
            registry=self.registry,
        )
        
        logger.info(f"Metrics initialized for {node_type} node: {node_id}")
    
    # === Convenience Methods ===
    
    def record_request(self, latency_seconds: float, status: str = 'success') -> None:
        """
        Record a completed request.
        
        Args:
            latency_seconds: Request duration
            status: 'success', 'failure', or 'timeout'
        """
        self.requests_total.labels(status=status).inc()
        self.request_latency_seconds.observe(latency_seconds)
    
    def record_expert_request(
        self,
        expert_id: int,
        latency_seconds: float,
        success: bool = True,
        failure_reason: Optional[str] = None,
    ) -> None:
        """
        Record expert node request.
        
        Args:
            expert_id: Expert identifier
            latency_seconds: Request duration
            success: Whether request succeeded
            failure_reason: Failure reason if not successful
        """
        self.expert_requests_total.labels(expert_id=str(expert_id)).inc()
        self.expert_latency_seconds.labels(expert_id=str(expert_id)).observe(
            latency_seconds
        )
        
        if not success and failure_reason:
            self.expert_failures_total.labels(
                expert_id=str(expert_id),
                reason=failure_reason,
            ).inc()
    
    def update_queue_depth(self, current: int, maximum: int) -> None:
        """
        Update queue depth metrics.
        
        Args:
            current: Current queue size
            maximum: Maximum queue capacity
        """
        self.queue_depth.set(current)
        self.queue_depth_max.set(maximum)
    
    def record_expert_forward(self, expert_id: int, duration_seconds: float) -> None:
        """
        Record expert forward pass timing.
        
        Args:
            expert_id: Expert identifier
            duration_seconds: Forward pass duration
        """
        self.expert_forward_seconds.labels(expert_id=str(expert_id)).observe(
            duration_seconds
        )
    
    def update_gpu_memory(self, device: str, allocated_bytes: int, reserved_bytes: int) -> None:
        """
        Update GPU memory metrics.
        
        Args:
            device: Device identifier (e.g., 'cuda:0')
            allocated_bytes: Currently allocated memory
            reserved_bytes: Reserved memory
        """
        self.gpu_memory_bytes.labels(device=device, type='allocated').set(allocated_bytes)
        self.gpu_memory_bytes.labels(device=device, type='reserved').set(reserved_bytes)
    
    def record_prefetch(self, hit: bool, accuracy: Optional[float] = None) -> None:
        """
        Record prefetch metrics.
        
        Args:
            hit: Whether prefetch completed in time
            accuracy: Optional prediction accuracy
        """
        self.prefetch_requests_total.inc()
        
        if hit:
            self.prefetch_hits_total.inc()
        
        if accuracy is not None:
            self.prefetch_accuracy.set(accuracy)
    
    def record_network_transfer(
        self,
        bytes_sent: int = 0,
        bytes_received: int = 0,
        destination: str = 'unknown',
        source: str = 'unknown',
    ) -> None:
        """
        Record network transfer metrics.
        
        Args:
            bytes_sent: Bytes sent
            bytes_received: Bytes received
            destination: Destination node
            source: Source node
        """
        if bytes_sent > 0:
            self.network_bytes_sent.labels(destination=destination).inc(bytes_sent)
        
        if bytes_received > 0:
            self.network_bytes_received.labels(source=source).inc(bytes_received)


class MetricsServer:
    """
    HTTP server for Prometheus metrics scraping.
    
    Exposes /metrics endpoint on configured port.
    """
    
    def __init__(
        self,
        metrics: VoltMetrics,
        port: int = 9090,
        host: str = '0.0.0.0',
    ):
        """
        Initialize metrics server.
        
        Args:
            metrics: VoltMetrics instance
            port: HTTP port
            host: Bind address
        """
        self.metrics = metrics
        self.port = port
        self.host = host
        self.server = None
    
    def start(self) -> None:
        """Start HTTP server for metrics."""
        try:
            start_http_server(self.port, addr=self.host, registry=self.metrics.registry)
            logger.info(f"Metrics server started on {self.host}:{self.port}/metrics")
        
        except Exception as e:
            logger.error(f"Failed to start metrics server: {e}")
    
    def stop(self) -> None:
        """Stop metrics server."""
        # Prometheus client doesn't provide stop method
        # Server runs in daemon thread and stops when process exits
        logger.info("Metrics server stopping...")


# Utility: GPU memory tracking
def update_gpu_metrics(metrics: VoltMetrics, device: str = 'cuda:0') -> None:
    """
    Update GPU memory metrics from PyTorch.
    
    Args:
        metrics: VoltMetrics instance
        device: Device to query
    """
    try:
        import torch
        
        if not torch.cuda.is_available():
            return
        
        device_obj = torch.device(device)
        
        allocated = torch.cuda.memory_allocated(device_obj)
        reserved = torch.cuda.memory_reserved(device_obj)
        
        metrics.update_gpu_memory(device, allocated, reserved)
    
    except Exception as e:
        logger.error(f"Failed to update GPU metrics: {e}")


# Context manager for timing
class TimedOperation:
    """Context manager for timing operations and recording metrics."""
    
    def __init__(
        self,
        metrics: VoltMetrics,
        operation_type: str,
        labels: Optional[Dict[str, str]] = None,
    ):
        """
        Initialize timed operation.
        
        Args:
            metrics: VoltMetrics instance
            operation_type: Type of operation ('request', 'expert', 'forward')
            labels: Optional labels for metrics
        """
        self.metrics = metrics
        self.operation_type = operation_type
        self.labels = labels or {}
        self.start_time = None
    
    def __enter__(self):
        """Start timing."""
        self.start_time = time.perf_counter()
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        """Stop timing and record metric."""
        duration = time.perf_counter() - self.start_time
        
        if self.operation_type == 'request':
            status = 'success' if exc_type is None else 'failure'
            self.metrics.record_request(duration, status)
        
        elif self.operation_type == 'expert':
            expert_id = self.labels.get('expert_id', 0)
            success = exc_type is None
            failure_reason = str(exc_type.__name__) if exc_type else None
            
            self.metrics.record_expert_request(
                expert_id, duration, success, failure_reason
            )
        
        elif self.operation_type == 'forward':
            expert_id = self.labels.get('expert_id', 0)
            self.metrics.record_expert_forward(expert_id, duration)
