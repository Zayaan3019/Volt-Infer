"""
Node Configuration Management for Volt-Infer

Centralized configuration for Router and Worker nodes with validation,
environment variable overrides, and Redis-based peer discovery.
"""

import os
from dataclasses import dataclass, field
from typing import Optional, List
from enum import Enum

from .exceptions import ConfigurationException


class NodeType(Enum):
    """Enumeration of node types in the distributed system."""
    ROUTER = "router"
    WORKER = "worker"


@dataclass
class NetworkConfig:
    """
    Network configuration for inter-node communication.
    
    Attributes:
        host: Bind address (0.0.0.0 for all interfaces, 127.0.0.1 for local)
        port: TCP port for data plane (10000-60000 range recommended)
        control_port: gRPC port for control plane operations
        max_connections: Maximum concurrent TCP connections
        tcp_nodelay: Disable Nagle's algorithm for low latency
        send_buffer_kb: TCP send buffer size in KB
        recv_buffer_kb: TCP receive buffer size in KB
    """
    host: str = "0.0.0.0"
    port: int = 50051
    control_port: int = 50052
    max_connections: int = 1024
    tcp_nodelay: bool = True
    send_buffer_kb: int = 256
    recv_buffer_kb: int = 256
    
    def __post_init__(self):
        """Validate network configuration."""
        if not (1024 <= self.port <= 65535):
            raise ConfigurationException(
                "port",
                f"Port {self.port} out of valid range [1024, 65535]"
            )
        
        if not (1024 <= self.control_port <= 65535):
            raise ConfigurationException(
                "control_port",
                f"Control port {self.control_port} out of valid range [1024, 65535]"
            )
        
        if self.port == self.control_port:
            raise ConfigurationException(
                "control_port",
                "Data and control ports must be different"
            )


@dataclass
class RedisConfig:
    """
    Redis configuration for peer discovery and global state.
    
    Attributes:
        url: Redis connection URL (redis://host:port/db)
        password: Optional authentication password
        db: Database number (0-15)
        key_prefix: Namespace prefix for all keys (e.g., 'voltinfer:')
        ttl_seconds: Node registration TTL (heartbeat interval)
        connection_pool_size: Max connections in pool
    """
    url: str = "redis://localhost:6379/0"
    password: Optional[str] = None
    db: int = 0
    key_prefix: str = "voltinfer:"
    ttl_seconds: int = 30
    connection_pool_size: int = 50
    
    def __post_init__(self):
        """Validate Redis configuration."""
        if not self.url.startswith(("redis://", "rediss://")):
            raise ConfigurationException(
                "url",
                f"Redis URL must start with redis:// or rediss://, got {self.url}"
            )
        
        if not (0 <= self.db <= 15):
            raise ConfigurationException(
                "db",
                f"Redis DB must be in range [0, 15], got {self.db}"
            )


@dataclass
class ModelConfig:
    """
    Model architecture configuration.
    
    Attributes:
        hidden_dim: Model hidden dimension (e.g., 4096 for Mixtral)
        num_experts: Total number of experts in the MoE model
        top_k: Number of experts to activate per token
        expert_capacity: Max tokens each expert can process per batch
        use_quantization: Enable INT8 weight quantization
        checkpoint_dir: Directory containing expert checkpoints
    """
    hidden_dim: int = 4096
    num_experts: int = 8
    top_k: int = 2
    expert_capacity: int = 256
    use_quantization: bool = True
    checkpoint_dir: str = "./checkpoints"
    
    def __post_init__(self):
        """Validate model configuration."""
        if self.top_k > self.num_experts:
            raise ConfigurationException(
                "top_k",
                f"top_k ({self.top_k}) cannot exceed num_experts ({self.num_experts})"
            )
        
        if self.hidden_dim <= 0:
            raise ConfigurationException(
                "hidden_dim",
                f"hidden_dim must be positive, got {self.hidden_dim}"
            )


@dataclass
class WorkerConfig:
    """
    Worker node specific configuration.
    
    Attributes:
        expert_ids: List of expert IDs hosted by this worker
        device: PyTorch device string (cuda:0, cuda:1, etc.)
        max_batch_size: Maximum batch size for expert computation
        queue_depth: Request queue capacity
        prefetch_enabled: Enable speculative prefetching
        fallback_expert_ids: Backup expert IDs for failover
    """
    expert_ids: List[int] = field(default_factory=list)
    device: str = "cuda:0"
    max_batch_size: int = 32
    queue_depth: int = 64
    prefetch_enabled: bool = True
    fallback_expert_ids: List[int] = field(default_factory=list)
    
    def __post_init__(self):
        """Validate worker configuration."""
        if not self.expert_ids:
            raise ConfigurationException(
                "expert_ids",
                "Worker must host at least one expert"
            )
        
        if len(self.expert_ids) != len(set(self.expert_ids)):
            raise ConfigurationException(
                "expert_ids",
                f"Duplicate expert IDs found: {self.expert_ids}"
            )


@dataclass
class RouterConfig:
    """
    Router node specific configuration.
    
    Attributes:
        router_weight_path: Path to router weight checkpoint
        timeout_ms: Per-expert request timeout in milliseconds
        retry_attempts: Number of retry attempts on failure
        load_balance_strategy: Load balancing strategy (round_robin, least_loaded)
        fallback_enabled: Enable fallback to backup experts
        p99_threshold_ms: P99 latency threshold for slow node detection
    """
    router_weight_path: str = "./checkpoints/router.pt"
    timeout_ms: float = 50.0
    retry_attempts: int = 2
    load_balance_strategy: str = "least_loaded"
    fallback_enabled: bool = True
    p99_threshold_ms: float = 50.0
    
    def __post_init__(self):
        """Validate router configuration."""
        valid_strategies = {"round_robin", "least_loaded", "random"}
        if self.load_balance_strategy not in valid_strategies:
            raise ConfigurationException(
                "load_balance_strategy",
                f"Must be one of {valid_strategies}, got {self.load_balance_strategy}"
            )


@dataclass
class ObservabilityConfig:
    """
    Monitoring and observability configuration.
    
    Attributes:
        metrics_enabled: Enable Prometheus metrics export
        metrics_port: HTTP port for /metrics endpoint
        tracing_enabled: Enable OpenTelemetry tracing
        trace_sample_rate: Fraction of requests to trace (0.0-1.0)
        log_level: Logging level (DEBUG, INFO, WARNING, ERROR)
    """
    metrics_enabled: bool = True
    metrics_port: int = 9090
    tracing_enabled: bool = False
    trace_sample_rate: float = 0.1
    log_level: str = "INFO"
    
    def __post_init__(self):
        """Validate observability configuration."""
        if not (0.0 <= self.trace_sample_rate <= 1.0):
            raise ConfigurationException(
                "trace_sample_rate",
                f"Must be in [0.0, 1.0], got {self.trace_sample_rate}"
            )
        
        valid_levels = {"DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"}
        if self.log_level.upper() not in valid_levels:
            raise ConfigurationException(
                "log_level",
                f"Must be one of {valid_levels}, got {self.log_level}"
            )


@dataclass
class VoltConfig:
    """
    Master configuration aggregating all subsystems.
    
    This is the single source of truth for node configuration.
    Supports environment variable overrides with VOLT_ prefix.
    """
    node_type: NodeType
    node_id: str
    network: NetworkConfig = field(default_factory=NetworkConfig)
    redis: RedisConfig = field(default_factory=RedisConfig)
    model: ModelConfig = field(default_factory=ModelConfig)
    worker: Optional[WorkerConfig] = None
    router: Optional[RouterConfig] = None
    observability: ObservabilityConfig = field(default_factory=ObservabilityConfig)
    
    def __post_init__(self):
        """
        Validate cross-cutting configuration constraints.
        
        Applies environment variable overrides and ensures node-type
        specific configs are present.
        """
        # Validate node-specific configs
        if self.node_type == NodeType.WORKER and self.worker is None:
            raise ConfigurationException(
                "worker",
                "WorkerConfig required for WORKER node type"
            )
        
        if self.node_type == NodeType.ROUTER and self.router is None:
            raise ConfigurationException(
                "router",
                "RouterConfig required for ROUTER node type"
            )
        
        # Apply environment overrides
        self._apply_env_overrides()
    
    def _apply_env_overrides(self):
        """
        Override config values from environment variables.
        
        Environment variables follow pattern: VOLT_<SECTION>_<KEY>
        Examples:
            VOLT_NETWORK_PORT=8080
            VOLT_REDIS_URL=redis://prod-redis:6379
            VOLT_MODEL_NUM_EXPERTS=16
        """
        # Network overrides
        if port := os.getenv("VOLT_NETWORK_PORT"):
            self.network.port = int(port)
        if host := os.getenv("VOLT_NETWORK_HOST"):
            self.network.host = host
        
        # Redis overrides
        if url := os.getenv("VOLT_REDIS_URL"):
            self.redis.url = url
        if password := os.getenv("VOLT_REDIS_PASSWORD"):
            self.redis.password = password
        
        # Model overrides
        if hidden_dim := os.getenv("VOLT_MODEL_HIDDEN_DIM"):
            self.model.hidden_dim = int(hidden_dim)
        if num_experts := os.getenv("VOLT_MODEL_NUM_EXPERTS"):
            self.model.num_experts = int(num_experts)
        
        # Observability overrides
        if log_level := os.getenv("VOLT_LOG_LEVEL"):
            self.observability.log_level = log_level.upper()
    
    @classmethod
    def from_env(cls, node_type: NodeType, node_id: str) -> "VoltConfig":
        """
        Factory method to create config primarily from environment variables.
        
        Args:
            node_type: Type of node (ROUTER or WORKER)
            node_id: Unique node identifier
            
        Returns:
            Fully configured VoltConfig instance
            
        Example:
            >>> import os
            >>> os.environ['VOLT_NETWORK_PORT'] = '8080'
            >>> config = VoltConfig.from_env(NodeType.ROUTER, 'router-1')
            >>> print(config.network.port)
            8080
        """
        config = cls(node_type=node_type, node_id=node_id)
        
        # Create node-specific configs based on type
        if node_type == NodeType.WORKER:
            expert_ids_str = os.getenv("VOLT_WORKER_EXPERT_IDS", "0")
            expert_ids = [int(x.strip()) for x in expert_ids_str.split(",")]
            config.worker = WorkerConfig(expert_ids=expert_ids)
        elif node_type == NodeType.ROUTER:
            config.router = RouterConfig()
        
        return config
    
    def to_dict(self) -> dict:
        """
        Serialize configuration to dictionary.
        
        Useful for logging, debugging, and configuration file generation.
        """
        result = {
            "node_type": self.node_type.value,
            "node_id": self.node_id,
            "network": {
                "host": self.network.host,
                "port": self.network.port,
                "control_port": self.network.control_port,
            },
            "redis": {
                "url": self.redis.url,
                "db": self.redis.db,
                "key_prefix": self.redis.key_prefix,
            },
            "model": {
                "hidden_dim": self.model.hidden_dim,
                "num_experts": self.model.num_experts,
                "top_k": self.model.top_k,
            },
        }
        
        if self.worker:
            result["worker"] = {
                "expert_ids": self.worker.expert_ids,
                "device": self.worker.device,
                "max_batch_size": self.worker.max_batch_size,
            }
        
        if self.router:
            result["router"] = {
                "timeout_ms": self.router.timeout_ms,
                "retry_attempts": self.router.retry_attempts,
                "load_balance_strategy": self.router.load_balance_strategy,
            }
        
        return result
