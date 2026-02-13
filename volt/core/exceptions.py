"""
Custom Exceptions for Volt-Infer Distributed System

Specialized exception hierarchy for handling distributed inference failures,
network errors, and resource constraints gracefully.
"""

from typing import Optional


class VoltInferException(Exception):
    """Base exception for all Volt-Infer errors."""
    
    def __init__(self, message: str, node_id: Optional[str] = None):
        """
        Initialize base exception.
        
        Args:
            message: Human-readable error description
            node_id: Optional identifier for the node where error occurred
        """
        self.node_id = node_id
        self.message = message
        super().__init__(self._format_message())
    
    def _format_message(self) -> str:
        """Format error message with node context."""
        if self.node_id:
            return f"[Node: {self.node_id}] {self.message}"
        return self.message


class NodeFailureException(VoltInferException):
    """
    Raised when an Expert Node becomes unreachable or crashes.
    
    This exception triggers the Router's failover logic to re-route
    tokens to backup replicas or skip the expert computation.
    """
    
    def __init__(
        self,
        node_id: str,
        expert_id: int,
        reason: str = "Node became unreachable"
    ):
        """
        Initialize node failure exception.
        
        Args:
            node_id: Network identifier of failed node (IP:Port)
            expert_id: Expert ID hosted on the failed node
            reason: Detailed failure reason (timeout, connection refused, etc.)
        """
        self.expert_id = expert_id
        self.reason = reason
        message = f"Expert {expert_id} failed: {reason}"
        super().__init__(message, node_id)


class TimeoutException(VoltInferException):
    """
    Raised when a remote operation exceeds deadline.
    
    Timeout policies:
        - P99 latency threshold: 50ms
        - Absolute deadline: 500ms
        - Retry budget: 2 attempts
    """
    
    def __init__(
        self,
        node_id: str,
        operation: str,
        timeout_ms: float,
        elapsed_ms: float
    ):
        """
        Initialize timeout exception.
        
        Args:
            node_id: Target node identifier
            operation: Name of timed-out operation (e.g., 'forward_pass')
            timeout_ms: Configured timeout threshold
            elapsed_ms: Actual elapsed time before timeout
        """
        self.operation = operation
        self.timeout_ms = timeout_ms
        self.elapsed_ms = elapsed_ms
        
        message = (
            f"Operation '{operation}' timed out after {elapsed_ms:.1f}ms "
            f"(limit: {timeout_ms:.1f}ms)"
        )
        super().__init__(message, node_id)


class ProtocolException(VoltInferException):
    """
    Raised when binary protocol parsing/validation fails.
    
    Common causes:
        - Magic number mismatch (corrupted stream)
        - Payload length exceeds buffer size
        - Unsupported protocol version
    """
    
    def __init__(
        self,
        reason: str,
        raw_data: Optional[bytes] = None,
        node_id: Optional[str] = None
    ):
        """
        Initialize protocol exception.
        
        Args:
            reason: Detailed protocol error description
            raw_data: Optional raw bytes that caused the error (for debugging)
            node_id: Source node identifier
        """
        self.raw_data = raw_data
        self.reason = reason
        
        message = f"Protocol error: {reason}"
        if raw_data:
            preview = raw_data[:32].hex()
            message += f" (data preview: {preview}...)"
        
        super().__init__(message, node_id)


class ExpertLoadException(VoltInferException):
    """
    Raised when an Expert Node cannot load model weights.
    
    This occurs during startup if:
        - Checkpoint file is missing/corrupted
        - Insufficient GPU memory
        - Quantization parameters are invalid
    """
    
    def __init__(
        self,
        expert_id: int,
        checkpoint_path: str,
        reason: str,
        node_id: Optional[str] = None
    ):
        """
        Initialize expert load exception.
        
        Args:
            expert_id: ID of expert that failed to load
            checkpoint_path: Path to checkpoint file
            reason: Failure reason
            node_id: Node identifier
        """
        self.expert_id = expert_id
        self.checkpoint_path = checkpoint_path
        self.reason = reason
        
        message = (
            f"Failed to load Expert {expert_id} from {checkpoint_path}: {reason}"
        )
        super().__init__(message, node_id)


class QueueOverflowException(VoltInferException):
    """
    Raised when Expert Node's request queue exceeds capacity.
    
    This triggers backpressure mechanisms:
        - Router reduces traffic to this node
        - Autoscaler provisions additional replicas
        - Client receives 503 Service Unavailable
    """
    
    def __init__(
        self,
        node_id: str,
        queue_depth: int,
        max_depth: int
    ):
        """
        Initialize queue overflow exception.
        
        Args:
            node_id: Overloaded node identifier
            queue_depth: Current queue size
            max_depth: Queue capacity limit
        """
        self.queue_depth = queue_depth
        self.max_depth = max_depth
        
        message = (
            f"Queue overflow: {queue_depth} requests pending "
            f"(limit: {max_depth})"
        )
        super().__init__(message, node_id)


class ConfigurationException(VoltInferException):
    """
    Raised when node configuration is invalid or incomplete.
    
    Examples:
        - Missing required fields (expert_id, port)
        - Invalid IP address format
        - Port already in use
        - Redis connection string malformed
    """
    
    def __init__(
        self,
        field: str,
        reason: str,
        node_id: Optional[str] = None
    ):
        """
        Initialize configuration exception.
        
        Args:
            field: Config field that failed validation
            reason: Validation error details
            node_id: Node identifier
        """
        self.field = field
        self.reason = reason
        
        message = f"Configuration error in '{field}': {reason}"
        super().__init__(message, node_id)


class GradientDegradationException(VoltInferException):
    """
    Raised when graceful degradation is triggered due to failures.
    
    This is NOT an error but an informational exception to signal
    that the system is operating in degraded mode (e.g., skipping
    failed experts or using cached results).
    """
    
    def __init__(
        self,
        degradation_type: str,
        details: str,
        node_id: Optional[str] = None
    ):
        """
        Initialize degradation exception.
        
        Args:
            degradation_type: Type of degradation (e.g., 'expert_skip')
            details: Explanation of degradation
            node_id: Node identifier
        """
        self.degradation_type = degradation_type
        self.details = details
        
        message = f"Degraded operation ({degradation_type}): {details}"
        super().__init__(message, node_id)


class NetworkPartitionException(VoltInferException):
    """
    Raised when Redis connection is lost, causing peer discovery failure.
    
    Impact:
        - Cannot discover new Expert Nodes
        - Existing connections continue working
        - System enters "island mode" with current topology
    """
    
    def __init__(
        self,
        redis_url: str,
        reason: str,
        node_id: Optional[str] = None
    ):
        """
        Initialize network partition exception.
        
        Args:
            redis_url: Redis connection URL
            reason: Connection failure reason
            node_id: Node identifier
        """
        self.redis_url = redis_url
        self.reason = reason
        
        message = f"Network partition: Redis at {redis_url} unreachable ({reason})"
        super().__init__(message, node_id)
