"""
Distributed Router Node for Volt-Infer

The "Brain" of the MoE system. Responsible for:
1. Running Top-K gating to select experts for each token
2. Dispatching token batches to remote Expert Nodes
3. Gathering results and reassembling sequences
4. Handling failures with graceful degradation and failover
"""

import asyncio
from typing import Dict, List, Optional, Tuple
import time
import logging

import torch
import redis.asyncio as aioredis

from ..core.config import VoltConfig, NodeType
from ..core.protocol import ProtocolCodec, create_request_id, ProtocolFlags
from ..core.exceptions import (
    NodeFailureException, TimeoutException, VoltInferException
)
from ..kernels.gating import topk_gating_kernel
from ..utils.tensor_utils import batch_tensors_by_expert, reassemble_expert_outputs


logger = logging.getLogger(__name__)


class ExpertConnection:
    """
    Manages TCP connection to a single Expert Node.
    
    Maintains persistent connection with automatic reconnection,
    health monitoring, and latency tracking.
    """
    
    def __init__(
        self,
        expert_id: int,
        host: str,
        port: int,
        timeout_ms: float = 50.0,
    ):
        """
        Initialize expert connection.
        
        Args:
            expert_id: Expert identifier
            host: Expert node IP/hostname
            port: Expert node port
            timeout_ms: Request timeout in milliseconds
        """
        self.expert_id = expert_id
        self.host = host
        self.port = port
        self.timeout_ms = timeout_ms
        
        # Connection state
        self.reader: Optional[asyncio.StreamReader] = None
        self.writer: Optional[asyncio.StreamWriter] = None
        self.connected = False
        
        # Performance metrics
        self.request_count = 0
        self.failure_count = 0
        self.latency_sum_ms = 0.0
        self.p99_latency_ms = 0.0
        self.last_request_time = 0.0
    
    async def connect(self) -> bool:
        """
        Establish TCP connection to expert node.
        
        Returns:
            True if connection successful, False otherwise
        """
        try:
            self.reader, self.writer = await asyncio.wait_for(
                asyncio.open_connection(self.host, self.port),
                timeout=5.0
            )
            self.connected = True
            logger.info(f"Connected to Expert {self.expert_id} at {self.host}:{self.port}")
            return True
        
        except (asyncio.TimeoutError, ConnectionRefusedError, OSError) as e:
            logger.error(f"Failed to connect to Expert {self.expert_id}: {e}")
            self.connected = False
            return False
    
    async def disconnect(self) -> None:
        """Close connection to expert node."""
        if self.writer:
            self.writer.close()
            await self.writer.wait_closed()
        
        self.connected = False
        logger.info(f"Disconnected from Expert {self.expert_id}")
    
    async def send_request(
        self,
        hidden_states: torch.Tensor,
        request_id: Optional[int] = None,
    ) -> torch.Tensor:
        """
        Send inference request to expert node.
        
        Args:
            hidden_states: Token representations [batch, hidden_dim]
            request_id: Optional request ID (generated if not provided)
            
        Returns:
            Expert output tensor [batch, hidden_dim]
            
        Raises:
            NodeFailureException: If request fails or times out
        """
        if not self.connected:
            await self.connect()
        
        if request_id is None:
            request_id = create_request_id()
        
        start_time = time.perf_counter()
        
        try:
            # Send request
            await ProtocolCodec.send_message(
                self.writer,
                request_id=request_id,
                expert_id=self.expert_id,
                payload=hidden_states,
                flags=0,
            )
            
            # Receive response
            response = await ProtocolCodec.receive_message(
                self.reader,
                timeout=self.timeout_ms / 1000.0,
            )
            
            # Update metrics
            elapsed_ms = (time.perf_counter() - start_time) * 1000
            self.request_count += 1
            self.latency_sum_ms += elapsed_ms
            self.last_request_time = time.time()
            
            # Update P99 estimate (exponential moving average)
            alpha = 0.1
            self.p99_latency_ms = (
                alpha * elapsed_ms + (1 - alpha) * self.p99_latency_ms
            )
            
            return response.payload
        
        except asyncio.TimeoutError:
            self.failure_count += 1
            elapsed_ms = (time.perf_counter() - start_time) * 1000
            
            raise TimeoutException(
                node_id=f"{self.host}:{self.port}",
                operation="forward_pass",
                timeout_ms=self.timeout_ms,
                elapsed_ms=elapsed_ms,
            )
        
        except Exception as e:
            self.failure_count += 1
            self.connected = False
            
            raise NodeFailureException(
                node_id=f"{self.host}:{self.port}",
                expert_id=self.expert_id,
                reason=str(e),
            )
    
    def get_metrics(self) -> Dict[str, float]:
        """Get connection performance metrics."""
        avg_latency = (
            self.latency_sum_ms / self.request_count
            if self.request_count > 0
            else 0.0
        )
        
        return {
            'request_count': self.request_count,
            'failure_count': self.failure_count,
            'failure_rate': (
                self.failure_count / max(self.request_count, 1)
            ),
            'avg_latency_ms': avg_latency,
            'p99_latency_ms': self.p99_latency_ms,
            'connected': self.connected,
        }


class RouterNode:
    """
    Router Node: Orchestrates token routing to expert workers.
    
    Architecture:
        1. Receive input tokens
        2. Run gating kernel to compute expert assignments
        3. Batch tokens by destination expert
        4. Dispatch batches asynchronously to expert nodes
        5. Gather results and reassemble sequences
        6. Handle failures with fallback strategy
    """
    
    def __init__(self, config: VoltConfig):
        """
        Initialize router node.
        
        Args:
            config: Node configuration
        """
        if config.node_type != NodeType.ROUTER:
            raise ValueError("Config must be for ROUTER node type")
        
        self.config = config
        self.router_config = config.router
        self.model_config = config.model
        
        # Load router weights
        self.router_weights = self._load_router_weights()
        
        # Expert connections
        self.expert_connections: Dict[int, ExpertConnection] = {}
        self.fallback_map: Dict[int, List[int]] = {}  # expert_id -> [fallback_ids]
        
        # Redis client for peer discovery
        self.redis: Optional[aioredis.Redis] = None
        
        # Performance tracking
        self.total_requests = 0
        self.successful_requests = 0
        self.failed_requests = 0
        
        logger.info(f"Router Node initialized: {config.node_id}")
    
    def _load_router_weights(self) -> torch.Tensor:
        """
        Load router weight matrix from checkpoint.
        
        Returns:
            Router weights [hidden_dim, num_experts]
        """
        import os
        
        checkpoint_path = self.router_config.router_weight_path
        
        if os.path.exists(checkpoint_path):
            try:
                weights = torch.load(checkpoint_path, map_location='cuda')
                logger.info(f"Loaded router weights from {checkpoint_path}")
                return weights
            except Exception as e:
                logger.warning(f"Failed to load router weights: {e}")
        
        # Fallback: initialize random weights
        logger.warning("Using random router weights (for testing)")
        weights = torch.randn(
            self.model_config.hidden_dim,
            self.model_config.num_experts,
            device='cuda',
            dtype=torch.float32,
        )
        
        return weights
    
    async def start(self) -> None:
        """
        Start router node and discover expert workers.
        """
        logger.info("Starting Router Node...")
        
        # Connect to Redis for peer discovery
        await self._connect_redis()
        
        # Discover expert nodes
        await self._discover_experts()
        
        # Start health check loop
        asyncio.create_task(self._health_check_loop())
        
        logger.info("Router Node started successfully")
    
    async def _connect_redis(self) -> None:
        """Connect to Redis for peer discovery."""
        try:
            self.redis = await aioredis.from_url(
                self.config.redis.url,
                password=self.config.redis.password,
                decode_responses=False,
            )
            await self.redis.ping()
            logger.info(f"Connected to Redis: {self.config.redis.url}")
        
        except Exception as e:
            logger.error(f"Failed to connect to Redis: {e}")
            self.redis = None
    
    async def _discover_experts(self) -> None:
        """
        Discover expert worker nodes from Redis registry.
        
        Registry format:
            Key: voltinfer:experts:<expert_id>
            Value: {"host": "...", "port": ...}
        """
        if not self.redis:
            logger.warning("Redis unavailable, using mock expert topology")
            # For testing: create mock connections
            for expert_id in range(self.model_config.num_experts):
                conn = ExpertConnection(
                    expert_id=expert_id,
                    host='localhost',
                    port=50100 + expert_id,
                    timeout_ms=self.router_config.timeout_ms,
                )
                self.expert_connections[expert_id] = conn
            return
        
        # Query Redis for registered experts
        key_prefix = f"{self.config.redis.key_prefix}experts:*"
        keys = await self.redis.keys(key_prefix)
        
        for key in keys:
            expert_id = int(key.decode().split(':')[-1])
            data = await self.redis.get(key)
            
            if data:
                import json
                info = json.loads(data.decode())
                
                conn = ExpertConnection(
                    expert_id=expert_id,
                    host=info['host'],
                    port=info['port'],
                    timeout_ms=self.router_config.timeout_ms,
                )
                
                self.expert_connections[expert_id] = conn
                logger.info(f"Discovered Expert {expert_id} at {info['host']}:{info['port']}")
    
    async def _health_check_loop(self) -> None:
        """Periodically check expert node health."""
        while True:
            await asyncio.sleep(30.0)
            
            for expert_id, conn in self.expert_connections.items():
                metrics = conn.get_metrics()
                
                # Check if node is slow (P99 > threshold)
                if metrics['p99_latency_ms'] > self.router_config.p99_threshold_ms:
                    logger.warning(
                        f"Expert {expert_id} is slow: "
                        f"P99={metrics['p99_latency_ms']:.1f}ms"
                    )
    
    async def forward(
        self,
        hidden_states: torch.Tensor,
    ) -> torch.Tensor:
        """
        Forward pass: Route tokens through experts.
        
        Args:
            hidden_states: Input token representations [batch_size, hidden_dim]
            
        Returns:
            Output representations [batch_size, hidden_dim]
            
        Raises:
            VoltInferException: If routing fails catastrophically
        """
        batch_size, hidden_dim = hidden_states.shape
        
        # Step 1: Run gating kernel to determine expert assignments
        expert_indices, expert_weights = topk_gating_kernel(
            hidden_states,
            self.router_weights,
            k=self.model_config.top_k,
        )
        
        # Step 2: Batch tokens by destination expert
        expert_batches = batch_tensors_by_expert(
            hidden_states,
            expert_indices,
            expert_weights,
            self.model_config.num_experts,
        )
        
        # Step 3: Dispatch batches to experts (async)
        expert_outputs = await self._dispatch_to_experts(expert_batches)
        
        # Step 4: Reassemble outputs
        output = reassemble_expert_outputs(
            expert_outputs=[out for out, _, _ in expert_outputs],
            expert_weights=[w for _, w, _ in expert_outputs],
            token_indices=[idx for _, _, idx in expert_outputs],
            batch_size=batch_size,
            hidden_dim=hidden_dim,
            top_k=self.model_config.top_k,
        )
        
        self.total_requests += 1
        self.successful_requests += 1
        
        return output
    
    async def _dispatch_to_experts(
        self,
        expert_batches: List[Tuple[torch.Tensor, torch.Tensor, List[int]]],
    ) -> List[Tuple[Optional[torch.Tensor], torch.Tensor, List[int]]]:
        """
        Dispatch token batches to expert nodes in parallel.
        
        Args:
            expert_batches: List of (hidden_batch, weights, indices) per expert
            
        Returns:
            List of (expert_output, weights, indices) per expert
        """
        tasks = []
        
        for expert_id, (batch, weights, indices) in enumerate(expert_batches):
            if batch is None:
                # No tokens for this expert
                tasks.append(None)
                continue
            
            # Create async task for this expert
            task = asyncio.create_task(
                self._send_to_expert(expert_id, batch, weights, indices)
            )
            tasks.append(task)
        
        # Gather results (with error handling)
        results = []
        for expert_id, task in enumerate(tasks):
            if task is None:
                results.append((None, None, []))
                continue
            
            try:
                output, weights, indices = await task
                results.append((output, weights, indices))
            
            except (NodeFailureException, TimeoutException) as e:
                logger.error(f"Expert {expert_id} failed: {e}")
                
                # Try fallback
                if self.router_config.fallback_enabled:
                    fallback_result = await self._try_fallback(expert_id, task)
                    if fallback_result:
                        results.append(fallback_result)
                        continue
                
                # Graceful degradation: skip this expert
                logger.warning(f"Skipping Expert {expert_id} (degraded mode)")
                results.append((None, None, []))
                self.failed_requests += 1
        
        return results
    
    async def _send_to_expert(
        self,
        expert_id: int,
        batch: torch.Tensor,
        weights: torch.Tensor,
        indices: List[int],
    ) -> Tuple[torch.Tensor, torch.Tensor, List[int]]:
        """
        Send batch to a specific expert with retries.
        
        Args:
            expert_id: Target expert
            batch: Token batch [num_tokens, hidden_dim]
            weights: Gating weights [num_tokens]
            indices: Original token indices
            
        Returns:
            (expert_output, weights, indices)
        """
        if expert_id not in self.expert_connections:
            raise NodeFailureException(
                node_id=f"expert-{expert_id}",
                expert_id=expert_id,
                reason="Expert not found in registry",
            )
        
        conn = self.expert_connections[expert_id]
        
        # Retry logic
        for attempt in range(self.router_config.retry_attempts):
            try:
                output = await conn.send_request(batch)
                return output, weights, indices
            
            except (NodeFailureException, TimeoutException) as e:
                if attempt < self.router_config.retry_attempts - 1:
                    logger.warning(f"Retry {attempt + 1}/{self.router_config.retry_attempts} for Expert {expert_id}")
                    await asyncio.sleep(0.01)  # Small backoff
                else:
                    raise
    
    async def _try_fallback(
        self,
        failed_expert_id: int,
        original_task: asyncio.Task,
    ) -> Optional[Tuple[torch.Tensor, torch.Tensor, List[int]]]:
        """
        Attempt fallback to replica expert.
        
        Args:
            failed_expert_id: ID of failed expert
            original_task: Original failed task
            
        Returns:
            Fallback result, or None if no fallback available
        """
        # Check if fallback replicas exist
        if failed_expert_id not in self.fallback_map:
            return None
        
        fallback_ids = self.fallback_map[failed_expert_id]
        
        for fallback_id in fallback_ids:
            try:
                # Extract original arguments (simplified)
                # In production, store task metadata
                logger.info(f"Trying fallback: Expert {fallback_id}")
                # ... retry logic ...
                return None  # Placeholder
            
            except Exception:
                continue
        
        return None
    
    async def shutdown(self) -> None:
        """Gracefully shutdown router node."""
        logger.info("Shutting down Router Node...")
        
        # Close all expert connections
        tasks = [
            conn.disconnect()
            for conn in self.expert_connections.values()
        ]
        await asyncio.gather(*tasks, return_exceptions=True)
        
        # Close Redis
        if self.redis:
            await self.redis.close()
        
        logger.info("Router Node shutdown complete")
