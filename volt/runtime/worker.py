"""
Worker Node for Volt-Infer

The "Muscle" of the MoE system. Hosts expert models and processes
token batches sent by Router nodes.

Responsibilities:
1. Load and manage expert model weights
2. Accept incoming TCP requests<br>3. Run expert forward pass on GPU
4. Return computed outputs
5. Track queue depth for autoscaling signals
"""

import asyncio
from typing import Optional, Dict
import logging
import time

import torch
import redis.asyncio as aioredis

from ..core.config import VoltConfig, NodeType
from ..core.protocol import ProtocolCodec, ProtocolMessage
from ..core.exceptions import ExpertLoadException, QueueOverflowException
from ..kernels.quantization import int8_dequant_kernel


logger = logging.getLogger(__name__)


class ExpertModel:
    """
    Wrapper for a single expert's weights and computation.
    
    Supports INT8 quantization for memory efficiency.
    """
    
    def __init__(
        self,
        expert_id: int,
        hidden_dim: int,
        intermediate_dim: int,
        device: str = 'cuda:0',
        use_quantization: bool = True,
    ):
        """
        Initialize expert model.
        
        Args:
            expert_id: Expert identifier
            hidden_dim: Input/output dimension
            intermediate_dim: FFN intermediate dimension (typically 4x hidden)
            device: Device for computation
            use_quantization: Use INT8 quantization
        """
        self.expert_id = expert_id
        self.hidden_dim = hidden_dim
        self.intermediate_dim = intermediate_dim
        self.device = device
        self.use_quantization = use_quantization
        
        # Expert FFN weights: two linear layers with activation
        # W1: [hidden_dim, intermediate_dim]
        # W2: [intermediate_dim, hidden_dim]
        self.w1 = None
        self.w2 = None
        
        # Quantization parameters (if enabled)
        self.w1_quantized = None
        self.w1_scales = None
        self.w2_quantized = None
        self.w2_scales = None
        
        # Performance tracking
        self.forward_count = 0
        self.total_latency_ms = 0.0
    
    def load_weights(self, checkpoint_path: str) -> None:
        """
        Load expert weights from checkpoint.
        
        Args:
            checkpoint_path: Path to checkpoint file
            
        Raises:
            ExpertLoadException: If loading fails
        """
        import os
        
        try:
            if os.path.exists(checkpoint_path):
                checkpoint = torch.load(checkpoint_path, map_location=self.device)
                self.w1 = checkpoint['w1']
                self.w2 = checkpoint['w2']
                logger.info(f"Loaded Expert {self.expert_id} from {checkpoint_path}")
            else:
                # Fallback: Initialize random weights (for testing)
                logger.warning(f"Checkpoint not found, using random weights for Expert {self.expert_id}")
                self.w1 = torch.randn(
                    self.hidden_dim,
                    self.intermediate_dim,
                    device=self.device,
                    dtype=torch.float16,
                )
                self.w2 = torch.randn(
                    self.intermediate_dim,
                    self.hidden_dim,
                    device=self.device,
                    dtype=torch.float16,
                )
            
            # Quantize if enabled
            if self.use_quantization:
                self._quantize_weights()
            
        except Exception as e:
            raise ExpertLoadException(
                expert_id=self.expert_id,
                checkpoint_path=checkpoint_path,
                reason=str(e),
            )
    
    def _quantize_weights(self) -> None:
        """Quantize weights to INT8."""
        from ..kernels.quantization import quantize_weights_int8
        
        logger.info(f"Quantizing Expert {self.expert_id} weights to INT8")
        
        # Quantize W1
        self.w1_quantized, self.w1_scales, _ = quantize_weights_int8(
            self.w1,
            per_channel=True,
            symmetric=True,
        )
        
        # Quantize W2
        self.w2_quantized, self.w2_scales, _ = quantize_weights_int8(
            self.w2,
            per_channel=True,
            symmetric=True,
        )
        
        # Free original weights to save memory
        memory_saved_mb = (
            (self.w1.numel() + self.w2.numel()) * 2  # FP16 = 2 bytes
            - (self.w1_quantized.numel() + self.w2_quantized.numel())  # INT8 = 1 byte
        ) / (1024 ** 2)
        
        logger.info(f"Expert {self.expert_id}: Saved {memory_saved_mb:.1f} MB via quantization")
        
        # Keep original weights for now (can be deleted if memory is tight)
        # del self.w1, self.w2
    
    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        """
        Expert forward pass.
        
        Implements standard FFN: output = W2 @ SiLU(W1 @ input)
        
        Args:
            hidden_states: Input tokens [batch_size, hidden_dim]
            
        Returns:
            Output tokens [batch_size, hidden_dim]
        """
        start_time = time.perf_counter()
        
        # Dequantize weights if needed
        if self.use_quantization:
            w1 = int8_dequant_kernel(
                self.w1_quantized,
                self.w1_scales,
                output_dtype=torch.float16,
            )
            w2 = int8_dequant_kernel(
                self.w2_quantized,
                self.w2_scales,
                output_dtype=torch.float16,
            )
        else:
            w1 = self.w1
            w2 = self.w2
        
        # Forward pass: FFN with SiLU activation
        # intermediate = SiLU(hidden @ W1)
        intermediate = torch.matmul(hidden_states, w1)
        intermediate = torch.nn.functional.silu(intermediate)
        
        # output = intermediate @ W2
        output = torch.matmul(intermediate, w2)
        
        # Track performance
        elapsed_ms = (time.perf_counter() - start_time) * 1000
        self.forward_count += 1
        self.total_latency_ms += elapsed_ms
        
        return output
    
    def get_metrics(self) -> Dict[str, float]:
        """Get expert performance metrics."""
        avg_latency = (
            self.total_latency_ms / self.forward_count
            if self.forward_count > 0
            else 0.0
        )
        
        return {
            'expert_id': self.expert_id,
            'forward_count': self.forward_count,
            'avg_latency_ms': avg_latency,
            'total_latency_ms': self.total_latency_ms,
        }


class WorkerNode:
    """
    Worker Node: Hosts expert models and processes requests.
    
    Architecture:
        1. Listen for TCP connections from Router
        2. Parse incoming protocol messages
        3. Queue requests (bounded queue for backpressure)
        4. Process requests on GPU
        5. Send responses back to Router
    """
    
    def __init__(self, config: VoltConfig):
        """
        Initialize worker node.
        
        Args:
            config: Node configuration
        """
        if config.node_type != NodeType.WORKER:
            raise ValueError("Config must be for WORKER node type")
        
        self.config = config
        self.worker_config = config.worker
        self.model_config = config.model
        
        # Load expert models
        self.experts: Dict[int, ExpertModel] = {}
        self._load_experts()
        
        # Request queue
        self.request_queue: asyncio.Queue = asyncio.Queue(
            maxsize=self.worker_config.queue_depth
        )
        
        # Server state
        self.server: Optional[asyncio.Server] = None
        self.running = False
        
        # Redis for registration
        self.redis: Optional[aioredis.Redis] = None
        
        # Metrics
        self.requests_processed = 0
        self.requests_failed = 0
        
        logger.info(f"Worker Node initialized: {config.node_id}")
    
    def _load_experts(self) -> None:
        """Load expert models hosted by this worker."""
        for expert_id in self.worker_config.expert_ids:
            expert = ExpertModel(
                expert_id=expert_id,
                hidden_dim=self.model_config.hidden_dim,
                intermediate_dim=self.model_config.hidden_dim * 4,  # Standard MoE
                device=self.worker_config.device,
                use_quantization=self.model_config.use_quantization,
            )
            
            # Load weights
            checkpoint_path = (
                f"{self.model_config.checkpoint_dir}/expert_{expert_id}.pt"
            )
            expert.load_weights(checkpoint_path)
            
            self.experts[expert_id] = expert
            logger.info(f"Loaded Expert {expert_id} on {self.worker_config.device}")
    
    async def start(self) -> None:
        """Start worker node server."""
        logger.info("Starting Worker Node...")
        
        # Connect to Redis for registration
        await self._connect_redis()
        
        # Register with Redis
        await self._register_with_redis()
        
        # Start TCP server
        self.server = await asyncio.start_server(
            self._handle_client,
            self.config.network.host,
            self.config.network.port,
        )
        
        self.running = True
        
        # Start request processor
        asyncio.create_task(self._process_requests())
        
        # Start heartbeat loop
        asyncio.create_task(self._heartbeat_loop())
        
        logger.info(
            f"Worker Node listening on "
            f"{self.config.network.host}:{self.config.network.port}"
        )
    
    async def _connect_redis(self) -> None:
        """Connect to Redis for registration."""
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
    
    async def _register_with_redis(self) -> None:
        """Register this worker with Redis for discovery."""
        if not self.redis:
            return
        
        import json
        
        for expert_id in self.worker_config.expert_ids:
            key = f"{self.config.redis.key_prefix}experts:{expert_id}"
            value = json.dumps({
                'host': self.config.network.host,
                'port': self.config.network.port,
                'node_id': self.config.node_id,
                'device': self.worker_config.device,
            })
            
            # Register with TTL
            await self.redis.setex(
                key,
                self.config.redis.ttl_seconds,
                value.encode(),
            )
            
            logger.info(f"Registered Expert {expert_id} with Redis")
    
    async def _heartbeat_loop(self) -> None:
        """Periodically refresh registration in Redis."""
        while self.running:
            await asyncio.sleep(self.config.redis.ttl_seconds / 2)
            await self._register_with_redis()
    
    async def _handle_client(
        self,
        reader: asyncio.StreamReader,
        writer: asyncio.StreamWriter,
    ) -> None:
        """
        Handle incoming client connection.
        
        Args:
            reader: Stream reader
            writer: Stream writer
        """
        peer_addr = writer.get_extra_info('peername')
        logger.debug(f"Connection from {peer_addr}")
        
        try:
            while True:
                # Receive request
                message = await ProtocolCodec.receive_message(reader)
                
                # Check queue depth
                if self.request_queue.qsize() >= self.worker_config.queue_depth:
                    raise QueueOverflowException(
                        node_id=self.config.node_id,
                        queue_depth=self.request_queue.qsize(),
                        max_depth=self.worker_config.queue_depth,
                    )
                
                # Queue request for processing
                await self.request_queue.put((message, writer))
        
        except asyncio.IncompleteReadError:
            logger.debug(f"Client {peer_addr} disconnected")
        
        except QueueOverflowException as e:
            logger.error(f"Queue overflow: {e}")
            # Send error response (simplified)
        
        except Exception as e:
            logger.error(f"Error handling client {peer_addr}: {e}")
        
        finally:
            writer.close()
            await writer.wait_closed()
    
    async def _process_requests(self) -> None:
        """Process requests from queue."""
        while self.running:
            try:
                # Get request from queue
                message, writer = await self.request_queue.get()
                
                # Process request
                asyncio.create_task(self._process_single_request(message, writer))
            
            except Exception as e:
                logger.error(f"Error in request processor: {e}")
    
    async def _process_single_request(
        self,
        message: ProtocolMessage,
        writer: asyncio.StreamWriter,
    ) -> None:
        """
        Process a single request.
        
        Args:
            message: Request message
            writer: Stream writer for response
        """
        try:
            expert_id = message.header.expert_id
            
            if expert_id not in self.experts:
                raise ValueError(f"Expert {expert_id} not hosted on this node")
            
            expert = self.experts[expert_id]
            
            # Run expert forward pass
            hidden_states = message.payload.to(self.worker_config.device)
            output = expert.forward(hidden_states)
            
            # Send response
            await ProtocolCodec.send_message(
                writer,
                request_id=message.header.request_id,
                expert_id=expert_id,
                payload=output,
            )
            
            self.requests_processed += 1
        
        except Exception as e:
            logger.error(f"Error processing request: {e}")
            self.requests_failed += 1
    
    def get_metrics(self) -> Dict:
        """Get worker node metrics."""
        expert_metrics = {
            f"expert_{eid}": expert.get_metrics()
            for eid, expert in self.experts.items()
        }
        
        return {
            'node_id': self.config.node_id,
            'requests_processed': self.requests_processed,
            'requests_failed': self.requests_failed,
            'queue_depth': self.request_queue.qsize(),
            'max_queue_depth': self.worker_config.queue_depth,
            'experts': expert_metrics,
        }
    
    async def shutdown(self) -> None:
        """Gracefully shutdown worker node."""
        logger.info("Shutting down Worker Node...")
        
        self.running = False
        
        if self.server:
            self.server.close()
            await self.server.wait_closed()
        
        if self.redis:
            # Deregister from Redis
            for expert_id in self.worker_config.expert_ids:
                key = f"{self.config.redis.key_prefix}experts:{expert_id}"
                await self.redis.delete(key)
            
            await self.redis.close()
        
        logger.info("Worker Node shutdown complete")
