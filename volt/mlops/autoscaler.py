"""
Autoscaler for Volt-Infer Expert Workers

Monitors queue depth and expert utilization metrics to automatically
provision or de-provision expert worker nodes. Integrates with container
orchestration platforms (Docker, Kubernetes) via webhooks.

Scaling Policy:
    Scale Up When:
        - Queue depth > threshold for sustained period
        - Expert P99 latency > SLA threshold
        - Request rate increasing beyond capacity
    
    Scale Down When:
        - Queue depth near zero for sustained period
        - Multiple replicas with low utilization
        - Request rate decreased significantly
"""

import asyncio
from typing import Dict, List, Optional, Callable
from dataclasses import dataclass
from enum import Enum
import time
import logging
import json

import httpx

from .metrics import VoltMetrics


logger = logging.getLogger(__name__)


class ScalingAction(Enum):
    """Autoscaling action types."""
    SCALE_UP = "scale_up"
    SCALE_DOWN = "scale_down"
    NO_ACTION = "no_action"


@dataclass
class ScalingDecision:
    """
    Result of autoscaling evaluation.
    
    Attributes:
        action: Scaling action to take
        expert_id: Target expert ID
        target_replicas: Desired number of replicas
        reason: Human-readable explanation
        confidence: Confidence score (0.0-1.0)
    """
    action: ScalingAction
    expert_id: int
    target_replicas: int
    reason: str
    confidence: float


@dataclass
class AutoscalerConfig:
    """
    Autoscaler configuration.
    
    Attributes:
        queue_depth_threshold: Queue depth trigger for scale-up
        queue_depth_window_seconds: Observation window duration
        p99_latency_threshold_ms: Latency SLA threshold
        min_replicas: Minimum replicas per expert
        max_replicas: Maximum replicas per expert
        scale_up_cooldown_seconds: Cooldown after scale-up
        scale_down_cooldown_seconds: Cooldown after scale-down
        webhook_url: Webhook URL for scaling actions
        enabled: Enable/disable autoscaling
    """
    queue_depth_threshold: int = 16
    queue_depth_window_seconds: float = 30.0
    p99_latency_threshold_ms: float = 50.0
    min_replicas: int = 1
    max_replicas: int = 8
    scale_up_cooldown_seconds: float = 60.0
    scale_down_cooldown_seconds: float = 300.0
    webhook_url: Optional[str] = None
    enabled: bool = True


class ExpertMetricsTracker:
    """
    Tracks time-series metrics for a single expert.
    
    Maintains sliding windows of queue depth, latency, and request rate
    for scaling decisions.
    """
    
    def __init__(self, expert_id: int, window_size: int = 100):
        """
        Initialize metrics tracker.
        
        Args:
            expert_id: Expert identifier
            window_size: Size of sliding window for metrics
        """
        self.expert_id = expert_id
        self.window_size = window_size
        
        # Time-series data
        self.queue_depths: List[tuple] = []  # [(timestamp, depth), ...]
        self.latencies: List[tuple] = []  # [(timestamp, latency_ms), ...]
        self.request_counts: List[tuple] = []  # [(timestamp, count), ...]
        
        # Current replica count
        self.current_replicas = 1
        
        # Last scaling action
        self.last_scale_up_time = 0.0
        self.last_scale_down_time = 0.0
    
    def record_queue_depth(self, depth: int) -> None:
        """Record queue depth observation."""
        self.queue_depths.append((time.time(), depth))
        
        # Trim to window size
        if len(self.queue_depths) > self.window_size:
            self.queue_depths.pop(0)
    
    def record_latency(self, latency_ms: float) -> None:
        """Record latency observation."""
        self.latencies.append((time.time(), latency_ms))
        
        if len(self.latencies) > self.window_size:
            self.latencies.pop(0)
    
    def record_request(self) -> None:
        """Record a request."""
        current_time = time.time()
        
        # Increment count for current time bucket (1-second buckets)
        if self.request_counts and (current_time - self.request_counts[-1][0] < 1.0):
            # Same time bucket
            timestamp, count = self.request_counts[-1]
            self.request_counts[-1] = (timestamp, count + 1)
        else:
            # New time bucket
            self.request_counts.append((current_time, 1))
        
        if len(self.request_counts) > self.window_size:
            self.request_counts.pop(0)
    
    def get_avg_queue_depth(self, window_seconds: float) -> float:
        """
        Get average queue depth over recent window.
        
        Args:
            window_seconds: Time window in seconds
            
        Returns:
            Average queue depth
        """
        if not self.queue_depths:
            return 0.0
        
        cutoff_time = time.time() - window_seconds
        recent = [depth for ts, depth in self.queue_depths if ts >= cutoff_time]
        
        if not recent:
            return 0.0
        
        return sum(recent) / len(recent)
    
    def get_p99_latency(self) -> float:
        """
        Get P99 latency from recent observations.
        
        Returns:
            P99 latency in milliseconds
        """
        if not self.latencies:
            return 0.0
        
        sorted_latencies = sorted([lat for _, lat in self.latencies])
        p99_idx = int(len(sorted_latencies) * 0.99)
        
        return sorted_latencies[p99_idx] if sorted_latencies else 0.0
    
    def get_request_rate(self) -> float:
        """
        Get current request rate (requests per second).
        
        Returns:
            Requests per second
        """
        if not self.request_counts:
            return 0.0
        
        # Sum requests over last 10 seconds
        cutoff_time = time.time() - 10.0
        recent_requests = sum(
            count for ts, count in self.request_counts if ts >= cutoff_time
        )
        
        return recent_requests / 10.0


class Autoscaler:
    """
    Autoscaler for expert worker nodes.
    
    Monitors metrics and makes scaling decisions based on configured policy.
    """
    
    def __init__(
        self,
        config: AutoscalerConfig,
        metrics: Optional[VoltMetrics] = None,
    ):
        """
        Initialize autoscaler.
        
        Args:
            config: Autoscaler configuration
            metrics: Optional metrics instance
        """
        self.config = config
        self.metrics = metrics
        
        # Per-expert trackers
        self.expert_trackers: Dict[int, ExpertMetricsTracker] = {}
        
        # HTTP client for webhook calls
        self.http_client = httpx.AsyncClient(timeout=10.0)
        
        # Scaling history
        self.scaling_history: List[ScalingDecision] = []
        
        # Background task
        self.monitor_task: Optional[asyncio.Task] = None
        self.running = False
        
        logger.info("Autoscaler initialized")
    
    def get_or_create_tracker(self, expert_id: int) -> ExpertMetricsTracker:
        """
        Get or create metrics tracker for expert.
        
        Args:
            expert_id: Expert identifier
            
        Returns:
            ExpertMetricsTracker instance
        """
        if expert_id not in self.expert_trackers:
            self.expert_trackers[expert_id] = ExpertMetricsTracker(expert_id)
        
        return self.expert_trackers[expert_id]
    
    def record_queue_depth(self, expert_id: int, depth: int) -> None:
        """Record queue depth for expert."""
        tracker = self.get_or_create_tracker(expert_id)
        tracker.record_queue_depth(depth)
    
    def record_latency(self, expert_id: int, latency_ms: float) -> None:
        """Record latency for expert."""
        tracker = self.get_or_create_tracker(expert_id)
        tracker.record_latency(latency_ms)
    
    def record_request(self, expert_id: int) -> None:
        """Record request for expert."""
        tracker = self.get_or_create_tracker(expert_id)
        tracker.record_request()
    
    async def start(self) -> None:
        """Start autoscaler monitoring loop."""
        if not self.config.enabled:
            logger.info("Autoscaler disabled")
            return
        
        self.running = True
        self.monitor_task = asyncio.create_task(self._monitor_loop())
        
        logger.info("Autoscaler started")
    
    async def stop(self) -> None:
        """Stop autoscaler."""
        self.running = False
        
        if self.monitor_task:
            self.monitor_task.cancel()
            try:
                await self.monitor_task
            except asyncio.CancelledError:
                pass
        
        await self.http_client.aclose()
        
        logger.info("Autoscaler stopped")
    
    async def _monitor_loop(self) -> None:
        """Main monitoring loop."""
        while self.running:
            try:
                # Evaluate scaling decisions for all experts
                decisions = await self._evaluate_all_experts()
                
                # Execute scaling actions
                for decision in decisions:
                    if decision.action != ScalingAction.NO_ACTION:
                        await self._execute_scaling(decision)
                
                # Sleep until next evaluation
                await asyncio.sleep(10.0)
            
            except Exception as e:
                logger.error(f"Error in autoscaler monitor loop: {e}")
                await asyncio.sleep(10.0)
    
    async def _evaluate_all_experts(self) -> List[ScalingDecision]:
        """
        Evaluate scaling decisions for all experts.
        
        Returns:
            List of scaling decisions
        """
        decisions = []
        
        for expert_id, tracker in self.expert_trackers.items():
            decision = await self._evaluate_expert(expert_id, tracker)
            decisions.append(decision)
        
        return decisions
    
    async def _evaluate_expert(
        self,
        expert_id: int,
        tracker: ExpertMetricsTracker,
    ) -> ScalingDecision:
        """
        Evaluate scaling decision for a single expert.
        
        Args:
            expert_id: Expert identifier
            tracker: Metrics tracker for this expert
            
        Returns:
            Scaling decision
        """
        # Get current metrics
        avg_queue_depth = tracker.get_avg_queue_depth(
            self.config.queue_depth_window_seconds
        )
        p99_latency = tracker.get_p99_latency()
        request_rate = tracker.get_request_rate()
        
        current_replicas = tracker.current_replicas
        
        # Check cooldown periods
        time_since_scale_up = time.time() - tracker.last_scale_up_time
        time_since_scale_down = time.time() - tracker.last_scale_down_time
        
        # === Scale Up Logic ===
        
        if (
            avg_queue_depth > self.config.queue_depth_threshold
            and time_since_scale_up > self.config.scale_up_cooldown_seconds
            and current_replicas < self.config.max_replicas
        ):
            return ScalingDecision(
                action=ScalingAction.SCALE_UP,
                expert_id=expert_id,
                target_replicas=current_replicas + 1,
                reason=f"Queue depth {avg_queue_depth:.1f} > threshold {self.config.queue_depth_threshold}",
                confidence=0.9,
            )
        
        if (
            p99_latency > self.config.p99_latency_threshold_ms
            and time_since_scale_up > self.config.scale_up_cooldown_seconds
            and current_replicas < self.config.max_replicas
        ):
            return ScalingDecision(
                action=ScalingAction.SCALE_UP,
                expert_id=expert_id,
                target_replicas=current_replicas + 1,
                reason=f"P99 latency {p99_latency:.1f}ms > threshold {self.config.p99_latency_threshold_ms}ms",
                confidence=0.85,
            )
        
        # === Scale Down Logic ===
        
        if (
            avg_queue_depth < self.config.queue_depth_threshold / 4
            and time_since_scale_down > self.config.scale_down_cooldown_seconds
            and current_replicas > self.config.min_replicas
            and request_rate < 1.0  # Very low traffic
        ):
            return ScalingDecision(
                action=ScalingAction.SCALE_DOWN,
                expert_id=expert_id,
                target_replicas=current_replicas - 1,
                reason=f"Queue depth {avg_queue_depth:.1f} consistently low, request rate {request_rate:.2f} req/s",
                confidence=0.7,
            )
        
        # === No Action ===
        
        return ScalingDecision(
            action=ScalingAction.NO_ACTION,
            expert_id=expert_id,
            target_replicas=current_replicas,
            reason="Metrics within acceptable range",
            confidence=1.0,
        )
    
    async def _execute_scaling(self, decision: ScalingDecision) -> None:
        """
        Execute a scaling decision.
        
        Args:
            decision: Scaling decision to execute
        """
        logger.info(
            f"Executing scaling action: {decision.action.value} "
            f"for Expert {decision.expert_id} "
            f"({decision.target_replicas} replicas) - {decision.reason}"
        )
        
        # Update tracker
        tracker = self.expert_trackers[decision.expert_id]
        tracker.current_replicas = decision.target_replicas
        
        if decision.action == ScalingAction.SCALE_UP:
            tracker.last_scale_up_time = time.time()
        elif decision.action == ScalingAction.SCALE_DOWN:
            tracker.last_scale_down_time = time.time()
        
        # Add to history
        self.scaling_history.append(decision)
        
        # Trigger webhook if configured
        if self.config.webhook_url:
            await self._trigger_webhook(decision)
    
    async def _trigger_webhook(self, decision: ScalingDecision) -> None:
        """
        Trigger webhook for scaling action.
        
        Args:
            decision: Scaling decision
        """
        try:
            payload = {
                'action': decision.action.value,
                'expert_id': decision.expert_id,
                'target_replicas': decision.target_replicas,
                'reason': decision.reason,
                'confidence': decision.confidence,
                'timestamp': time.time(),
            }
            
            response = await self.http_client.post(
                self.config.webhook_url,
                json=payload,
                headers={'Content-Type': 'application/json'},
            )
            
            if response.status_code == 200:
                logger.info(f"Webhook triggered successfully: {decision.action.value}")
            else:
                logger.error(
                    f"Webhook failed with status {response.status_code}: "
                    f"{response.text}"
                )
        
        except Exception as e:
            logger.error(f"Failed to trigger webhook: {e}")
    
    def get_scaling_stats(self) -> Dict:
        """
        Get autoscaling statistics.
        
        Returns:
            Dictionary with scaling metrics
        """
        total_decisions = len(self.scaling_history)
        scale_ups = sum(
            1 for d in self.scaling_history if d.action == ScalingAction.SCALE_UP
        )
        scale_downs = sum(
            1 for d in self.scaling_history if d.action == ScalingAction.SCALE_DOWN
        )
        
        return {
            'total_decisions': total_decisions,
            'scale_ups': scale_ups,
            'scale_downs': scale_downs,
            'current_replicas': {
                expert_id: tracker.current_replicas
                for expert_id, tracker in self.expert_trackers.items()
            },
            'recent_decisions': [
                {
                    'action': d.action.value,
                    'expert_id': d.expert_id,
                    'replicas': d.target_replicas,
                    'reason': d.reason,
                }
                for d in self.scaling_history[-10:]  # Last 10 decisions
            ],
        }
