"""
Speculative Prefetching Scheduler for Volt-Infer

Implements latency masking through predictive expert activation.
Uses N-gram patterns to predict which experts will be needed for
upcoming tokens and sends "warmup signals" preemptively.

Algorithm:
    Given token sequence [t₀, t₁, t₂, ...]:
    1. Observe historical pattern: token tᵢ → experts [E₃, E₇]
    2. While processing tᵢ, predict that tᵢ₊₁ will need experts [E₃, E₇]
    3. Send prefetch request to E₃, E₇ to warm their caches
    4. By the time tᵢ₊₁ arrives, experts are ready (reduced latency)
"""

import asyncio
from typing import Dict, List, Tuple, Optional, Set
from collections import defaultdict, deque
import logging

import torch

logger = logging.getLogger(__name__)


class NGramPredictor:
    """
    N-gram based expert activation predictor.
    
    Tracks patterns like:
        - Token at position i activates experts [E₁, E₄]
        - Token at position i+1 typically activates experts [E₂, E₅]
        - Bigram (Eᵢ, Eⱼ) → Next expert Eₖ
    """
    
    def __init__(self, n: int = 2, num_experts: int = 8):
        """
        Initialize N-gram predictor.
        
        Args:
            n: N-gram order (2 = bigram, 3 = trigram)
            num_experts: Total number of experts in system
        """
        self.n = n
        self.num_experts = num_experts
        
        # History: deque of recent expert activations
        # Format: [(token_idx, [expert_ids]), ...]
        self.history: deque = deque(maxlen=1000)
        
        # N-gram frequency table
        # Key: tuple of (n-1) expert IDs
        # Value: Counter of next expert ID frequencies
        self.ngram_counts: Dict[Tuple[int, ...], Dict[int, int]] = defaultdict(
            lambda: defaultdict(int)
        )
        
        # Total observations per context
        self.context_totals: Dict[Tuple[int, ...], int] = defaultdict(int)
        
        # Prediction accuracy tracking
        self.predictions_made = 0
        self.predictions_correct = 0
    
    def observe(self, token_idx: int, expert_ids: List[int]) -> None:
        """
        Observe token-expert activation pattern.
        
        Args:
            token_idx: Token sequence position
            expert_ids: List of experts activated for this token
        """
        # Add to history
        self.history.append((token_idx, expert_ids))
        
        # Update N-gram counts
        if len(self.history) >= self.n:
            # Get last n-1 activations as context
            context_activations = list(self.history)[-self.n:]
            
            # Build context from first n-1 tokens
            context = tuple(
                sorted(experts)
                for _, experts in context_activations[:-1]
            )
            
            # Current token's experts are the "next" in the pattern
            for expert_id in expert_ids:
                # Flatten context for key
                flat_context = tuple(
                    exp for experts_list in context for exp in experts_list
                )
                
                self.ngram_counts[flat_context][expert_id] += 1
                self.context_totals[flat_context] += 1
    
    def predict(
        self,
        recent_experts: List[List[int]],
        top_k: int = 2,
        threshold: float = 0.1,
    ) -> List[int]:
        """
        Predict next expert activations based on recent history.
        
        Args:
            recent_experts: List of expert IDs from last n-1 tokens
            top_k: Number of experts to predict
            threshold: Minimum probability threshold for prediction
            
        Returns:
            List of predicted expert IDs
        """
        if len(recent_experts) < self.n - 1:
            # Not enough history
            return []
        
        # Build context from recent activations
        context = tuple(
            sorted(experts) for experts in recent_experts[-(self.n - 1):]
        )
        flat_context = tuple(
            exp for experts_list in context for exp in experts_list
        )
        
        # Look up N-gram predictions
        if flat_context not in self.ngram_counts:
            return []
        
        predictions = self.ngram_counts[flat_context]
        total = self.context_totals[flat_context]
        
        # Compute probabilities and filter by threshold
        candidates = []
        for expert_id, count in predictions.items():
            probability = count / total
            if probability >= threshold:
                candidates.append((expert_id, probability))
        
        # Sort by probability and take top-k
        candidates.sort(key=lambda x: x[1], reverse=True)
        predicted_experts = [exp_id for exp_id, _ in candidates[:top_k]]
        
        self.predictions_made += 1
        
        return predicted_experts
    
    def evaluate_prediction(
        self,
        predicted: List[int],
        actual: List[int],
    ) -> float:
        """
        Evaluate prediction accuracy.
        
        Args:
            predicted: Predicted expert IDs
            actual: Actual expert IDs activated
            
        Returns:
            Accuracy score (0.0 to 1.0)
        """
        if not predicted:
            return 0.0
        
        predicted_set = set(predicted)
        actual_set = set(actual)
        
        # Compute overlap
        overlap = len(predicted_set & actual_set)
        accuracy = overlap / len(predicted_set)
        
        if overlap > 0:
            self.predictions_correct += 1
        
        return accuracy
    
    def get_metrics(self) -> Dict[str, float]:
        """Get predictor performance metrics."""
        overall_accuracy = (
            self.predictions_correct / self.predictions_made
            if self.predictions_made > 0
            else 0.0
        )
        
        return {
            'predictions_made': self.predictions_made,
            'predictions_correct': self.predictions_correct,
            'overall_accuracy': overall_accuracy,
            'ngram_patterns': len(self.ngram_counts),
            'history_size': len(self.history),
        }


class PrefetchScheduler:
    """
    Speculative prefetching scheduler.
    
    Coordinates between prediction and actual dispatching to
    send warmup requests to expert nodes before tokens arrive.
    """
    
    def __init__(
        self,
        predictor: NGramPredictor,
        lookahead_tokens: int = 1,
        prefetch_enabled: bool = True,
    ):
        """
        Initialize prefetch scheduler.
        
        Args:
            predictor: N-gram predictor for expert patterns
            lookahead_tokens: Number of tokens to lookahead
            prefetch_enabled: Enable/disable prefetching
        """
        self.predictor = predictor
        self.lookahead_tokens = lookahead_tokens
        self.prefetch_enabled = prefetch_enabled
        
        # Prefetch request queue
        self.prefetch_queue: asyncio.Queue = asyncio.Queue()
        
        # Active prefetch requests (expert_id -> Future)
        self.active_prefetches: Dict[int, asyncio.Future] = {}
        
        # Metrics
        self.prefetch_requests_sent = 0
        self.prefetch_hits = 0  # Prefetch completed before actual request
        self.prefetch_misses = 0  # Prefetch not ready in time
        self.prefetch_wasted = 0  # Prefetch never used
        
        logger.info(f"Prefetch Scheduler initialized (lookahead={lookahead_tokens})")
    
    def schedule_prefetch(
        self,
        expert_ids: List[int],
        dispatch_callback: callable,
    ) -> None:
        """
        Schedule prefetch requests to experts.
        
        Args:
            expert_ids: List of expert IDs to prefetch
            dispatch_callback: Async function to send prefetch request
        """
        if not self.prefetch_enabled:
            return
        
        for expert_id in expert_ids:
            if expert_id in self.active_prefetches:
                # Already prefetching this expert
                continue
            
            # Create prefetch task
            task = asyncio.create_task(
                self._execute_prefetch(expert_id, dispatch_callback)
            )
            
            self.active_prefetches[expert_id] = task
            self.prefetch_requests_sent += 1
    
    async def _execute_prefetch(
        self,
        expert_id: int,
        dispatch_callback: callable,
    ) -> None:
        """
        Execute a prefetch request.
        
        Args:
            expert_id: Expert to prefetch
            dispatch_callback: Function to call for prefetch
        """
        try:
            # Send warmup signal (empty tensor or cache warming)
            logger.debug(f"Prefetching Expert {expert_id}")
            
            # In production, this would send a special "warmup" request
            # that prepares the expert but doesn't return full results
            # await dispatch_callback(expert_id, prefetch=True)
            
            # Placeholder: simulate prefetch latency
            await asyncio.sleep(0.01)
        
        except Exception as e:
            logger.error(f"Prefetch failed for Expert {expert_id}: {e}")
        
        finally:
            # Remove from active prefetches
            if expert_id in self.active_prefetches:
                del self.active_prefetches[expert_id]
    
    def check_prefetch_ready(self, expert_id: int) -> bool:
        """
        Check if a prefetch request has completed.
        
        Args:
            expert_id: Expert to check
            
        Returns:
            True if prefetch is ready (cache warm)
        """
        if expert_id not in self.active_prefetches:
            return False
        
        task = self.active_prefetches[expert_id]
        
        if task.done():
            self.prefetch_hits += 1
            return True
        else:
            self.prefetch_misses += 1
            return False
    
    def cancel_prefetch(self, expert_id: int) -> None:
        """
        Cancel an active prefetch request.
        
        Args:
            expert_id: Expert to cancel
        """
        if expert_id not in self.active_prefetches:
            return
        
        task = self.active_prefetches[expert_id]
        task.cancel()
        
        del self.active_prefetches[expert_id]
        self.prefetch_wasted += 1
    
    def observe_and_predict(
        self,
        token_idx: int,
        actual_experts: List[int],
        recent_history: List[List[int]],
    ) -> List[int]:
        """
        Observe actual expert usage and predict next.
        
        Args:
            token_idx: Current token index
            actual_experts: Experts actually used
            recent_history: Recent expert activation history
            
        Returns:
            Predicted experts for next token
        """
        # Record observation
        self.predictor.observe(token_idx, actual_experts)
        
        # Make prediction for next token
        predicted = self.predictor.predict(recent_history + [actual_experts])
        
        return predicted
    
    def get_metrics(self) -> Dict[str, float]:
        """Get scheduler performance metrics."""
        predictor_metrics = self.predictor.get_metrics()
        
        hit_rate = (
            self.prefetch_hits / (self.prefetch_hits + self.prefetch_misses)
            if (self.prefetch_hits + self.prefetch_misses) > 0
            else 0.0
        )
        
        waste_rate = (
            self.prefetch_wasted / self.prefetch_requests_sent
            if self.prefetch_requests_sent > 0
            else 0.0
        )
        
        return {
            **predictor_metrics,
            'prefetch_requests_sent': self.prefetch_requests_sent,
            'prefetch_hits': self.prefetch_hits,
            'prefetch_misses': self.prefetch_misses,
            'prefetch_wasted': self.prefetch_wasted,
            'hit_rate': hit_rate,
            'waste_rate': waste_rate,
            'active_prefetches': len(self.active_prefetches),
        }


# Factory function
def create_scheduler(
    num_experts: int = 8,
    ngram_order: int = 2,
    lookahead_tokens: int = 1,
    prefetch_enabled: bool = True,
) -> PrefetchScheduler:
    """
    Create a configured prefetch scheduler.
    
    Args:
        num_experts: Total number of experts
        ngram_order: N-gram predictor order
        lookahead_tokens: Lookahead window size
        prefetch_enabled: Enable prefetching
        
    Returns:
        Configured PrefetchScheduler instance
    """
    predictor = NGramPredictor(n=ngram_order, num_experts=num_experts)
    scheduler = PrefetchScheduler(
        predictor=predictor,
        lookahead_tokens=lookahead_tokens,
        prefetch_enabled=prefetch_enabled,
    )
    
    return scheduler
