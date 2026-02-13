"""
Paged Attention KV-Cache Manager for Volt-Infer

Efficient management of Key-Value cache pages for Transformer attention
with support for distributed routing and memory-efficient storage.

Based on vLLM's paged attention design: each KV cache is split into
fixed-size pages (e.g., 16 tokens per page) that can be independently
allocated, transferred, and freed.
"""

from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass
import threading

import torch


# Configuration Constants
DEFAULT_PAGE_SIZE = 16  # tokens per page
DEFAULT_BLOCK_SIZE = 256  # KB per block


@dataclass
class PageTableEntry:
    """
    Entry in the page table mapping logical to physical pages.
    
    Attributes:
        logical_page_id: Virtual page ID (sequence-local)
        physical_page_id: Physical page ID in memory pool
        ref_count: Number of sequences sharing this page (for prefix sharing)
        pinned: Whether page is pinned in memory (cannot be evicted)
        last_access_time: Timestamp for LRU eviction policy
    """
    logical_page_id: int
    physical_page_id: int
    ref_count: int = 1
    pinned: bool = False
    last_access_time: float = 0.0


class PagedKVCache:
    """
    Paged KV-Cache manager with memory pooling and copy-on-write.
    
    Design Principles:
        1. Fixed-size pages enable efficient memory allocation
        2. Page table indirection supports prefix sharing (important for prompts)
        3. Copy-on-write for forked sequences (beam search)
        4. LRU eviction for memory pressure scenarios
    
    Memory Layout:
        Physical Memory: [Page 0][Page 1][Page 2]...[Page N]
        Each page stores keys and values for PAGE_SIZE tokens
        
        Page Table: sequence_id -> [logical -> physical] mapping
    """
    
    def __init__(
        self,
        num_layers: int,
        num_heads: int,
        head_dim: int,
        page_size: int = DEFAULT_PAGE_SIZE,
        max_pages: int = 1024,
        device: str = 'cuda:0',
        dtype: torch.dtype = torch.float16,
    ):
        """
        Initialize paged KV cache manager.
        
        Args:
            num_layers: Number of transformer layers
            num_heads: Number of attention heads per layer
            head_dim: Dimension per attention head
            page_size: Number of tokens per page
            max_pages: Maximum number of physical pages to allocate
            device: Device for cache storage
            dtype: Data type for cache tensors
        """
        self.num_layers = num_layers
        self.num_heads = num_heads
        self.head_dim = head_dim
        self.page_size = page_size
        self.max_pages = max_pages
        self.device = device
        self.dtype = dtype
        
        # Allocate physical memory pool
        # Shape: [num_layers, 2, max_pages, page_size, num_heads, head_dim]
        # The '2' dimension is for keys and values
        self.memory_pool = torch.zeros(
            num_layers,
            2,  # keys and values
            max_pages,
            page_size,
            num_heads,
            head_dim,
            device=device,
            dtype=dtype,
        )
        
        # Free page tracking
        self.free_pages: List[int] = list(range(max_pages))
        self.free_pages_lock = threading.Lock()
        
        # Page tables: sequence_id -> {logical_page_id -> PageTableEntry}
        self.page_tables: Dict[int, Dict[int, PageTableEntry]] = {}
        self.page_tables_lock = threading.Lock()
        
        # Usage statistics
        self.allocated_pages = 0
        self.evicted_pages = 0
        self.cache_hits = 0
        self.cache_misses = 0
    
    def allocate_page(self) -> Optional[int]:
        """
        Allocate a physical page from the free list.
        
        Returns:
            Physical page ID, or None if no pages available
        """
        with self.free_pages_lock:
            if not self.free_pages:
                # Try to evict a page
                evicted_page = self._evict_page()
                if evicted_page is None:
                    return None
                self.free_pages.append(evicted_page)
            
            physical_page_id = self.free_pages.pop(0)
            self.allocated_pages += 1
            
            return physical_page_id
    
    def free_page(self, physical_page_id: int) -> None:
        """
        Return a physical page to the free list.
        
        Args:
            physical_page_id: Physical page to free
        """
        with self.free_pages_lock:
            # Zero out the page for security
            self.memory_pool[:, :, physical_page_id, :, :, :].zero_()
            
            self.free_pages.append(physical_page_id)
            self.allocated_pages -= 1
    
    def create_sequence(self, sequence_id: int) -> None:
        """
        Create a new sequence with an empty page table.
        
        Args:
            sequence_id: Unique sequence identifier
        """
        with self.page_tables_lock:
            if sequence_id in self.page_tables:
                raise ValueError(f"Sequence {sequence_id} already exists")
            
            self.page_tables[sequence_id] = {}
    
    def delete_sequence(self, sequence_id: int) -> None:
        """
        Delete a sequence and free all its pages.
        
        Args:
            sequence_id: Sequence to delete
        """
        with self.page_tables_lock:
            if sequence_id not in self.page_tables:
                return
            
            page_table = self.page_tables.pop(sequence_id)
            
            # Free all pages (respecting ref counts)
            for entry in page_table.values():
                entry.ref_count -= 1
                if entry.ref_count == 0:
                    self.free_page(entry.physical_page_id)
    
    def append_tokens(
        self,
        sequence_id: int,
        layer_idx: int,
        key: torch.Tensor,  # [num_new_tokens, num_heads, head_dim]
        value: torch.Tensor,  # [num_new_tokens, num_heads, head_dim]
    ) -> bool:
        """
        Append new tokens' KV cache to a sequence.
        
        Args:
            sequence_id: Target sequence
            layer_idx: Transformer layer index
            key: Key tensor for new tokens
            value: Value tensor for new tokens
            
        Returns:
            True if successful, False if out of memory
        """
        num_new_tokens = key.shape[0]
        
        with self.page_tables_lock:
            if sequence_id not in self.page_tables:
                self.create_sequence(sequence_id)
            
            page_table = self.page_tables[sequence_id]
            
            # Determine how many pages we need
            num_existing_tokens = len(page_table) * self.page_size
            total_tokens = num_existing_tokens + num_new_tokens
            num_pages_needed = (total_tokens + self.page_size - 1) // self.page_size
            
            # Allocate new pages if necessary
            for logical_page_id in range(len(page_table), num_pages_needed):
                physical_page_id = self.allocate_page()
                if physical_page_id is None:
                    # Out of memory
                    return False
                
                page_table[logical_page_id] = PageTableEntry(
                    logical_page_id=logical_page_id,
                    physical_page_id=physical_page_id,
                )
            
            # Copy tokens into pages
            token_offset = num_existing_tokens
            for i in range(num_new_tokens):
                # Determine which page this token belongs to
                global_token_idx = token_offset + i
                logical_page_id = global_token_idx // self.page_size
                page_token_offset = global_token_idx % self.page_size
                
                # Get physical page
                entry = page_table[logical_page_id]
                physical_page_id = entry.physical_page_id
                
                # Write to memory pool
                self.memory_pool[layer_idx, 0, physical_page_id, page_token_offset] = key[i]
                self.memory_pool[layer_idx, 1, physical_page_id, page_token_offset] = value[i]
            
            return True
    
    def get_kv(
        self,
        sequence_id: int,
        layer_idx: int,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Retrieve all KV cache for a sequence at a specific layer.
        
        Args:
            sequence_id: Target sequence
            layer_idx: Transformer layer index
            
        Returns:
            (keys, values) tensors [seq_len, num_heads, head_dim]
        """
        with self.page_tables_lock:
            if sequence_id not in self.page_tables:
                raise ValueError(f"Sequence {sequence_id} not found")
            
            page_table = self.page_tables[sequence_id]
            
            if not page_table:
                # Empty sequence
                return (
                    torch.empty(0, self.num_heads, self.head_dim, device=self.device, dtype=self.dtype),
                    torch.empty(0, self.num_heads, self.head_dim, device=self.device, dtype=self.dtype),
                )
            
            # Gather all pages
            num_pages = len(page_table)
            keys_list = []
            values_list = []
            
            for logical_page_id in range(num_pages):
                entry = page_table[logical_page_id]
                physical_page_id = entry.physical_page_id
                
                # Read from memory pool
                page_keys = self.memory_pool[layer_idx, 0, physical_page_id]  # [page_size, num_heads, head_dim]
                page_values = self.memory_pool[layer_idx, 1, physical_page_id]
                
                keys_list.append(page_keys)
                values_list.append(page_values)
            
            # Concatenate all pages
            keys = torch.cat(keys_list, dim=0)
            values = torch.cat(values_list, dim=0)
            
            return keys, values
    
    def fork_sequence(
        self,
        source_sequence_id: int,
        target_sequence_id: int,
    ) -> None:
        """
        Fork a sequence (copy-on-write for beam search).
        
        Args:
            source_sequence_id: Source sequence to fork from
            target_sequence_id: New sequence ID
        """
        with self.page_tables_lock:
            if source_sequence_id not in self.page_tables:
                raise ValueError(f"Source sequence {source_sequence_id} not found")
            
            if target_sequence_id in self.page_tables:
                raise ValueError(f"Target sequence {target_sequence_id} already exists")
            
            # Shallow copy: increment ref counts
            source_page_table = self.page_tables[source_sequence_id]
            target_page_table = {}
            
            for logical_page_id, entry in source_page_table.items():
                entry.ref_count += 1
                target_page_table[logical_page_id] = entry
            
            self.page_tables[target_sequence_id] = target_page_table
    
    def _evict_page(self) -> Optional[int]:
        """
        Evict a page using LRU policy.
        
        Returns:
            Physical page ID to evict, or None if no evictable pages
        """
        # Find least recently used, unpinned page
        oldest_time = float('inf')
        evict_sequence_id = None
        evict_logical_page_id = None
        
        for sequence_id, page_table in self.page_tables.items():
            for logical_page_id, entry in page_table.items():
                if not entry.pinned and entry.last_access_time < oldest_time:
                    oldest_time = entry.last_access_time
                    evict_sequence_id = sequence_id
                    evict_logical_page_id = logical_page_id
        
        if evict_sequence_id is None:
            # No evictable pages
            return None
        
        # Remove from page table
        page_table = self.page_tables[evict_sequence_id]
        entry = page_table.pop(evict_logical_page_id)
        
        self.evicted_pages += 1
        
        return entry.physical_page_id
    
    def get_memory_usage(self) -> Dict[str, float]:
        """
        Get current memory usage statistics.
        
        Returns:
            Dictionary with memory metrics
        """
        total_memory_mb = (
            self.memory_pool.element_size() * self.memory_pool.numel()
        ) / (1024 ** 2)
        
        used_memory_mb = (
            (self.allocated_pages / self.max_pages) * total_memory_mb
        )
        
        return {
            'total_mb': total_memory_mb,
            'used_mb': used_memory_mb,
            'free_mb': total_memory_mb - used_memory_mb,
            'utilization': self.allocated_pages / self.max_pages,
            'allocated_pages': self.allocated_pages,
            'free_pages': len(self.free_pages),
            'evicted_pages': self.evicted_pages,
        }
    
    def get_sequence_info(self, sequence_id: int) -> Dict[str, int]:
        """
        Get information about a specific sequence.
        
        Args:
            sequence_id: Target sequence
            
        Returns:
            Dictionary with sequence metrics
        """
        with self.page_tables_lock:
            if sequence_id not in self.page_tables:
                raise ValueError(f"Sequence {sequence_id} not found")
            
            page_table = self.page_tables[sequence_id]
            
            return {
                'num_pages': len(page_table),
                'num_tokens': len(page_table) * self.page_size,
                'physical_pages': [
                    entry.physical_page_id for entry in page_table.values()
                ],
            }


def create_kv_cache_for_expert(
    num_layers: int = 32,
    num_heads: int = 32,
    head_dim: int = 128,
    device: str = 'cuda:0',
) -> PagedKVCache:
    """
    Factory function to create a KV cache for an expert node.
    
    Args:
        num_layers: Number of transformer layers
        num_heads: Number of attention heads
        head_dim: Dimension per head
        device: Device string
        
    Returns:
        Configured PagedKVCache instance
    """
    return PagedKVCache(
        num_layers=num_layers,
        num_heads=num_heads,
        head_dim=head_dim,
        device=device,
    )
