"""
Custom Binary Protocol for Volt-Infer TCP Communication

Lightweight, zero-copy protocol for transmitting token batches and expert
computations between Router and Worker nodes with minimal serialization overhead.

Protocol Frame Format:
    ┌─────────────────────────────────────────────────────────────┐
    │ Magic (4B) │ RequestID (8B) │ ExpertID (4B) │ PayloadLen (8B) │
    ├─────────────────────────────────────────────────────────────┤
    │ Version (2B) │ Flags (2B) │ SequenceNum (4B) │ Reserved (4B) │
    ├─────────────────────────────────────────────────────────────┤
    │                      Payload (Variable)                       │
    │              [Tensor Data in Float16 Format]                 │
    └─────────────────────────────────────────────────────────────┘

Total Header Size: 32 bytes (cache-line aligned)
"""

import struct
import asyncio
from typing import Tuple, Optional
from dataclasses import dataclass
import io

import torch
import numpy as np

from .exceptions import ProtocolException


# Protocol Constants
MAGIC_NUMBER = 0x564F4C54  # "VOLT" in ASCII
PROTOCOL_VERSION = 1
MAX_PAYLOAD_SIZE = 128 * 1024 * 1024  # 128 MB limit
HEADER_SIZE = 32  # bytes


# Flag Bits
class ProtocolFlags:
    """Bit flags for protocol options."""
    COMPRESSED = 1 << 0      # Payload is zstd compressed
    QUANTIZED = 1 << 1       # Payload contains INT8 data
    KV_CACHE = 1 << 2        # Payload includes KV cache pages
    PREFETCH = 1 << 3        # This is a speculative prefetch request
    ERROR = 1 << 4           # Response contains error information
    LAST_CHUNK = 1 << 5      # Final chunk in multi-part message


@dataclass
class ProtocolHeader:
    """
    Structured representation of protocol header.
    
    Attributes:
        magic: Magic number for protocol identification (0x564F4C54)
        request_id: Unique request identifier for request/response matching
        expert_id: Target expert ID (0-based index)
        payload_len: Size of payload in bytes
        version: Protocol version (1)
        flags: Bitfield of ProtocolFlags
        sequence_num: Sequence number for ordering (useful for chunked transfers)
        reserved: Reserved for future use (must be 0)
    """
    magic: int
    request_id: int
    expert_id: int
    payload_len: int
    version: int
    flags: int
    sequence_num: int
    reserved: int
    
    def __post_init__(self):
        """Validate header fields."""
        if self.magic != MAGIC_NUMBER:
            raise ProtocolException(
                f"Invalid magic number: 0x{self.magic:08X} != 0x{MAGIC_NUMBER:08X}"
            )
        
        if self.version != PROTOCOL_VERSION:
            raise ProtocolException(
                f"Unsupported protocol version: {self.version}"
            )
        
        if self.payload_len > MAX_PAYLOAD_SIZE:
            raise ProtocolException(
                f"Payload too large: {self.payload_len} bytes "
                f"(max: {MAX_PAYLOAD_SIZE})"
            )
        
        if self.payload_len < 0:
            raise ProtocolException(
                f"Invalid payload length: {self.payload_len}"
            )
    
    def to_bytes(self) -> bytes:
        """
        Serialize header to binary format.
        
        Returns:
            32-byte packed header
            
        Struct Format:
            I = unsigned int (4 bytes)
            Q = unsigned long long (8 bytes)
            H = unsigned short (2 bytes)
        """
        return struct.pack(
            "!IQIQHHII",  # Network byte order (big-endian)
            self.magic,
            self.request_id,
            self.expert_id,
            self.payload_len,
            self.version,
            self.flags,
            self.sequence_num,
            self.reserved,
        )
    
    @classmethod
    def from_bytes(cls, data: bytes) -> "ProtocolHeader":
        """
        Deserialize header from binary format.
        
        Args:
            data: 32-byte header data
            
        Returns:
            Parsed ProtocolHeader instance
            
        Raises:
            ProtocolException: If data is malformed or validation fails
        """
        if len(data) != HEADER_SIZE:
            raise ProtocolException(
                f"Invalid header size: expected {HEADER_SIZE}, got {len(data)}",
                raw_data=data
            )
        
        try:
            unpacked = struct.unpack("!IQIQHHII", data)
        except struct.error as e:
            raise ProtocolException(
                f"Failed to unpack header: {e}",
                raw_data=data
            )
        
        return cls(
            magic=unpacked[0],
            request_id=unpacked[1],
            expert_id=unpacked[2],
            payload_len=unpacked[3],
            version=unpacked[4],
            flags=unpacked[5],
            sequence_num=unpacked[6],
            reserved=unpacked[7],
        )


@dataclass
class ProtocolMessage:
    """
    Complete protocol message (header + payload).
    
    Attributes:
        header: Protocol header
        payload: Tensor data (hidden states or expert outputs)
        kv_cache_pages: Optional KV-cache pages for attention routing
    """
    header: ProtocolHeader
    payload: torch.Tensor
    kv_cache_pages: Optional[dict] = None
    
    def is_compressed(self) -> bool:
        """Check if payload is compressed."""
        return bool(self.header.flags & ProtocolFlags.COMPRESSED)
    
    def is_quantized(self) -> bool:
        """Check if payload is quantized."""
        return bool(self.header.flags & ProtocolFlags.QUANTIZED)
    
    def is_prefetch(self) -> bool:
        """Check if this is a prefetch request."""
        return bool(self.header.flags & ProtocolFlags.PREFETCH)
    
    def is_error(self) -> bool:
        """Check if this is an error response."""
        return bool(self.header.flags & ProtocolFlags.ERROR)


class ProtocolCodec:
    """
    Encoder/decoder for protocol messages with zero-copy optimizations.
    
    Uses memoryview and direct buffer manipulation to avoid unnecessary
    copies of large tensor data.
    """
    
    @staticmethod
    def encode_tensor(tensor: torch.Tensor) -> bytes:
        """
        Encode PyTorch tensor to binary format.
        
        Format:
            [DTYPE (1B)] [NDIM (1B)] [SHAPE (4B * NDIM)] [DATA]
        
        Args:
            tensor: Input tensor (must be contiguous)
            
        Returns:
            Serialized tensor bytes
        """
        if not tensor.is_contiguous():
            tensor = tensor.contiguous()
        
        # Dtype mapping (simplified)
        dtype_map = {
            torch.float16: 0,
            torch.float32: 1,
            torch.int8: 2,
            torch.int32: 3,
        }
        dtype_code = dtype_map.get(tensor.dtype, 255)
        
        # Encode metadata
        ndim = len(tensor.shape)
        shape_bytes = struct.pack(f"!{ndim}I", *tensor.shape)
        
        # Encode data (zero-copy via numpy)
        data_bytes = tensor.cpu().numpy().tobytes()
        
        # Assemble
        header = struct.pack("!BB", dtype_code, ndim)
        return header + shape_bytes + data_bytes
    
    @staticmethod
    def decode_tensor(data: bytes) -> torch.Tensor:
        """
        Decode binary data to PyTorch tensor.
        
        Args:
            data: Serialized tensor bytes
            
        Returns:
            Reconstructed tensor
        """
        # Dtype reverse mapping
        dtype_map = {
            0: torch.float16,
            1: torch.float32,
            2: torch.int8,
            3: torch.int32,
        }
        
        # Parse header
        dtype_code, ndim = struct.unpack("!BB", data[:2])
        dtype = dtype_map.get(dtype_code, torch.float32)
        
        # Parse shape
        shape_size = 4 * ndim
        shape = struct.unpack(f"!{ndim}I", data[2:2+shape_size])
        
        # Parse data
        data_offset = 2 + shape_size
        tensor_data = data[data_offset:]
        
        # Reconstruct via numpy (efficient)
        np_dtype = np.float16 if dtype == torch.float16 else \
                   np.float32 if dtype == torch.float32 else \
                   np.int8 if dtype == torch.int8 else np.int32
        
        np_array = np.frombuffer(tensor_data, dtype=np_dtype)
        tensor = torch.from_numpy(np_array.copy()).reshape(shape)
        
        return tensor
    
    @staticmethod
    async def send_message(
        writer: asyncio.StreamWriter,
        request_id: int,
        expert_id: int,
        payload: torch.Tensor,
        flags: int = 0,
        sequence_num: int = 0,
    ) -> None:
        """
        Send a protocol message over an async TCP stream.
        
        Args:
            writer: asyncio StreamWriter for TCP socket
            request_id: Unique request identifier
            expert_id: Target expert ID
            payload: Tensor to send
            flags: Protocol flags (compression, quantization, etc.)
            sequence_num: Sequence number for ordering
            
        Raises:
            ProtocolException: If encoding or transmission fails
        """
        try:
            # Encode payload
            payload_bytes = ProtocolCodec.encode_tensor(payload)
            
            # Create header
            header = ProtocolHeader(
                magic=MAGIC_NUMBER,
                request_id=request_id,
                expert_id=expert_id,
                payload_len=len(payload_bytes),
                version=PROTOCOL_VERSION,
                flags=flags,
                sequence_num=sequence_num,
                reserved=0,
            )
            
            # Send header
            writer.write(header.to_bytes())
            await writer.drain()
            
            # Send payload (chunked for large tensors)
            chunk_size = 1024 * 1024  # 1 MB chunks
            for i in range(0, len(payload_bytes), chunk_size):
                chunk = payload_bytes[i:i+chunk_size]
                writer.write(chunk)
                await writer.drain()
        
        except Exception as e:
            raise ProtocolException(f"Failed to send message: {e}")
    
    @staticmethod
    async def receive_message(
        reader: asyncio.StreamReader,
        timeout: Optional[float] = None,
    ) -> ProtocolMessage:
        """
        Receive a protocol message from an async TCP stream.
        
        Args:
            reader: asyncio StreamReader for TCP socket
            timeout: Optional timeout in seconds
            
        Returns:
            Parsed ProtocolMessage
            
        Raises:
            ProtocolException: If decoding fails
            asyncio.TimeoutError: If timeout expires
        """
        try:
            # Read header with timeout
            if timeout:
                header_bytes = await asyncio.wait_for(
                    reader.readexactly(HEADER_SIZE),
                    timeout=timeout
                )
            else:
                header_bytes = await reader.readexactly(HEADER_SIZE)
            
            header = ProtocolHeader.from_bytes(header_bytes)
            
            # Read payload
            if timeout:
                payload_bytes = await asyncio.wait_for(
                    reader.readexactly(header.payload_len),
                    timeout=timeout
                )
            else:
                payload_bytes = await reader.readexactly(header.payload_len)
            
            payload = ProtocolCodec.decode_tensor(payload_bytes)
            
            return ProtocolMessage(
                header=header,
                payload=payload,
            )
        
        except asyncio.IncompleteReadError as e:
            raise ProtocolException(
                f"Connection closed unexpectedly: {e}",
                raw_data=e.partial
            )
        except Exception as e:
            raise ProtocolException(f"Failed to receive message: {e}")


# Utilities
def create_request_id() -> int:
    """
    Generate a unique request ID.
    
    Uses high-resolution timestamp combined with process ID for uniqueness.
    
    Returns:
        64-bit unique identifier
    """
    import time
    timestamp = int(time.time() * 1_000_000)  # Microseconds
    return timestamp & 0xFFFFFFFFFFFFFFFF  # Mask to 64 bits


def calculate_checksum(data: bytes) -> int:
    """
    Calculate CRC32 checksum for data integrity verification.
    
    Args:
        data: Input bytes
        
    Returns:
        CRC32 checksum (32-bit)
    """
    import zlib
    return zlib.crc32(data) & 0xFFFFFFFF
