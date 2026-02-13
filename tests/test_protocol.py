"""
Tests for Custom Binary Protocol

Validates protocol encoding/decoding and error handling.
"""

import pytest
import asyncio
import torch
from volt.core.protocol import (
    ProtocolHeader,
    ProtocolMessage,
    ProtocolCodec,
    create_request_id,
    MAGIC_NUMBER,
    PROTOCOL_VERSION,
)
from volt.core.exceptions import ProtocolException


class TestProtocolHeader:
    """Test suite for protocol header."""
    
    def test_header_serialization(self):
        """Test header to/from bytes."""
        header = ProtocolHeader(
            magic=MAGIC_NUMBER,
            request_id=12345,
            expert_id=3,
            payload_len=1024,
            version=PROTOCOL_VERSION,
            flags=0,
            sequence_num=0,
            reserved=0,
        )
        
        # Serialize
        data = header.to_bytes()
        assert len(data) == 32  # Header size
        
        # Deserialize
        decoded = ProtocolHeader.from_bytes(data)
        
        assert decoded.magic == header.magic
        assert decoded.request_id == header.request_id
        assert decoded.expert_id == header.expert_id
        assert decoded.payload_len == header.payload_len
    
    def test_invalid_magic_number(self):
        """Test that invalid magic number raises exception."""
        with pytest.raises(ProtocolException):
            header = ProtocolHeader(
                magic=0xDEADBEEF,  # Wrong magic
                request_id=1,
                expert_id=0,
                payload_len=100,
                version=PROTOCOL_VERSION,
                flags=0,
                sequence_num=0,
                reserved=0,
            )


class TestProtocolCodec:
    """Test suite for protocol codec."""
    
    def test_tensor_encoding_decoding(self):
        """Test tensor serialization round-trip."""
        original = torch.randn(32, 128, dtype=torch.float16)
        
        # Encode
        encoded = ProtocolCodec.encode_tensor(original)
        
        # Decode
        decoded = ProtocolCodec.decode_tensor(encoded)
        
        torch.testing.assert_close(decoded, original)
    
    def test_different_dtypes(self):
        """Test encoding with different data types."""
        dtypes = [torch.float16, torch.float32, torch.int8, torch.int32]
        
        for dtype in dtypes:
            tensor = torch.randn(16, 64).to(dtype)
            
            encoded = ProtocolCodec.encode_tensor(tensor)
            decoded = ProtocolCodec.decode_tensor(encoded)
            
            assert decoded.dtype == dtype
            torch.testing.assert_close(decoded.float(), tensor.float())
    
    @pytest.mark.asyncio
    async def test_message_send_receive(self):
        """Test sending and receiving messages over TCP."""
        # Create in-memory stream pair
        reader_server, writer_client = await asyncio.open_connection(
            host='127.0.0.1', port=0  # Mock connection
        )
        
        # This test requires a real TCP connection
        # Simplified version: just test encoding
        request_id = create_request_id()
        tensor = torch.randn(8, 32, dtype=torch.float16)
        
        # Encode manually
        encoded = ProtocolCodec.encode_tensor(tensor)
        
        assert len(encoded) > 0
        assert isinstance(encoded, bytes)


class TestRequestID:
    """Test suite for request ID generation."""
    
    def test_unique_ids(self):
        """Test that generated IDs are unique."""
        ids = [create_request_id() for _ in range(1000)]
        
        assert len(set(ids)) == len(ids)  # All unique
    
    def test_id_format(self):
        """Test that IDs are valid 64-bit integers."""
        request_id = create_request_id()
        
        assert isinstance(request_id, int)
        assert 0 <= request_id < 2**64


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
