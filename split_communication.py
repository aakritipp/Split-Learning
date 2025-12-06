# ------------------------------------------------------------------------------------------
# Serialization Helpers for Split Learning
#
# This module provides infrastructure for true split deployment where client and server
# run as independent processes/machines. It includes:
# 1. Dataclasses for structured payloads at the split boundary
# 2. Serialization/deserialization functions using torch.save/torch.load
# 3. Abstract communication backend interface for future implementations
#
# Usage:
#   - Current simulation mode: No changes needed, existing code works as-is
#   - Future deployment: Use payloads and serialize/deserialize for network transfer
# ------------------------------------------------------------------------------------------

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Optional, List, Tuple, Any, Union
from io import BytesIO

import torch


# =============================================================================
# Payload Dataclasses
# =============================================================================

@dataclass
class ForwardPayload:
    """
    Client -> Server: Payload containing activations at the split boundary.
    
    This is sent from the client to the server during the forward pass.
    In true split learning, this would be serialized and sent over the network.
    
    Attributes:
        activations: Hidden states at the split point (batch, seq_len, hidden_dim)
        presents: Attention cache tensors for KV continuation (list of tensors)
        input_shape: Original input shape (batch_size, seq_len) for reconstruction
        rng_state: PyTorch RNG state for continuous RNG mode (optional)
        seed: Perturbation seed for shared seed mode (optional)
        batch_id: Identifier to match forward/backward payloads (optional)
    """
    activations: torch.Tensor
    presents: Optional[List[torch.Tensor]]
    input_shape: Tuple[int, ...]
    rng_state: Optional[torch.Tensor] = None
    seed: Optional[int] = None
    batch_id: Optional[int] = None
    
    def to_device(self, device: Union[str, torch.device]) -> "ForwardPayload":
        """Move all tensors to the specified device."""
        return ForwardPayload(
            activations=self.activations.to(device),
            presents=[p.to(device) if p is not None else None for p in self.presents] if self.presents else None,
            input_shape=self.input_shape,
            rng_state=self.rng_state,  # RNG state stays on CPU
            seed=self.seed,
            batch_id=self.batch_id,
        )
    
    def detach(self) -> "ForwardPayload":
        """Detach all tensors from the computation graph."""
        return ForwardPayload(
            activations=self.activations.detach(),
            presents=[p.detach() if p is not None else None for p in self.presents] if self.presents else None,
            input_shape=self.input_shape,
            rng_state=self.rng_state,
            seed=self.seed,
            batch_id=self.batch_id,
        )


@dataclass
class BackwardPayload:
    """
    Server -> Client: Payload containing gradients or loss for the backward pass.
    
    In FO (first-order) mode: Contains gradients at the split point
    In ZO (zeroth-order) mode: Contains only the loss value (no gradients needed)
    
    Attributes:
        grad_activations: Gradient of loss w.r.t. activations at split (FO mode)
        loss: Scalar loss value (ZO mode, or for logging)
        batch_id: Identifier to match forward/backward payloads (optional)
    """
    grad_activations: Optional[torch.Tensor] = None
    loss: Optional[float] = None
    batch_id: Optional[int] = None
    
    def to_device(self, device: Union[str, torch.device]) -> "BackwardPayload":
        """Move all tensors to the specified device."""
        return BackwardPayload(
            grad_activations=self.grad_activations.to(device) if self.grad_activations is not None else None,
            loss=self.loss,
            batch_id=self.batch_id,
        )


@dataclass
class ZOMetadata:
    """
    Metadata for zeroth-order (MeZO) optimization coordination.
    
    This contains all the information needed for client and server to
    coordinate their perturbations in ZO mode.
    
    Attributes:
        seed: Shared perturbation seed
        rng_state: RNG state after client perturbation (for continuous RNG)
        zo_eps: Perturbation scale epsilon
        scaling_factor: Current perturbation direction (+1, -2, +1)
        step_phase: Current phase ("perturb_pos", "perturb_neg", "restore", "update")
    """
    seed: int
    rng_state: Optional[torch.Tensor] = None
    zo_eps: float = 1e-3
    scaling_factor: int = 1
    step_phase: str = "perturb_pos"


# =============================================================================
# Serialization Functions
# =============================================================================

def serialize_payload(payload: Any) -> bytes:
    """
    Serialize a payload to bytes for network transmission.
    
    Uses torch.save with BytesIO for efficient tensor serialization.
    Supports ForwardPayload, BackwardPayload, ZOMetadata, or any picklable object.
    
    Args:
        payload: The payload object to serialize
        
    Returns:
        Serialized bytes representation
    """
    buffer = BytesIO()
    torch.save(payload, buffer)
    return buffer.getvalue()


def deserialize_payload(data: bytes, map_location: Optional[Union[str, torch.device]] = None) -> Any:
    """
    Deserialize bytes back into a payload object.
    
    Args:
        data: Serialized bytes from serialize_payload
        map_location: Device to load tensors to (e.g., 'cpu', 'cuda:0')
        
    Returns:
        Deserialized payload object
    """
    buffer = BytesIO(data)
    return torch.load(buffer, map_location=map_location, weights_only=False)


def serialize_tensor(tensor: torch.Tensor) -> bytes:
    """Serialize a single tensor to bytes."""
    buffer = BytesIO()
    torch.save(tensor, buffer)
    return buffer.getvalue()


def deserialize_tensor(data: bytes, map_location: Optional[Union[str, torch.device]] = None) -> torch.Tensor:
    """Deserialize bytes back into a tensor."""
    buffer = BytesIO(data)
    return torch.load(buffer, map_location=map_location, weights_only=False)


# =============================================================================
# Abstract Communication Backend
# =============================================================================

class CommunicationBackend(ABC):
    """
    Abstract interface for split learning communication backends.
    
    Implementations can use different transport layers:
    - LocalBackend: In-memory (current simulation mode)
    - SocketBackend: TCP sockets (future)
    - GRPCBackend: gRPC (future)
    - MPIBackend: MPI for HPC (future)
    
    Usage:
        backend = LocalBackend()
        
        # Client side
        backend.send_forward(forward_payload)
        backward_payload = backend.recv_backward()
        
        # Server side
        forward_payload = backend.recv_forward()
        backend.send_backward(backward_payload)
    """
    
    @abstractmethod
    def send_forward(self, payload: ForwardPayload) -> None:
        """Send forward payload from client to server."""
        pass
    
    @abstractmethod
    def recv_forward(self) -> ForwardPayload:
        """Receive forward payload on server side."""
        pass
    
    @abstractmethod
    def send_backward(self, payload: BackwardPayload) -> None:
        """Send backward payload from server to client."""
        pass
    
    @abstractmethod
    def recv_backward(self) -> BackwardPayload:
        """Receive backward payload on client side."""
        pass
    
    @abstractmethod
    def send_zo_metadata(self, metadata: ZOMetadata) -> None:
        """Send ZO metadata (seed, RNG state) from client to server."""
        pass
    
    @abstractmethod
    def recv_zo_metadata(self) -> ZOMetadata:
        """Receive ZO metadata on server side."""
        pass
    
    def close(self) -> None:
        """Clean up resources (optional override)."""
        pass


class LocalBackend(CommunicationBackend):
    """
    In-memory communication backend for simulation (current behavior).
    
    Stores payloads in memory without any network transfer.
    This is the default backend that maintains backward compatibility
    with the existing simulation-based split learning.
    """
    
    def __init__(self):
        self._forward_payload: Optional[ForwardPayload] = None
        self._backward_payload: Optional[BackwardPayload] = None
        self._zo_metadata: Optional[ZOMetadata] = None
    
    def send_forward(self, payload: ForwardPayload) -> None:
        """Store forward payload in memory."""
        self._forward_payload = payload
    
    def recv_forward(self) -> ForwardPayload:
        """Retrieve forward payload from memory."""
        if self._forward_payload is None:
            raise RuntimeError("No forward payload available. Call send_forward first.")
        payload = self._forward_payload
        self._forward_payload = None  # Clear after reading
        return payload
    
    def send_backward(self, payload: BackwardPayload) -> None:
        """Store backward payload in memory."""
        self._backward_payload = payload
    
    def recv_backward(self) -> BackwardPayload:
        """Retrieve backward payload from memory."""
        if self._backward_payload is None:
            raise RuntimeError("No backward payload available. Call send_backward first.")
        payload = self._backward_payload
        self._backward_payload = None  # Clear after reading
        return payload
    
    def send_zo_metadata(self, metadata: ZOMetadata) -> None:
        """Store ZO metadata in memory."""
        self._zo_metadata = metadata
    
    def recv_zo_metadata(self) -> ZOMetadata:
        """Retrieve ZO metadata from memory."""
        if self._zo_metadata is None:
            raise RuntimeError("No ZO metadata available. Call send_zo_metadata first.")
        metadata = self._zo_metadata
        self._zo_metadata = None  # Clear after reading
        return metadata


# =============================================================================
# Utility Functions
# =============================================================================

def compute_payload_size(payload: Any) -> int:
    """
    Compute the serialized size of a payload in bytes.
    
    Useful for monitoring communication overhead.
    
    Args:
        payload: The payload to measure
        
    Returns:
        Size in bytes
    """
    return len(serialize_payload(payload))


def format_payload_size(size_bytes: int) -> str:
    """Format byte size as human-readable string."""
    if size_bytes < 1024:
        return f"{size_bytes} B"
    elif size_bytes < 1024 * 1024:
        return f"{size_bytes / 1024:.2f} KB"
    elif size_bytes < 1024 * 1024 * 1024:
        return f"{size_bytes / (1024 * 1024):.2f} MB"
    else:
        return f"{size_bytes / (1024 * 1024 * 1024):.2f} GB"

