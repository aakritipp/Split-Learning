"""
Communication utilities for split learning.

This module provides dataclasses and serialization functions for exchanging
payloads between client and server in a split learning setup. It defines
the protocol for forward/backward communication and ZO metadata exchange.

Key components:
- ForwardPayload: Client-to-server activations and metadata
- BackwardPayload: Server-to-client gradients or loss values
- ZOMetadata: Perturbation seeds and scaling for ZO optimization
- CommunicationStats: Track communication costs
- LocalBackend: In-memory simulation for testing
"""
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
        labels: Target labels for loss computation (optional, passed to server)
        attention_mask: Attention mask for the input (optional)
        phase: ZO phase indicator ("perturb_pos", "perturb_neg", "restore", etc.)
        option_len: List of option lengths for partial loss computation (matching run.py)
        num_options: List of num_options for classification-style training (train_as_classification)
        mode: "train" or "inference" - in inference mode, server returns logits instead of loss
    """
    activations: torch.Tensor
    presents: Optional[List[torch.Tensor]]
    input_shape: Tuple[int, ...]
    rng_state: Optional[torch.Tensor] = None
    seed: Optional[int] = None
    batch_id: Optional[int] = None
    labels: Optional[torch.Tensor] = None
    input_ids: Optional[torch.Tensor] = None  # For classification mode loss computation
    attention_mask: Optional[torch.Tensor] = None
    phase: Optional[str] = None
    option_len: Optional[List[int]] = None
    num_options: Optional[List[int]] = None  # For train_as_classification mode
    mode: str = "train"  # "train" or "inference"
    num_perturbations: int = 1  # For ZO-FO: scale loss by 1/num_perturbations before backward
    perturbation_idx: int = 0  # Current perturbation index (0 to num_perturbations-1)
    
    def to_device(self, device: Union[str, torch.device]) -> "ForwardPayload":
        """Move all tensors to the specified device."""
        return ForwardPayload(
            activations=self.activations.to(device),
            presents=[p.to(device) if p is not None else None for p in self.presents] if self.presents else None,
            input_shape=self.input_shape,
            rng_state=self.rng_state,  # RNG state stays on CPU
            seed=self.seed,
            batch_id=self.batch_id,
            labels=self.labels.to(device) if self.labels is not None else None,
            input_ids=self.input_ids.to(device) if self.input_ids is not None else None,
            attention_mask=self.attention_mask.to(device) if self.attention_mask is not None else None,
            phase=self.phase,
            option_len=self.option_len,
            num_options=self.num_options,
            mode=self.mode,
            num_perturbations=self.num_perturbations,
            perturbation_idx=self.perturbation_idx,
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
            labels=self.labels.detach() if self.labels is not None else None,
            input_ids=self.input_ids.detach() if self.input_ids is not None else None,
            attention_mask=self.attention_mask.detach() if self.attention_mask is not None else None,
            phase=self.phase,
            option_len=self.option_len,
            num_options=self.num_options,
            mode=self.mode,
            num_perturbations=self.num_perturbations,
            perturbation_idx=self.perturbation_idx,
        )


@dataclass
class BackwardPayload:
    """
    Server -> Client: Payload containing gradients or loss for the backward pass.
    
    In FO (first-order) mode: Contains gradients at the split point
    In ZO (zeroth-order) mode: Contains only the loss value (no gradients needed)
    In inference mode: Contains logits for prediction
    
    Attributes:
        grad_activations: Gradient of loss w.r.t. activations at split (FO mode)
        loss: Scalar loss value (ZO mode, or for logging)
        batch_id: Identifier to match forward/backward payloads (optional)
        logits: Output logits for inference mode (shape: batch, seq_len, vocab_size)
    """
    grad_activations: Optional[torch.Tensor] = None
    loss: Optional[float] = None
    batch_id: Optional[int] = None
    logits: Optional[torch.Tensor] = None
    
    def to_device(self, device: Union[str, torch.device]) -> "BackwardPayload":
        """Move all tensors to the specified device."""
        return BackwardPayload(
            grad_activations=self.grad_activations.to(device) if self.grad_activations is not None else None,
            loss=self.loss,
            batch_id=self.batch_id,
            logits=self.logits.to(device) if self.logits is not None else None,
        )


@dataclass
class ZOMetadata:
    """
    Metadata for zeroth-order optimization coordination.
    
    This contains all the information needed for client and server to
    coordinate their perturbations in ZO mode.
    
    Attributes:
        seed: Shared perturbation seed
        rng_state: RNG state after client perturbation (for continuous RNG)
        zo_eps: Perturbation scale epsilon
        scaling_factor: Current perturbation direction (+1, -2, +1)
        step_phase: Current phase ("perturb_pos", "perturb_neg", "restore", "update")
        projected_grad: The computed gradient estimate for update phase
        restore_scaling_factor: Scaling factor to restore params before update
        accumulated_updates: List of (seed, projected_grad) tuples for batch update
    """
    seed: int
    rng_state: Optional[torch.Tensor] = None
    zo_eps: float = 1e-3
    scaling_factor: int = 1
    step_phase: str = "perturb_pos"
    projected_grad: Optional[float] = None
    restore_scaling_factor: int = 1  # For restoring params before update
    perturbation_idx: int = 0  # For ZO-FO: track which perturbation we're on
    num_perturbations: int = 1  # For ZO-FO: total perturbations to know when to step
    accumulated_updates: Optional[List[Tuple[int, float]]] = None  # For batched ZO updates


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
