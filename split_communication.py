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
        labels: Target labels for loss computation (optional, passed to server)
        attention_mask: Attention mask for the input (optional)
        phase: ZO phase indicator ("perturb_pos", "perturb_neg", "restore", etc.)
        option_len: List of option lengths for partial loss computation (matching run.py)
        mode: "train" or "inference" - in inference mode, server returns logits instead of loss
    """
    activations: torch.Tensor
    presents: Optional[List[torch.Tensor]]
    input_shape: Tuple[int, ...]
    rng_state: Optional[torch.Tensor] = None
    seed: Optional[int] = None
    batch_id: Optional[int] = None
    labels: Optional[torch.Tensor] = None
    attention_mask: Optional[torch.Tensor] = None
    phase: Optional[str] = None
    option_len: Optional[List[int]] = None
    mode: str = "train"  # "train" or "inference"
    
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
            attention_mask=self.attention_mask.to(device) if self.attention_mask is not None else None,
            phase=self.phase,
            option_len=self.option_len,
            mode=self.mode,
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
            attention_mask=self.attention_mask.detach() if self.attention_mask is not None else None,
            phase=self.phase,
            option_len=self.option_len,
            mode=self.mode,
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
    Metadata for zeroth-order (MeZO) optimization coordination.
    
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
    """
    seed: int
    rng_state: Optional[torch.Tensor] = None
    zo_eps: float = 1e-3
    scaling_factor: int = 1
    step_phase: str = "perturb_pos"
    projected_grad: Optional[float] = None
    restore_scaling_factor: int = 1  # For restoring params before update


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


# =============================================================================
# Communication Cost Tracker (DeComFL-style)
# =============================================================================

@dataclass
class CommunicationStats:
    """
    Track communication costs similar to DeComFL paper.
    
    DeComFL reduces communication from O(d) to O(1) by transmitting only
    scalar values and seeds instead of full model parameters/gradients.
    
    This tracker measures actual bytes transferred to compare with:
    1. Traditional FL: O(d) - full model/gradient transmission
    2. Split Learning: O(activation_size) - intermediate activations
    3. DeComFL-style ZO: O(1) - only scalars and seeds
    
    Reference: https://github.com/ZidongLiu/DeComFL
    """
    # Per-round statistics
    bytes_sent_per_round: List[int] = field(default_factory=list)
    bytes_received_per_round: List[int] = field(default_factory=list)
    
    # Cumulative totals
    total_bytes_sent: int = 0
    total_bytes_received: int = 0
    
    # Round counter
    num_rounds: int = 0
    
    # Payload type breakdown
    forward_payload_bytes: int = 0
    backward_payload_bytes: int = 0
    zo_metadata_bytes: int = 0
    
    def record_send(self, data_bytes: int, payload_type: str = "forward"):
        """Record bytes sent."""
        self.total_bytes_sent += data_bytes
        self.bytes_sent_per_round.append(data_bytes)
        
        if payload_type == "forward":
            self.forward_payload_bytes += data_bytes
        elif payload_type == "zo_metadata":
            self.zo_metadata_bytes += data_bytes
    
    def record_receive(self, data_bytes: int, payload_type: str = "backward"):
        """Record bytes received."""
        self.total_bytes_received += data_bytes
        self.bytes_received_per_round.append(data_bytes)
        
        if payload_type == "backward":
            self.backward_payload_bytes += data_bytes
    
    def increment_round(self):
        """Mark completion of a communication round."""
        self.num_rounds += 1
    
    @property
    def total_bytes(self) -> int:
        """Total bytes transferred (sent + received)."""
        return self.total_bytes_sent + self.total_bytes_received
    
    @property
    def avg_bytes_per_round(self) -> float:
        """Average bytes per communication round."""
        if self.num_rounds == 0:
            return 0.0
        return self.total_bytes / self.num_rounds
    
    def compute_traditional_fl_cost(self, model_params: int, dtype_bytes: int = 4) -> int:
        """
        Compute what communication would cost in traditional FL.
        
        In traditional FL, each round transmits:
        - Server -> Client: Full model parameters (d × dtype_bytes)
        - Client -> Server: Full gradients (d × dtype_bytes)
        
        Args:
            model_params: Number of model parameters (d)
            dtype_bytes: Bytes per parameter (4 for float32, 2 for float16)
            
        Returns:
            Total bytes that would be transferred in traditional FL
        """
        per_round_cost = 2 * model_params * dtype_bytes  # Both directions
        return per_round_cost * self.num_rounds
    
    def compute_decomfl_theoretical_cost(self, num_perturbations: int = 1,
                                          scalar_bytes: int = 4,
                                          seed_bytes: int = 8) -> int:
        """
        Compute theoretical DeComFL communication cost.
        
        DeComFL transmits only:
        - Client -> Server: P scalar values (projected gradients)
        - Server -> Client: Random seed
        
        Args:
            num_perturbations: Number of perturbations (P)
            scalar_bytes: Bytes per scalar (4 for float32)
            seed_bytes: Bytes for random seed
            
        Returns:
            Total theoretical bytes in DeComFL
        """
        per_round_cost = num_perturbations * scalar_bytes + seed_bytes
        return per_round_cost * self.num_rounds
    
    def get_summary(self, model_params: int = 0, num_perturbations: int = 1) -> dict:
        """
        Get comprehensive communication cost summary.
        
        Args:
            model_params: Total model parameters for comparison
            num_perturbations: Number of ZO perturbations
            
        Returns:
            Dictionary with all communication metrics
        """
        summary = {
            # Actual measured communication
            "total_bytes_sent": self.total_bytes_sent,
            "total_bytes_received": self.total_bytes_received,
            "total_bytes": self.total_bytes,
            "total_bytes_formatted": format_payload_size(self.total_bytes),
            "num_communication_rounds": self.num_rounds,
            "avg_bytes_per_round": self.avg_bytes_per_round,
            "avg_bytes_per_round_formatted": format_payload_size(int(self.avg_bytes_per_round)),
            
            # Breakdown by payload type
            "forward_payload_bytes": self.forward_payload_bytes,
            "backward_payload_bytes": self.backward_payload_bytes,
            "zo_metadata_bytes": self.zo_metadata_bytes,
            
            # Convert to common units
            "total_kb": self.total_bytes / 1024,
            "total_mb": self.total_bytes / (1024 * 1024),
            "total_gb": self.total_bytes / (1024 * 1024 * 1024),
        }
        
        # Add comparisons if model_params provided
        if model_params > 0:
            traditional_fl_cost = self.compute_traditional_fl_cost(model_params)
            decomfl_theoretical = self.compute_decomfl_theoretical_cost(num_perturbations)
            
            summary.update({
                # Traditional FL comparison
                "traditional_fl_bytes": traditional_fl_cost,
                "traditional_fl_formatted": format_payload_size(traditional_fl_cost),
                "savings_vs_traditional_fl": 1 - (self.total_bytes / traditional_fl_cost) if traditional_fl_cost > 0 else 0,
                "compression_ratio_vs_fl": traditional_fl_cost / self.total_bytes if self.total_bytes > 0 else 0,
                
                # DeComFL theoretical comparison
                "decomfl_theoretical_bytes": decomfl_theoretical,
                "decomfl_theoretical_formatted": format_payload_size(decomfl_theoretical),
            })
        
        return summary
    
    def print_summary(self, model_params: int = 0, num_perturbations: int = 1):
        """Print a formatted summary of communication costs."""
        summary = self.get_summary(model_params, num_perturbations)
        
        print("\n" + "=" * 60)
        print("COMMUNICATION COST SUMMARY (DeComFL-style)")
        print("=" * 60)
        print(f"Total Communication Rounds:  {summary['num_communication_rounds']}")
        print(f"Total Bytes Transferred:     {summary['total_bytes_formatted']}")
        print(f"  - Sent (client→server):    {format_payload_size(summary['total_bytes_sent'])}")
        print(f"  - Received (server→client):{format_payload_size(summary['total_bytes_received'])}")
        print(f"Average Bytes per Round:     {summary['avg_bytes_per_round_formatted']}")
        print("-" * 60)
        print("Payload Breakdown:")
        print(f"  - Forward payloads:        {format_payload_size(summary['forward_payload_bytes'])}")
        print(f"  - Backward payloads:       {format_payload_size(summary['backward_payload_bytes'])}")
        print(f"  - ZO metadata:             {format_payload_size(summary['zo_metadata_bytes'])}")
        
        if model_params > 0:
            print("-" * 60)
            print("Comparison with Traditional FL:")
            print(f"  Traditional FL would use:  {summary['traditional_fl_formatted']}")
            print(f"  Actual (Split+ZO):         {summary['total_bytes_formatted']}")
            print(f"  Savings:                   {summary['savings_vs_traditional_fl']*100:.2f}%")
            print(f"  Compression Ratio:         {summary['compression_ratio_vs_fl']:.2f}x")
            print("-" * 60)
            print(f"DeComFL Theoretical (pure):  {summary['decomfl_theoretical_formatted']}")
        print("=" * 60)

