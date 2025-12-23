"""
gRPC and TCP communication backends for distributed split learning.

This module provides network communication backends that enable split learning
across multiple machines. It implements both gRPC (robust) and TCP (simple)
protocols for payload exchange.

Key components:
- SplitLearningServicer: gRPC service implementation for server side
- TCPBackend: Simple TCP socket-based communication
- GRPCBackend: Full gRPC-based communication with streaming support
"""
import logging
import time
import threading
from concurrent import futures
from typing import Optional, Any, Callable
from io import BytesIO

import grpc
import torch

from split_communication import (
    CommunicationBackend,
    ForwardPayload,
    BackwardPayload,
    ZOMetadata,
    serialize_payload,
    deserialize_payload,
)

logger = logging.getLogger(__name__)

# Maximum message size (1GB for large activations)
MAX_MESSAGE_SIZE = 1024 * 1024 * 1024


# =============================================================================
# gRPC Service Implementation (Server-side)
# =============================================================================

class SplitLearningServicer:
    """
    gRPC service implementation for the server side of split learning.
    
    Handles incoming requests from the client and processes them through
    the server portion of the model.
    """
    
    def __init__(self, device: str = "cuda"):
        self.device = device
        self._forward_handler: Optional[Callable] = None
        self._train_handler: Optional[Callable] = None
        self._lock = threading.Lock()
        self._is_ready = False
        
    def set_forward_handler(self, handler: Callable):
        """Set the handler function for forward pass requests."""
        self._forward_handler = handler
        
    def set_train_handler(self, handler: Callable):
        """Set the handler function for training requests."""
        self._train_handler = handler
        self._is_ready = True
        
    def Forward(self, request_data: bytes, context) -> bytes:
        """Handle forward pass request from client."""
        if self._forward_handler is None:
            raise RuntimeError("Forward handler not set")
            
        with self._lock:
            # Deserialize the forward payload
            forward_payload = deserialize_payload(request_data, map_location=self.device)
            
            # Process through server model
            backward_payload = self._forward_handler(forward_payload)
            
            # Serialize and return
            return serialize_payload(backward_payload)
    
    def Train(self, request_data: bytes, zo_metadata_bytes: Optional[bytes], 
              step: int, mode: str) -> tuple:
        """Handle training request from client."""
        if self._train_handler is None:
            raise RuntimeError("Train handler not set")
            
        with self._lock:
            # Deserialize payloads
            forward_payload = deserialize_payload(request_data, map_location=self.device)
            zo_metadata = None
            if zo_metadata_bytes:
                zo_metadata = deserialize_payload(zo_metadata_bytes)
            
            # Process training step
            backward_payload, loss = self._train_handler(
                forward_payload, zo_metadata, step, mode
            )
            
            # Serialize and return
            return serialize_payload(backward_payload), loss
    
    def HealthCheck(self) -> tuple:
        """Check if server is ready."""
        return self._is_ready, "ready" if self._is_ready else "not_ready"


import socket
import struct
import pickle


class TCPBackend(CommunicationBackend):
    def __init__(
        self,
        mode: str = 'client',  # 'client' or 'server'
        host: str = 'localhost',
        port: int = 50051,
        device: str = 'cpu',
        timeout: float = 300.0,  # 5 minute timeout for large transfers
    ):
        self.mode = mode
        self.host = host
        self.port = port
        self.device = device
        self.timeout = timeout
        
        self._socket: Optional[socket.socket] = None
        self._client_socket: Optional[socket.socket] = None
        self._connected = False
        self._lock = threading.Lock()
        
    def start(self):
        """Start the server (server mode only)."""
        if self.mode != 'server':
            raise RuntimeError("start() is only for server mode")
            
        self._socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self._socket.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        self._socket.bind((self.host, self.port))
        self._socket.listen(1)
        logger.info(f"Split Learning Server listening on {self.host}:{self.port}")
        
        # Wait for client connection
        self._accept_client()
        
    def _accept_client(self):
        """Accept a client connection. Can be called multiple times to re-accept."""
        if self._client_socket:
            try:
                self._client_socket.close()
            except:
                pass
            self._client_socket = None
            self._connected = False
            
        logger.info("Waiting for client connection...")
        self._client_socket, client_addr = self._socket.accept()
        self._client_socket.settimeout(self.timeout)
        self._connected = True
        logger.info(f"Client connected from {client_addr}")
    
    def wait_for_client(self):
        """Public method to wait for a new client connection (for reconnection after probe disconnects)."""
        self._accept_client()
        
    def reaccept_client(self):
        """Re-accept a new client connection (for handling probe connections)."""
        self._accept_client()
        
    def connect(self):
        """Connect to server (client mode only)."""
        if self.mode != 'client':
            raise RuntimeError("connect() is only for client mode")
            
        self._socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self._socket.settimeout(self.timeout)
        
        logger.info(f"Connecting to server at {self.host}:{self.port}...")
        self._socket.connect((self.host, self.port))
        self._connected = True
        logger.info("Connected to server")
        
    def _get_socket(self) -> socket.socket:
        """Get the active socket for communication."""
        if self.mode == 'server':
            if self._client_socket is None:
                raise RuntimeError("No client connected")
            return self._client_socket
        else:
            if self._socket is None:
                raise RuntimeError("Not connected to server")
            return self._socket
    
    def _send_data(self, data: bytes):
        """Send data with length prefix."""
        sock = self._get_socket()
        length = len(data)
        sock.sendall(struct.pack('>Q', length))
        # Send data in chunks
        sock.sendall(data)
        
    def _recv_data(self) -> bytes:
        """Receive data with length prefix."""
        sock = self._get_socket()
        length_bytes = self._recv_exactly(sock, 8)
        length = struct.unpack('>Q', length_bytes)[0]
        return self._recv_exactly(sock, length)
    
    def _recv_exactly(self, sock: socket.socket, n: int) -> bytes:
        """Receive exactly n bytes."""
        data = bytearray()
        while len(data) < n:
            chunk = sock.recv(min(n - len(data), 65536))
            if not chunk:
                raise ConnectionError("Connection closed")
            data.extend(chunk)
        return bytes(data)
    
    def send_forward(self, payload: ForwardPayload) -> None:
        """Send forward payload from client to server."""
        with self._lock:
            data = serialize_payload(('forward', payload))
            self._send_data(data)
            logger.debug(f"Sent forward payload: {len(data)} bytes")
    
    def recv_forward(self) -> ForwardPayload:
        """Receive forward payload on server side."""
        with self._lock:
            data = self._recv_data()
            msg_type, payload = deserialize_payload(data, map_location=self.device)
            if msg_type != 'forward':
                raise RuntimeError(f"Expected 'forward', got '{msg_type}'")
            logger.debug(f"Received forward payload: {len(data)} bytes")
            return payload
    
    def send_backward(self, payload: BackwardPayload) -> None:
        """Send backward payload from server to client."""
        with self._lock:
            data = serialize_payload(('backward', payload))
            self._send_data(data)
            logger.debug(f"Sent backward payload: {len(data)} bytes")
    
    def recv_backward(self) -> BackwardPayload:
        """Receive backward payload on client side."""
        with self._lock:
            data = self._recv_data()
            msg_type, payload = deserialize_payload(data, map_location=self.device)
            if msg_type != 'backward':
                raise RuntimeError(f"Expected 'backward', got '{msg_type}'")
            logger.debug(f"Received backward payload: {len(data)} bytes")
            return payload
    
    def send_zo_metadata(self, metadata: ZOMetadata) -> None:
        """Send ZO metadata from client to server."""
        with self._lock:
            data = serialize_payload(('zo_metadata', metadata))
            self._send_data(data)
            logger.debug(f"Sent ZO metadata: seed={metadata.seed}, {len(data)} bytes")
    
    def recv_zo_metadata(self) -> ZOMetadata:
        """Receive ZO metadata on server side."""
        with self._lock:
            data = self._recv_data()
            msg_type, metadata = deserialize_payload(data, map_location='cpu')
            if msg_type != 'zo_metadata':
                raise RuntimeError(f"Expected 'zo_metadata', got '{msg_type}'")
            logger.debug(f"Received ZO metadata: seed={metadata.seed}")
            return metadata
    
    def send_command(self, command: str, data: Any = None) -> Any:
        """Send a control command and wait for response."""
        with self._lock:
            cmd_data = serialize_payload(('command', command, data))
            self._send_data(cmd_data)
            
            response_data = self._recv_data()
            msg_type, response = deserialize_payload(response_data, map_location=self.device)
            if msg_type != 'response':
                raise RuntimeError(f"Expected 'response', got '{msg_type}'")
            return response
    
    def recv_command(self) -> tuple:
        """Receive a control command (server side)."""
        with self._lock:
            data = self._recv_data()
            msg_type, command, cmd_data = deserialize_payload(data, map_location=self.device)
            if msg_type != 'command':
                raise RuntimeError(f"Expected 'command', got '{msg_type}'")
            return command, cmd_data
    
    def send_response(self, response: Any) -> None:
        """Send a response to a command (server side)."""
        with self._lock:
            data = serialize_payload(('response', response))
            self._send_data(data)
    
    def close(self):
        """Close the connection."""
        self._connected = False
        if self._client_socket:
            self._client_socket.close()
            self._client_socket = None
        if self._socket:
            self._socket.close()
            self._socket = None
        logger.info("Connection closed")


class GRPCBackend(CommunicationBackend):
    def __init__(
        self,
        mode: str = 'client',
        host: str = 'localhost',
        port: int = 50051,
        device: str = 'cpu',
        max_workers: int = 4,
    ):
        self.mode = mode
        self.host = host
        self.port = port
        self.device = device
        self.max_workers = max_workers
        
        self._server = None
        self._channel = None
        self._stub = None
        self._servicer = None
        self._lock = threading.Lock()
        
    def start_server(self, train_handler: Callable):
        if self.mode != 'server':
            raise RuntimeError("start_server() is only for server mode")
        
        # Create servicer
        self._servicer = SplitLearningServicer(device=self.device)
        self._servicer.set_train_handler(train_handler)
        
        # Create server
        self._server = grpc.server(
            futures.ThreadPoolExecutor(max_workers=self.max_workers),
            options=[
                ('grpc.max_send_message_length', MAX_MESSAGE_SIZE),
                ('grpc.max_receive_message_length', MAX_MESSAGE_SIZE),
            ]
        )
        
        # Register the custom servicer (we'll use raw bytes for simplicity)
        # In production, you'd generate stubs from the proto file
        self._register_servicer()
        
        # Start server
        address = f'{self.host}:{self.port}'
        self._server.add_insecure_port(address)
        self._server.start()
        logger.info(f"gRPC Server started on {address}")
        
    def _register_servicer(self):
        """Register the servicer with generic handlers."""
        # For simplicity, we use a generic handler that routes based on method name
        # In production, you'd use generated stubs from protoc
        
        def handle_rpc(handler_call_details, request_iterator, servicer_context):
            method = handler_call_details.method
            
            if method.endswith('/Train'):
                for request in request_iterator:
                    try:
                        # Parse request (simple format: step|mode|forward_data|zo_data)
                        forward_data, zo_data, step, mode = deserialize_payload(request)
                        backward_data, loss = self._servicer.Train(
                            forward_data, zo_data, step, mode
                        )
                        yield serialize_payload((backward_data, loss))
                    except Exception as e:
                        logger.error(f"Train error: {e}")
                        yield serialize_payload((None, str(e)))
                        
            elif method.endswith('/HealthCheck'):
                healthy, status = self._servicer.HealthCheck()
                yield serialize_payload((healthy, status))
                
        # Add generic handler
        handler = grpc.method_handlers_generic_handler(
            'splitlearning.SplitLearningService',
            {'Train': grpc.stream_stream_rpc_method_handler(handle_rpc)}
        )
        self._server.add_generic_rpc_handlers((handler,))
        
    def connect(self, timeout: float = 30.0):
        """Connect to the gRPC server."""
        if self.mode != 'client':
            raise RuntimeError("connect() is only for client mode")
            
        address = f'{self.host}:{self.port}'
        logger.info(f"Connecting to gRPC server at {address}...")
        
        self._channel = grpc.insecure_channel(
            address,
            options=[
                ('grpc.max_send_message_length', MAX_MESSAGE_SIZE),
                ('grpc.max_receive_message_length', MAX_MESSAGE_SIZE),
            ]
        )
        
        # Wait for connection
        try:
            grpc.channel_ready_future(self._channel).result(timeout=timeout)
            logger.info("Connected to gRPC server")
        except grpc.FutureTimeoutError:
            raise ConnectionError(f"Could not connect to server at {address}")
            
    def send_forward(self, payload: ForwardPayload) -> None:
        """Send forward payload (client side)."""
        raise NotImplementedError("Use send_train_request for gRPC backend")
        
    def recv_forward(self) -> ForwardPayload:
        """Receive forward payload (server side)."""
        raise NotImplementedError("Use train_handler callback for gRPC backend")
        
    def send_backward(self, payload: BackwardPayload) -> None:
        """Send backward payload (server side)."""
        raise NotImplementedError("Use train_handler return for gRPC backend")
        
    def recv_backward(self) -> BackwardPayload:
        """Receive backward payload (client side)."""
        raise NotImplementedError("Use send_train_request for gRPC backend")
    
    def send_zo_metadata(self, metadata: ZOMetadata) -> None:
        """Send ZO metadata (included in train request)."""
        raise NotImplementedError("Include in send_train_request")
        
    def recv_zo_metadata(self) -> ZOMetadata:
        """Receive ZO metadata (included in train request)."""
        raise NotImplementedError("Included in train_handler callback")
    
    def send_train_request(
        self,
        forward_payload: ForwardPayload,
        zo_metadata: Optional[ZOMetadata] = None,
        step: int = 0,
        mode: str = 'zo'
    ) -> tuple:
        if self.mode != 'client':
            raise RuntimeError("send_train_request() is only for client mode")
            
        # Serialize the request
        forward_data = serialize_payload(forward_payload)
        zo_data = serialize_payload(zo_metadata) if zo_metadata else None
        request = serialize_payload((forward_data, zo_data, step, mode))
        
        # Make RPC call (using raw bytes for simplicity)
        with self._lock:
            # For a simple implementation, we'll use a unary call
            # In production, use the generated stubs
            response = self._channel.unary_unary(
                '/splitlearning.SplitLearningService/Train',
                request_serializer=lambda x: x,
                response_deserializer=lambda x: x,
            )(request)
            
            backward_data, loss = deserialize_payload(response, map_location=self.device)
            backward_payload = deserialize_payload(backward_data, map_location=self.device)
            
            return backward_payload, loss
    
    def wait_for_termination(self):
        """Block until the server is terminated."""
        if self._server:
            self._server.wait_for_termination()
            
    def close(self):
        """Close the connection/server."""
        if self._channel:
            self._channel.close()
            self._channel = None
        if self._server:
            self._server.stop(grace=5)
            self._server = None
        logger.info("gRPC backend closed")


def create_backend(
    backend_type: str = 'local',
    mode: str = 'client',
    host: str = 'localhost',
    port: int = 50051,
    device: str = 'cpu',
    **kwargs
) -> CommunicationBackend:
    """
    Factory function to create communication backends.
    
    Args:
        backend_type: 'local', 'tcp', or 'grpc'
        mode: 'client' or 'server'
        host: Server hostname/IP
        port: Server port
        device: Device to load tensors on
        **kwargs: Additional backend-specific arguments
        
    Returns:
        CommunicationBackend instance
    """
    from split_communication import LocalBackend
    
    if backend_type == 'local':
        return LocalBackend()
    elif backend_type == 'tcp':
        return TCPBackend(mode=mode, host=host, port=port, device=device, **kwargs)
    elif backend_type == 'grpc':
        return GRPCBackend(mode=mode, host=host, port=port, device=device, **kwargs)
    else:
        raise ValueError(f"Unknown backend type: {backend_type}")

