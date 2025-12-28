"""
Split learning model implementation for OPT architecture (GPT-2 style).

This module provides a custom OPT implementation optimized for split learning,
with built-in causal masking and LoRA support. Unlike splitmodel.py which uses
HuggingFace OPT backend, this provides full control over the architecture.

Key components:
- OPTAttention: Custom attention with built-in causal mask
- OPTModel_Client/Server: Split OPT components
- SplitOPT: Combined split OPT model
- Weight loading from HuggingFace pretrained models
"""
import logging
import math
import copy
import json
from collections import OrderedDict

import torch
from torch import nn
from torch.nn import Parameter
import torch.nn.functional as F
from transformers.modeling_outputs import CausalLMOutput

import lora

# Serialization helpers for future true split deployment
from split_communication import (
    ForwardPayload,
    BackwardPayload,
    ZOMetadata,
    serialize_payload,
    deserialize_payload,
)


# =============================================================================
# Activation Functions
# =============================================================================

def relu(x):
    """ReLU activation - OPT uses ReLU instead of GELU"""
    return F.relu(x)


# =============================================================================
# Layer Normalization
# =============================================================================

class LayerNorm(nn.Module):
    """LayerNorm with optional bias (OPT style)"""
    def __init__(self, hidden_size, eps=1e-5, bias=True):
        super(LayerNorm, self).__init__()
        self.weight = nn.Parameter(torch.ones(hidden_size))
        self.bias = nn.Parameter(torch.zeros(hidden_size)) if bias else None
        self.variance_epsilon = eps

    def forward(self, x):
        u = x.mean(-1, keepdim=True)
        s = (x - u).pow(2).mean(-1, keepdim=True)
        x = (x - u) / torch.sqrt(s + self.variance_epsilon)
        if self.bias is not None:
            return self.weight * x + self.bias
        return self.weight * x


# =============================================================================
# OPT Attention (with built-in causal mask, like GPT2)
# =============================================================================

class OPTAttention(nn.Module):
    """
    OPT Multi-Head Self-Attention with built-in causal mask.
    
    Key features:
    - Causal mask is pre-computed and stored as a buffer (like GPT2)
    - Supports LoRA on Q and V projections
    - No external mask preparation needed
    """
    def __init__(self, hidden_size, num_heads, max_positions, config, scale=True):
        super(OPTAttention, self).__init__()
        
        self.hidden_size = hidden_size
        self.num_heads = num_heads
        self.head_dim = hidden_size // num_heads
        self.scale = scale
        
        assert hidden_size % num_heads == 0, \
            f"hidden_size {hidden_size} must be divisible by num_heads {num_heads}"
        
        # Pre-compute causal mask (like GPT2's self.bias)
        self.register_buffer(
            "bias",
            torch.tril(torch.ones(max_positions, max_positions)).view(1, 1, max_positions, max_positions)
        )
        
        # Q, K, V projections with LoRA support
        # LoRA on Q and V (like GPT2), not on K
        lora_r = getattr(config, 'lora_attn_dim', 0)
        lora_alpha = getattr(config, 'lora_attn_alpha', 128)
        lora_dropout = getattr(config, 'lora_dropout', 0.0)
        
        if lora_r > 0:
            self.q_proj = lora.Linear(
                hidden_size, hidden_size,
                r=lora_r, lora_alpha=lora_alpha, lora_dropout=lora_dropout,
                merge_weights=False
            )
            self.v_proj = lora.Linear(
                hidden_size, hidden_size,
                r=lora_r, lora_alpha=lora_alpha, lora_dropout=lora_dropout,
                merge_weights=False
            )
        else:
            self.q_proj = nn.Linear(hidden_size, hidden_size)
            self.v_proj = nn.Linear(hidden_size, hidden_size)
        
        # K projection without LoRA
        self.k_proj = nn.Linear(hidden_size, hidden_size)
        
        # Output projection
        self.out_proj = nn.Linear(hidden_size, hidden_size)
        
        self.config = config

    def _attn(self, q, k, v, attention_mask=None, len_kv=None):
        """
        Compute attention with causal mask (OPT style).

        Args:
            q: Query tensor (batch, head, seq, head_dim)
            k: Key tensor (batch, head, head_dim, seq) - already transposed
            v: Value tensor (batch, head, seq, head_dim)
            attention_mask: Padding mask (batch, seq) - 1 for valid, 0 for padding
            len_kv: Optional length for variable-length masking
        """
        # Compute attention scores
        w = torch.matmul(q, k)  # (batch, head, q_len, k_len)

        # Scale
        if self.scale:
            w = w / math.sqrt(self.head_dim)

        # Use dtype-aware masking value (avoid overflow in fp16)
        dtype = w.dtype
        if dtype == torch.float16:
            mask_value = -65504.0  # Safe value for fp16
        elif dtype == torch.bfloat16:
            mask_value = -3.4e38  # Safe value for bf16
        else:
            mask_value = torch.finfo(dtype).min

        # Apply causal mask (OPT style)
        nd, ns = w.size(-2), w.size(-1)
        causal_mask = self.bias[:, :, ns-nd:ns, :ns]
        # Use boolean mask for efficiency
        w = w.masked_fill(causal_mask == 0, mask_value)

        # Apply padding mask if provided
        if attention_mask is not None:
            # attention_mask: (batch, seq) -> (batch, 1, 1, seq)
            # 1 = valid token, 0 = padding token (should be masked)
            padding_mask = (attention_mask == 0).unsqueeze(1).unsqueeze(2)
            w = w.masked_fill(padding_mask, mask_value)

        # Optional variable-length masking
        if len_kv is not None:
            _len = torch.arange(k.size(-1), device=k.device)
            _input_msk = _len[None, :] >= (len_kv)[:, None]
            w = w.masked_fill(_input_msk.unsqueeze(1).unsqueeze(2), mask_value)

        # Softmax (force float32 for numerical stability, then convert back)
        w = F.softmax(w, dim=-1, dtype=torch.float32).to(v.dtype)

        # Weighted sum of values
        return torch.matmul(w, v)

    def split_heads(self, x, k=False):
        """Split hidden dimension into multiple heads"""
        new_shape = x.size()[:-1] + (self.num_heads, self.head_dim)
        x = x.view(*new_shape)
        if k:
            return x.permute(0, 2, 3, 1).contiguous()  # (batch, head, head_dim, seq)
        else:
            return x.permute(0, 2, 1, 3).contiguous()  # (batch, head, seq, head_dim)

    def merge_heads(self, x):
        """Merge heads back into hidden dimension"""
        x = x.permute(0, 2, 1, 3).contiguous()
        new_shape = x.size()[:-2] + (self.hidden_size,)
        return x.view(*new_shape)

    def forward(self, hidden_states, attention_mask=None, layer_past=None, len_past=None):
        """
        Forward pass with optional KV cache for generation.
        
        Args:
            hidden_states: Input tensor (batch, seq, hidden)
            attention_mask: Padding mask (batch, seq) - 1 for valid, 0 for padding
            layer_past: Cached (key, value) from previous steps
            len_past: Length of past sequence for incremental decoding
        """
        # Project Q, K, V
        query = self.q_proj(hidden_states)
        key = self.k_proj(hidden_states)
        value = self.v_proj(hidden_states)
        
        # Split into heads
        query = self.split_heads(query)
        key = self.split_heads(key, k=True)
        value = self.split_heads(value)
        
        len_kv = None
        
        # Handle KV cache for generation
        if layer_past is not None:
            if len_past is None:
                past_key, past_value = layer_past[0].transpose(-2, -1), layer_past[1]
                key = torch.cat((past_key, key), dim=-1)
                value = torch.cat((past_value, value), dim=-2)
            else:
                key_seq = key.shape[-1]
                assert key_seq == 1
                
                _batch = torch.arange(0, key.shape[0], dtype=torch.long, device=key.device)
                past_key, past_value = layer_past[0], layer_past[1]
                
                past_key[_batch, :, len_past, :] = key.squeeze(-1)
                past_value[_batch, :, len_past, :] = value.squeeze(-2)
                
                key = past_key.transpose(-2, -1)
                value = past_value
                len_kv = len_past + 1
        
        # Store present for caching
        present = torch.stack((key.transpose(-2, -1), value))
        
        # Compute attention
        attn_output = self._attn(query, key, value, attention_mask=attention_mask, len_kv=len_kv)
        
        # Merge heads and project output
        attn_output = self.merge_heads(attn_output)
        attn_output = self.out_proj(attn_output)
        
        return attn_output, present


# =============================================================================
# OPT MLP (Feed-Forward Network)
# =============================================================================

class OPTMLP(nn.Module):
    """
    OPT Feed-Forward Network.
    
    Uses activation function from config (typically ReLU for OPT).
    """
    def __init__(self, hidden_size, ffn_dim, config):
        super(OPTMLP, self).__init__()
        self.fc1 = nn.Linear(hidden_size, ffn_dim)
        self.fc2 = nn.Linear(ffn_dim, hidden_size)
        
        # Get activation from config - OPT typically uses ReLU
        activation_fn = getattr(config, 'activation_function', 'relu')
        if activation_fn == 'gelu':
            self.act = F.gelu
        elif activation_fn == 'gelu_new' or activation_fn == 'gelu_fast':
            self.act = lambda x: F.gelu(x, approximate='tanh')
        elif activation_fn == 'silu' or activation_fn == 'swish':
            self.act = F.silu
        else:  # relu (default for OPT)
            self.act = F.relu
        
    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.fc2(x)
        return x


# =============================================================================
# OPT Block (Transformer Layer)
# =============================================================================

class OPTBlock(nn.Module):
    """
    OPT Transformer Block.
    
    Architecture (Pre-LayerNorm):
    - LayerNorm -> Self-Attention -> Residual
    - LayerNorm -> FFN -> Residual
    """
    def __init__(self, hidden_size, num_heads, ffn_dim, max_positions, config, scale=True):
        super(OPTBlock, self).__init__()
        
        # Get layer norm epsilon from config (default 1e-5)
        layer_norm_eps = getattr(config, 'layer_norm_epsilon', 1e-5)
        
        self.self_attn_layer_norm = LayerNorm(hidden_size, eps=layer_norm_eps)
        self.self_attn = OPTAttention(hidden_size, num_heads, max_positions, config, scale)
        
        self.final_layer_norm = LayerNorm(hidden_size, eps=layer_norm_eps)
        self.mlp = OPTMLP(hidden_size, ffn_dim, config)
        
        # Dropout
        dropout = getattr(config, 'dropout', 0.1)
        self.dropout = nn.Dropout(dropout)

    def forward(self, hidden_states, attention_mask=None, layer_past=None, len_past=None):
        """
        Forward pass through the block.
        
        Args:
            hidden_states: Input tensor (batch, seq, hidden)
            attention_mask: Padding mask (batch, seq) - 1 for valid, 0 for padding
        
        Returns:
            hidden_states: Output tensor
            present: Attention cache for generation
        """
        # Self-attention with residual
        residual = hidden_states
        hidden_states = self.self_attn_layer_norm(hidden_states)
        attn_output, present = self.self_attn(hidden_states, attention_mask=attention_mask, layer_past=layer_past, len_past=len_past)
        hidden_states = residual + self.dropout(attn_output)
        
        # FFN with residual
        residual = hidden_states
        hidden_states = self.final_layer_norm(hidden_states)
        mlp_output = self.mlp(hidden_states)
        hidden_states = residual + self.dropout(mlp_output)
        
        return hidden_states, present


# =============================================================================
# OPT Configuration
# =============================================================================

class OPTConfig:
    """
    Configuration for OPT model.
    
    Mirrors GPT2Config structure for consistency.
    """
    def __init__(
        self,
        vocab_size=50272,
        max_position_embeddings=2048,
        hidden_size=768,
        num_hidden_layers=12,
        num_attention_heads=12,
        ffn_dim=3072,  # Usually 4 * hidden_size
        dropout=0.1,
        attention_dropout=0.0,
        activation_function="relu",
        layer_norm_epsilon=1e-5,
        # Embedding projection (for OPT-350M+)
        word_embed_proj_dim=None,  # If None, defaults to hidden_size
        # LoRA parameters
        lora_attn_dim=0,
        lora_attn_alpha=128,
        lora_dropout=0.0,
        # Padding
        pad_token_id=1,
        bos_token_id=2,
        eos_token_id=2,
    ):
        self.vocab_size = vocab_size
        self.max_position_embeddings = max_position_embeddings
        self.hidden_size = hidden_size
        self.num_hidden_layers = num_hidden_layers
        self.num_attention_heads = num_attention_heads
        self.ffn_dim = ffn_dim
        self.dropout = dropout
        self.attention_dropout = attention_dropout
        self.activation_function = activation_function
        self.layer_norm_epsilon = layer_norm_epsilon
        
        # Embedding projection dimension (OPT-350M+ uses different dim for embeddings)
        self.word_embed_proj_dim = word_embed_proj_dim if word_embed_proj_dim is not None else hidden_size
        
        # LoRA
        self.lora_attn_dim = lora_attn_dim
        self.lora_attn_alpha = lora_attn_alpha
        self.lora_dropout = lora_dropout
        
        # Special tokens
        self.pad_token_id = pad_token_id
        self.bos_token_id = bos_token_id
        self.eos_token_id = eos_token_id
        
        # Aliases for compatibility
        self.n_layer = num_hidden_layers
        self.n_embd = hidden_size
        self.n_head = num_attention_heads
        self.model_type = "opt_split"

    def to_dict(self):
        """Return JSON-serializable dictionary"""
        output = {}
        for k, v in self.__dict__.items():
            if k.startswith("_"):
                continue
            try:
                json.dumps(v)
                output[k] = v
            except TypeError:
                output[k] = str(v)
        output["__class__"] = self.__class__.__name__
        return output

    def to_json_string(self) -> str:
        return json.dumps(self.to_dict(), indent=2, sort_keys=True) + "\n"

    @classmethod
    def from_pretrained(cls, model_name: str):
        """Create config from HuggingFace model"""
        from transformers import AutoConfig
        hf_config = AutoConfig.from_pretrained(model_name)
        
        # Get word_embed_proj_dim (differs from hidden_size for OPT-350M+)
        word_embed_proj_dim = getattr(hf_config, 'word_embed_proj_dim', hf_config.hidden_size)
        
        return cls(
            vocab_size=hf_config.vocab_size,
            max_position_embeddings=hf_config.max_position_embeddings,
            hidden_size=hf_config.hidden_size,
            num_hidden_layers=hf_config.num_hidden_layers,
            num_attention_heads=hf_config.num_attention_heads,
            ffn_dim=hf_config.ffn_dim,
            dropout=getattr(hf_config, 'dropout', 0.1),
            attention_dropout=getattr(hf_config, 'attention_dropout', 0.0),
            activation_function=getattr(hf_config, 'activation_function', 'relu'),
            layer_norm_epsilon=1e-5,  # OPT uses 1e-5
            word_embed_proj_dim=word_embed_proj_dim,
            pad_token_id=getattr(hf_config, 'pad_token_id', 1),
            bos_token_id=getattr(hf_config, 'bos_token_id', 2),
            eos_token_id=getattr(hf_config, 'eos_token_id', 2),
        )


# =============================================================================
# OPT Model Client (Embeddings + First N Layers)
# =============================================================================

class OPTModel_Client(nn.Module):
    """
    Client-side OPT model with embeddings and first N layers.
    
    Like GPT2Model_Client but for OPT architecture.
    """
    def __init__(self, config, split_layer: int = 3):
        super(OPTModel_Client, self).__init__()
        
        self.config = config
        self.split_layer = split_layer
        
        # Validate
        assert 0 <= split_layer <= config.num_hidden_layers, \
            f"split_layer must be between 0 and {config.num_hidden_layers}"
        
        # Get embedding dimension (may differ from hidden_size for OPT-350M+)
        embed_dim = getattr(config, 'word_embed_proj_dim', config.hidden_size)
        
        # Embeddings use word_embed_proj_dim
        self.embed_tokens = nn.Embedding(config.vocab_size, embed_dim, padding_idx=config.pad_token_id)
        # OPT uses offset=2 for position embeddings, so size is max_position_embeddings + 2
        self.embed_positions = nn.Embedding(config.max_position_embeddings + 2, config.hidden_size)
        
        # Projection layer (only for OPT-350M+ where word_embed_proj_dim != hidden_size)
        if embed_dim != config.hidden_size:
            self.project_in = nn.Linear(embed_dim, config.hidden_size, bias=False)
        else:
            self.project_in = None
        
        # Dropout
        self.dropout = nn.Dropout(config.dropout)
        
        # Client layers
        if split_layer > 0:
            self.h = nn.ModuleList([
                OPTBlock(
                    config.hidden_size,
                    config.num_attention_heads,
                    config.ffn_dim,
                    config.max_position_embeddings,
                    config,
                    scale=True
                ) for _ in range(split_layer)
            ])
        else:
            self.h = nn.ModuleList([])

    def forward(self, input_ids, attention_mask=None, position_ids=None, past=None, len_past=None):
        """
        Forward pass through client model.
        
        Args:
            input_ids: Token IDs (batch, seq)
            attention_mask: Padding mask (batch, seq) - 1 for valid, 0 for padding
            position_ids: Position IDs (optional)
            past: Past key-values for generation
            len_past: Length of past for incremental decoding
            
        Returns:
            hidden_states: Output after client layers
            presents: Attention cache from client layers
        """
        batch_size, seq_length = input_ids.shape
        
        if past is None:
            past_length = 0
            past = [None] * len(self.h) if self.h else []
        elif len_past is None:
            past_length = past[0][0].size(-2) if past[0] is not None else 0
        else:
            # len_past is provided (can be tensor or int)
            past_length = len_past if isinstance(len_past, int) else int(len_past.item()) if hasattr(len_past, 'item') else int(len_past)
        
        # Position IDs (OPT uses LearnedPositionalEmbedding with offset=2)
        # Correct calculation matching HuggingFace OPT implementation
        if position_ids is None:
            if attention_mask is not None:
                # Create position ids from attention mask (handles left padding correctly)
                # cumsum gives positions starting from 1 for first real token
                # subtract 1 to make it 0-indexed, then add 2 for OPT offset
                # Padding positions get 1 (the padding position in OPT)
                position_ids = attention_mask.long().cumsum(-1) - 1
                position_ids.masked_fill_(attention_mask == 0, 1)  # Padding gets position 1
                
                # Handle past_length for generation
                if past_length > 0:
                    position_ids = position_ids[:, -seq_length:]
                    position_ids = position_ids + past_length
                
                position_ids = position_ids + 2  # OPT offset
            else:
                # No attention mask - simple sequential positions
                position_ids = torch.arange(
                    2 + past_length, 
                    seq_length + past_length + 2,
                    dtype=torch.long, 
                    device=input_ids.device
                )
                position_ids = position_ids.unsqueeze(0).expand(batch_size, -1)
        
        # Embeddings
        inputs_embeds = self.embed_tokens(input_ids)
        
        # Project token embeddings if needed (OPT-350M+ has different embed dim)
        if self.project_in is not None:
            inputs_embeds = self.project_in(inputs_embeds)
        
        position_embeds = self.embed_positions(position_ids)
        hidden_states = inputs_embeds + position_embeds
        hidden_states = self.dropout(hidden_states)
        
        # Process through client layers
        presents = []
        for i, (block, layer_past) in enumerate(zip(self.h, past)):
            hidden_states, present = block(hidden_states, attention_mask=attention_mask, layer_past=layer_past, len_past=len_past)
            presents.append(present)
        
        return hidden_states, presents


# =============================================================================
# OPT Model Server (Remaining Layers + Final LayerNorm)
# =============================================================================

class OPTModel_Server(nn.Module):
    """
    Server-side OPT model with remaining layers and final normalization.
    
    Like GPT2Model_Server but for OPT architecture.
    """
    def __init__(self, config, split_layer: int = 3):
        super(OPTModel_Server, self).__init__()
        
        self.config = config
        self.split_layer = split_layer
        
        server_layers = config.num_hidden_layers - split_layer
        
        # Get embedding dimension (may differ from hidden_size for OPT-350M+)
        embed_dim = getattr(config, 'word_embed_proj_dim', config.hidden_size)
        
        # For weight tying with LM head (uses word_embed_proj_dim)
        self.embed_tokens = nn.Embedding(config.vocab_size, embed_dim, padding_idx=config.pad_token_id)
        
        # Projection layer for LM head output (only for OPT-350M+)
        if embed_dim != config.hidden_size:
            self.project_out = nn.Linear(config.hidden_size, embed_dim, bias=False)
        else:
            self.project_out = None
        
        # Server layers
        if server_layers > 0:
            self.h = nn.ModuleList([
                OPTBlock(
                    config.hidden_size,
                    config.num_attention_heads,
                    config.ffn_dim,
                    config.max_position_embeddings,
                    config,
                    scale=True
                ) for _ in range(server_layers)
            ])
        else:
            self.h = nn.ModuleList([])
        
        # Final layer norm
        self.final_layer_norm = LayerNorm(config.hidden_size, eps=config.layer_norm_epsilon)

    def forward(self, hidden_states, presents, input_shape, attention_mask=None, past=None, len_past=None):
        """
        Forward pass through server model.
        
        Args:
            hidden_states: Input from client
            presents: List to append attention cache
            input_shape: Original input shape
            attention_mask: Padding mask (batch, seq) - 1 for valid, 0 for padding
            past: Past key-values for generation
            len_past: Length of past for incremental decoding
            
        Returns:
            hidden_states: Output after server layers
            presents: Complete attention cache
        """
        if past is None:
            past = [None] * len(self.h) if self.h else []
        
        if presents is None:
            presents = []
        
        # Process through server layers
        for i, (block, layer_past) in enumerate(zip(self.h, past)):
            hidden_states, present = block(hidden_states, attention_mask=attention_mask, layer_past=layer_past, len_past=len_past)
            presents.append(present)
        
        # Final layer norm
        hidden_states = self.final_layer_norm(hidden_states)
        
        # Derive shape from actual tensor dimensions (DataParallel-safe)
        # input_shape tuple doesn't get scattered, but hidden_states does
        batch_size = hidden_states.size(0)
        seq_len = hidden_states.size(1) if hidden_states.dim() == 3 else input_shape[1]
        output_shape = (batch_size, seq_len, hidden_states.size(-1))
        return hidden_states.view(*output_shape), presents


# =============================================================================
# OPT LM Head
# =============================================================================

class OPTLMHead(nn.Module):
    """Language Model Head with weight tying"""
    def __init__(self, embed_weights, config):
        super(OPTLMHead, self).__init__()
        self.hidden_size = config.hidden_size
        self.set_embeddings_weights(embed_weights)

    def set_embeddings_weights(self, embed_weights):
        embed_shape = embed_weights.shape
        self.decoder = nn.Linear(embed_shape[1], embed_shape[0], bias=False)
        self.decoder.weight = embed_weights  # Tied weights

    def forward(self, hidden_states):
        return self.decoder(hidden_states)


# =============================================================================
# OPT LM Model Client
# =============================================================================

class OPTLMModel_Client(nn.Module):
    """
    Client Language Model for split learning.
    
    INDEPENDENT module - can run on separate machine.
    """
    def __init__(self, config, split_layer: int = 3):
        super(OPTLMModel_Client, self).__init__()
        self.transformer_Client = OPTModel_Client(config, split_layer=split_layer)
        self.config = config
        self.split_layer = split_layer

    def forward(self, input_ids, attention_mask=None, past=None, len_past=None):
        hidden_states, presents = self.transformer_Client(input_ids, attention_mask=attention_mask, past=past, len_past=len_past)
        return hidden_states, presents

    def _init_weights(self, module):
        if isinstance(module, (nn.Linear, nn.Embedding)):
            module.weight.data.normal_(mean=0.0, std=0.02)
        elif isinstance(module, nn.LayerNorm):
            module.bias.data.zero_()
            module.weight.data.fill_(1.0)
        if isinstance(module, nn.Linear) and module.bias is not None:
            module.bias.data.zero_()

    def load_weight(self, state_dict):
        """Load weights from state dict"""
        if 'model_state_dict' in state_dict:
            state_dict = state_dict['model_state_dict']
        
        # Filter for client layers only
        client_state_dict = {}
        for key, value in state_dict.items():
            if key.startswith('h.'):
                parts = key.split('.')
                layer_idx = int(parts[1])
                if layer_idx < self.split_layer:
                    client_state_dict[key] = value
            else:
                client_state_dict[key] = value
        
        self.transformer_Client.load_state_dict(client_state_dict, strict=False)


# =============================================================================
# OPT LM Model Server
# =============================================================================

class OPTLMModel_Server(nn.Module):
    """
    Server Language Model for split learning.
    
    INDEPENDENT module - can run on separate machine.
    """
    def __init__(self, config, split_layer: int = 3):
        super(OPTLMModel_Server, self).__init__()
        self.transformer_Server = OPTModel_Server(config, split_layer=split_layer)
        self.lm_head = OPTLMHead(self.transformer_Server.embed_tokens.weight, config)
        self.apply(self._init_weights)
        self.config = config
        self.split_layer = split_layer

    def set_tied(self):
        """Ensure LM head shares weights with embeddings"""
        self.lm_head.set_embeddings_weights(self.transformer_Server.embed_tokens.weight)

    def forward(
        self,
        input_ids_shape,
        hidden_states_client,
        presents_client,
        lm_labels=None,
        lm_mask=None,
        attention_mask=None,
        label_smooth=0.0,
        is_report_accuracy=False
    ):
        """
        Forward pass through server.
        
        Args:
            input_ids_shape: Original input shape
            hidden_states_client: Activations from client
            presents_client: Attention cache from client
            lm_labels: Target labels for loss
            lm_mask: Mask for loss computation
            attention_mask: Padding mask (batch, seq) - 1 for valid, 0 for padding
            label_smooth: Label smoothing factor
            is_report_accuracy: Whether to compute accuracy
            
        Returns:
            logits: Language model logits
            loss: Cross-entropy loss (if labels provided)
        """
        hidden_states, presents = self.transformer_Server(
            hidden_states_client, presents_client, input_ids_shape, attention_mask=attention_mask
        )
        
        # Project out before LM head (for OPT-350M+ where word_embed_proj_dim != hidden_size)
        if self.transformer_Server.project_out is not None:
            hidden_states = self.transformer_Server.project_out(hidden_states)
        
        # LM head
        lm_logits = self.lm_head(hidden_states)
        
        # Derive actual batch/seq dims from tensors (DataParallel-safe)
        # input_ids_shape tuple doesn't get scattered, but tensors do
        _batch = lm_logits.size(0)
        _len = lm_logits.size(1)
        
        if lm_labels is not None:
            # Compute loss
            if is_report_accuracy:
                _pred_token = torch.argmax(lm_logits, dim=-1)
                _hit = (_pred_token == lm_labels) * lm_mask if lm_mask is not None else (_pred_token == lm_labels)
                
                _t1_acc = torch.zeros(_batch, dtype=torch.float, device=lm_logits.device)
                _all_acc = torch.zeros(_batch, dtype=torch.float, device=lm_logits.device)
                
                for _b in range(_batch):
                    for _i in range(_len):
                        if lm_mask is not None and lm_mask[_b, _i] >= 1.0:
                            if _hit[_b, _i] > 0:
                                _t1_acc[_b] = 1.0
                            break
                    
                    _is_succ = True
                    for _i in range(_len):
                        if lm_mask is not None and lm_mask[_b, _i] >= 1.0:
                            if _hit[_b, _i] <= 0:
                                _is_succ = False
                                break
                    if _is_succ:
                        _all_acc[_b] = 1.0
            
            if label_smooth > 0.0001:
                logprobs = F.log_softmax(lm_logits.view(-1, lm_logits.size(-1)), dim=-1)
                nll_loss = -logprobs.gather(dim=-1, index=lm_labels.view(-1).unsqueeze(1))
                nll_loss = nll_loss.squeeze(1)
                smooth_loss = -logprobs.mean(dim=-1)
                loss = (1.0 - label_smooth) * nll_loss + label_smooth * smooth_loss
                loss = loss.view(_batch, _len)
            else:
                loss_fct = nn.CrossEntropyLoss(ignore_index=-100, reduction='none')
                loss = loss_fct(lm_logits.view(-1, lm_logits.size(-1)), lm_labels.view(-1)).view(_batch, _len)
            
            if lm_mask is None:
                lm_mask = torch.ones(loss.shape, dtype=loss.dtype, device=loss.device)
            loss = loss * lm_mask
            loss = loss.sum() / (lm_mask.sum() + 0.0001)
            
            if is_report_accuracy:
                return lm_logits, loss, _t1_acc, _all_acc
            else:
                return lm_logits, loss
        
        return lm_logits, presents

    def _init_weights(self, module):
        if isinstance(module, (nn.Linear, nn.Embedding)):
            module.weight.data.normal_(mean=0.0, std=0.02)
        elif isinstance(module, nn.LayerNorm):
            module.bias.data.zero_()
            module.weight.data.fill_(1.0)
        if isinstance(module, nn.Linear) and module.bias is not None:
            module.bias.data.zero_()

    def load_weight(self, state_dict):
        """Load weights from state dict"""
        if 'model_state_dict' in state_dict:
            state_dict = state_dict['model_state_dict']
        
        # Remap layer indices for server
        new_state_dict = {}
        for key, value in state_dict.items():
            if key.startswith('h.'):
                parts = key.split('.')
                layer_idx = int(parts[1])
                if layer_idx >= self.split_layer:
                    new_key = '.'.join(['h', str(layer_idx - self.split_layer)] + parts[2:])
                    new_state_dict[new_key] = value
            else:
                new_state_dict[key] = value
        
        self.transformer_Server.load_state_dict(new_state_dict, strict=False)
        self.set_tied()


# =============================================================================
# Split Verification
# =============================================================================

def verify_split_correctness(client_module, server_module):
    """Verify client and server are independent"""
    client_param_ids = {id(p) for p in client_module.parameters()}
    server_param_ids = {id(p) for p in server_module.parameters()}
    
    shared_params = client_param_ids & server_param_ids
    if shared_params:
        return False, f"Client and server share {len(shared_params)} parameters."
    
    client_trainable = sum(1 for p in client_module.parameters() if p.requires_grad)
    server_trainable = sum(1 for p in server_module.parameters() if p.requires_grad)
    
    if client_trainable == 0:
        return False, "Client has no trainable parameters."
    if server_trainable == 0:
        return False, "Server has no trainable parameters."
    
    return True, f"Split verified: Client has {client_trainable} trainable params, Server has {server_trainable} trainable params."


# =============================================================================
# SplitOPT Main Class
# =============================================================================

class SplitOPT(nn.Module):
    """
    True Split OPT model for split learning with ZO support.
    
    Architecture (GPT2-style implementation):
    - Client: Embeddings + first N layers (configurable)
    - Server: Remaining layers + LM head
    
    Key Features:
    1. Custom attention with built-in causal mask
    2. LoRA support on Q and V projections
    3. Dynamic split layer configuration
    4. Client and Server are truly INDEPENDENT
    
    Args:
        config: OPTConfig object (or create from pretrained)
        split_layer: Layer index where to split (default: 3)
    """
    
    def __init__(self, config, split_layer: int = 3):
        super().__init__()
        
        # Validate
        assert 0 <= split_layer <= config.num_hidden_layers, \
            f"split_layer must be between 0 and {config.num_hidden_layers}"
        
        self.split_layer = split_layer
        self.config = config
        
        # Create client and server
        self.client = OPTLMModel_Client(config, split_layer=split_layer)
        self.server = OPTLMModel_Server(config, split_layer=split_layer)
        
        # Storage for split learning
        self.client_output = None
        self.server_input = None
        
        # Memory tracking
        self.client_peak_memory_mb = 0.0
        self.server_peak_memory_mb = 0.0
        self._track_memory = False
        
        # Initialize weights
        self._initialize_weights()
        
        # Log configuration
        self._log_split_info()

    def _initialize_weights(self):
        """Initialize and tie weights"""
        self.client.transformer_Client.embed_tokens.weight.data.copy_(
            self.server.transformer_Server.embed_tokens.weight.data
        )
        self.server.set_tied()

    def _log_split_info(self):
        """Log split configuration"""
        num_layers = self.config.num_hidden_layers
        logging.info(f"SplitOPT (GPT2-style) initialized:")
        logging.info(f"  - Total layers: {num_layers}")
        logging.info(f"  - Client: Embeddings + {self.split_layer} layers")
        logging.info(f"  - Server: {num_layers - self.split_layer} layers + LM head")

    def enable_memory_tracking(self, enable=True):
        self._track_memory = enable
        if enable and torch.cuda.is_available():
            torch.cuda.reset_peak_memory_stats()

    def forward(self, input_ids, labels=None, attention_mask=None, **kwargs):
        """
        Forward pass through split model.
        
        No external mask preparation needed - attention handles it internally!
        """
        # Track memory
        if self._track_memory and torch.cuda.is_available():
            torch.cuda.reset_peak_memory_stats()
        
        # ============ CLIENT SIDE ============
        hidden_states, presents = self.client(input_ids, attention_mask=attention_mask)
        
        if self._track_memory and torch.cuda.is_available():
            client_peak = torch.cuda.max_memory_allocated() / (1024 ** 2)
            self.client_peak_memory_mb = max(self.client_peak_memory_mb, client_peak)
            torch.cuda.reset_peak_memory_stats()
        
        # ============ COMMUNICATION ============
        if self.training and labels is not None:
            self.client_output = hidden_states
            self.server_input = hidden_states.clone().detach().requires_grad_(True)
            hidden_states_to_server = self.server_input
            presents_to_server = [p.clone().detach() if p is not None else None for p in presents]
        else:
            hidden_states_to_server = hidden_states
            presents_to_server = presents
        
        # ============ SERVER SIDE ============
        server_outputs = self.server(
            input_ids_shape=input_ids.shape,
            hidden_states_client=hidden_states_to_server,
            presents_client=presents_to_server if isinstance(presents_to_server, list) else presents,
            lm_labels=labels,
            lm_mask=attention_mask,
            attention_mask=attention_mask,
            label_smooth=0.0
        )
        
        if self._track_memory and torch.cuda.is_available():
            server_peak = torch.cuda.max_memory_allocated() / (1024 ** 2)
            self.server_peak_memory_mb = max(self.server_peak_memory_mb, server_peak)
        
        if labels is not None:
            logits, loss = server_outputs
            return CausalLMOutput(loss=loss, logits=logits)
        else:
            logits, _ = server_outputs
            return CausalLMOutput(logits=logits)

    @property
    def device(self):
        return next(self.parameters()).device

    def verify_split(self):
        """Verify split correctness"""
        is_valid, msg = verify_split_correctness(self.client, self.server)
        if not is_valid:
            return is_valid, msg
        
        if not hasattr(self.client, 'transformer_Client') or not hasattr(self.server, 'transformer_Server'):
            return False, "Model structure incorrect"
        
        return True, f"Model split verified. {msg}"

    def get_split_info(self):
        """Get split configuration details"""
        client_params = sum(p.numel() for p in self.client.parameters())
        server_params = sum(p.numel() for p in self.server.parameters())
        client_trainable = sum(p.numel() for p in self.client.parameters() if p.requires_grad)
        server_trainable = sum(p.numel() for p in self.server.parameters() if p.requires_grad)
        
        total_params = client_params + server_params
        
        return {
            "split_type": "layer_split",
            "split_layer": self.split_layer,
            "client_layers": self.split_layer,
            "server_layers": self.config.num_hidden_layers - self.split_layer,
            "total_layers": self.config.num_hidden_layers,
            "client_params": client_params,
            "server_params": server_params,
            "client_trainable": client_trainable,
            "server_trainable": server_trainable,
            "total_params": total_params,
            "client_percentage": client_params / total_params * 100 if total_params > 0 else 0,
            "server_percentage": server_params / total_params * 100 if total_params > 0 else 0,
        }

    def load_weight(self, model_name_or_state_dict):
        """
        Load weights from HuggingFace model or state dict.
        
        Args:
            model_name_or_state_dict: HuggingFace model name (str) or state dict
        """
        if isinstance(model_name_or_state_dict, str):
            self._load_from_huggingface(model_name_or_state_dict)
        else:
            state_dict = model_name_or_state_dict
            self.client.load_weight(state_dict)
            self.server.load_weight(state_dict)
        
        self.server.set_tied()

    def _load_from_huggingface(self, model_name):
        """Load and split HuggingFace OPT weights"""
        from transformers import OPTForCausalLM
        
        logging.info(f"Loading weights from HuggingFace: {model_name}")
        hf_model = OPTForCausalLM.from_pretrained(model_name)
        hf_state_dict = hf_model.state_dict()
        
        split_layer = self.split_layer
        client_state_dict = OrderedDict()
        server_state_dict = OrderedDict()
        
        loaded_client_layers = set()
        loaded_server_layers = set()
        
        for key, value in hf_state_dict.items():
            # Remove HF prefixes
            new_key = key
            if new_key.startswith("model.decoder."):
                new_key = new_key[len("model.decoder."):]
            
            # embed_tokens goes to both (for weight tying)
            if new_key.startswith("embed_tokens."):
                client_key = f"transformer_Client.{new_key}"
                server_key = f"transformer_Server.{new_key}"
                client_state_dict[client_key] = value.clone()
                server_state_dict[server_key] = value.clone()
            
            # embed_positions only goes to client (server doesn't need positions)
            elif new_key.startswith("embed_positions."):
                client_key = f"transformer_Client.{new_key}"
                client_state_dict[client_key] = value.clone()
            
            # project_in goes to client (for OPT-350M+)
            elif new_key.startswith("project_in."):
                client_key = f"transformer_Client.{new_key}"
                client_state_dict[client_key] = value.clone()
            
            # project_out goes to server (for OPT-350M+)
            elif new_key.startswith("project_out."):
                server_key = f"transformer_Server.{new_key}"
                server_state_dict[server_key] = value.clone()
            
            # Final layer norm goes to server
            elif new_key.startswith("final_layer_norm."):
                server_key = f"transformer_Server.{new_key}"
                server_state_dict[server_key] = value.clone()
            
            # LM head
            elif key.startswith("lm_head."):
                server_state_dict[key] = value.clone()
            
            # Split layers
            elif new_key.startswith("layers."):
                parts = new_key.split(".")
                layer_idx = int(parts[1])
                rest = ".".join(parts[2:])
                
                # Map HF layer structure to our structure
                # HF: layers.X.self_attn.{q_proj, k_proj, v_proj, out_proj}
                # HF: layers.X.self_attn_layer_norm
                # HF: layers.X.fc1, layers.X.fc2
                # HF: layers.X.final_layer_norm
                
                # Our: h.X.self_attn.{q_proj, k_proj, v_proj, out_proj}
                # Our: h.X.self_attn_layer_norm
                # Our: h.X.mlp.{fc1, fc2}
                # Our: h.X.final_layer_norm
                
                # Convert key
                if rest.startswith("fc1"):
                    mapped_rest = "mlp." + rest
                elif rest.startswith("fc2"):
                    mapped_rest = "mlp." + rest
                else:
                    mapped_rest = rest
                
                if layer_idx < split_layer:
                    client_key = f"transformer_Client.h.{layer_idx}.{mapped_rest}"
                    client_state_dict[client_key] = value.clone()
                    loaded_client_layers.add(layer_idx)
                else:
                    new_layer_idx = layer_idx - split_layer
                    server_key = f"transformer_Server.h.{new_layer_idx}.{mapped_rest}"
                    server_state_dict[server_key] = value.clone()
                    loaded_server_layers.add(layer_idx)
        
        # Load into models with verification
        missing_c, unexpected_c = self.client.load_state_dict(client_state_dict, strict=False)
        missing_s, unexpected_s = self.server.load_state_dict(server_state_dict, strict=False)
        
        # Verify embeddings loaded correctly
        embed_sum = self.client.transformer_Client.embed_tokens.weight.abs().sum().item()
        if embed_sum == 0:
            logging.error("CRITICAL: Client embeddings not loaded properly!")
        
        pos_embed_sum = self.client.transformer_Client.embed_positions.weight.abs().sum().item()
        if pos_embed_sum == 0:
            logging.error("CRITICAL: Client position embeddings not loaded properly!")
        
        # Log loading summary
        if missing_c:
            # Filter out LoRA-related keys which are expected to be missing
            real_missing = [k for k in missing_c if 'lora' not in k.lower()]
            if real_missing:
                logging.warning(f"Client missing keys (non-LoRA): {real_missing}")
        
        if missing_s:
            real_missing = [k for k in missing_s if 'lora' not in k.lower()]
            if real_missing:
                logging.warning(f"Server missing keys (non-LoRA): {real_missing}")
        
        self.server.set_tied()
        logging.info(f"Successfully loaded HuggingFace weights (split_layer={split_layer})")
        logging.info(f"  Client layers: {sorted(loaded_client_layers) if loaded_client_layers else 'embeddings only'}")
        logging.info(f"  Server layers: {sorted(loaded_server_layers)}")

    # =========================================================================
    # Serialization Helpers (for future true split deployment)
    # =========================================================================
    
    def create_forward_payload(
        self,
        hidden_states: torch.Tensor,
        presents: list,
        input_shape: tuple,
        rng_state: torch.Tensor = None,
        seed: int = None,
        batch_id: int = None,
    ) -> ForwardPayload:
        """
        Create a serializable forward payload for client -> server communication.
        
        This is an optional helper for future true split deployment.
        The existing forward() method continues to work unchanged.
        
        Args:
            hidden_states: Activations at the split point (after client layers)
            presents: Attention cache from client layers
            input_shape: Original input shape (batch_size, seq_len)
            rng_state: RNG state after client perturbation (for continuous RNG mode)
            seed: Shared perturbation seed (for shared seed mode)
            batch_id: Optional batch identifier for matching payloads
            
        Returns:
            ForwardPayload ready for serialization
        """
        return ForwardPayload(
            activations=hidden_states.detach().clone(),
            presents=[p.detach().clone() if p is not None else None for p in presents] if presents else None,
            input_shape=input_shape,
            rng_state=rng_state,
            seed=seed,
            batch_id=batch_id,
        )
    
    def create_backward_payload(
        self,
        grad_activations: torch.Tensor = None,
        loss: float = None,
        batch_id: int = None,
    ) -> BackwardPayload:
        """
        Create a serializable backward payload for server -> client communication.
        
        In FO mode: Contains gradients at the split point
        In ZO mode: Contains only the loss value
        
        Args:
            grad_activations: Gradient w.r.t. activations (FO mode)
            loss: Loss value (ZO mode or for logging)
            batch_id: Optional batch identifier for matching payloads
            
        Returns:
            BackwardPayload ready for serialization
        """
        return BackwardPayload(
            grad_activations=grad_activations.detach().clone() if grad_activations is not None else None,
            loss=float(loss) if loss is not None else None,
            batch_id=batch_id,
        )
    
    def create_zo_metadata(
        self,
        seed: int,
        rng_state: torch.Tensor = None,
        zo_eps: float = 1e-3,
        scaling_factor: int = 1,
        step_phase: str = "perturb_pos",
    ) -> ZOMetadata:
        """
        Create ZO metadata for coordinating perturbations between client and server.
        
        Args:
            seed: Shared perturbation seed
            rng_state: RNG state after client perturbation (for continuous RNG)
            zo_eps: Perturbation scale epsilon
            scaling_factor: Current perturbation direction (+1, -2, +1)
            step_phase: Current phase of ZO step
            
        Returns:
            ZOMetadata for coordination
        """
        return ZOMetadata(
            seed=seed,
            rng_state=rng_state,
            zo_eps=zo_eps,
            scaling_factor=scaling_factor,
            step_phase=step_phase,
        )
    
    def process_forward_payload(
        self,
        payload: ForwardPayload,
        requires_grad: bool = True,
    ) -> tuple:
        """
        Process a received forward payload on the server side.
        
        Converts the payload into tensors ready for server forward pass.
        
        Args:
            payload: Received ForwardPayload from client
            requires_grad: Whether activations should require gradients (for FO mode)
            
        Returns:
            Tuple of (hidden_states, presents, input_shape) for server forward
        """
        hidden_states = payload.activations.to(self.device)
        if requires_grad:
            hidden_states = hidden_states.requires_grad_(True)
        
        presents = None
        if payload.presents:
            presents = [p.to(self.device) if p is not None else None for p in payload.presents]
        
        return hidden_states, presents, payload.input_shape
