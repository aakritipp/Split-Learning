"""
Split learning model implementation for GPT-2 and OPT architectures.

This module provides split model classes that separate a transformer model into
independent client and server components. Only activations cross the network
boundary, enabling privacy-preserving distributed training.

Key components:
- GPT2LMModel_Client/Server: Split GPT-2 components
- SplitGPT2: Combined split GPT-2 model
- SplitOPT: Split OPT model using HuggingFace backend
- Payload serialization utilities for network communication
"""
import logging
import math
import os
from collections import OrderedDict 
import copy
import math
import json

import torch
from torch import nn
from torch.nn import CrossEntropyLoss, MSELoss
import torch.nn.functional as F
from torch.optim import Optimizer
from torch.optim.lr_scheduler import LambdaLR
from torch.nn.parameter import Parameter
from transformers.modeling_outputs import CausalLMOutput

import lora

# For SplitOPT we rely directly on HuggingFace OPTForCausalLM
from transformers import OPTForCausalLM

# Serialization helpers for future true split deployment
from split_communication import (
    ForwardPayload,
    BackwardPayload,
    ZOMetadata,
    serialize_payload,
    deserialize_payload,
)


def gelu(x):
    return 0.5 * x * (1 + torch.tanh(math.sqrt(2 / math.pi) * (x + 0.044715 * torch.pow(x, 3))))


def gelu_fast(x):
    return 0.5 * x * (1.0 + torch.tanh(x * 0.7978845608 * (1.0 + 0.044715 * x * x)))


def gelu_new(x):
    """ Implementation of the gelu activation function currently in Google Bert repo (identical to OpenAI GPT).
        Also see https://arxiv.org/abs/1606.08415
    """
    return 0.5 * x * (1.0 + torch.tanh(math.sqrt(2.0 / math.pi) * (x + 0.044715 * torch.pow(x, 3.0))))


def swish(x):
    return x * torch.sigmoid(x)


def _gelu_python(x):
    """ Original Implementation of the gelu activation function in Google Bert repo when initially created.
        For information: OpenAI GPT's gelu is slightly different (and gives slightly different results):
        0.5 * x * (1 + torch.tanh(math.sqrt(2 / math.pi) * (x + 0.044715 * torch.pow(x, 3))))
        This is now written in C in torch.nn.functional
        Also see https://arxiv.org/abs/1606.08415
    """
    return x * 0.5 * (1.0 + torch.erf(x / math.sqrt(2.0)))


class LayerNorm(nn.Module):
    def __init__(self, hidden_size, eps=1e-12):
        """Construct a layernorm module in the TF style (epsilon inside the square root)."""
        super(LayerNorm, self).__init__()
        self.weight = nn.Parameter(torch.ones(hidden_size))
        self.bias = nn.Parameter(torch.zeros(hidden_size))
        self.variance_epsilon = eps

    def forward(self, x):
        u = x.mean(-1, keepdim=True)
        s = (x - u).pow(2).mean(-1, keepdim=True)
        x = (x - u) / torch.sqrt(s + self.variance_epsilon)
        return self.weight * x + self.bias


class Conv1D(nn.Module):
    def __init__(self, nf, nx):
        super(Conv1D, self).__init__()
        self.nf = nf
        w = torch.empty(nx, nf)
        nn.init.normal_(w, std=0.02)
        self.weight = Parameter(w)
        self.bias = Parameter(torch.zeros(nf))

    def forward(self, x):
        size_out = x.size()[:-1] + (self.nf,)
        x = torch.addmm(self.bias, x.view(-1, x.size(-1)), self.weight)
        x = x.view(*size_out)
        return x


class Attention(nn.Module):
    def __init__(self, nx, n_ctx, config, scale=False):
        super(Attention, self).__init__()
        n_state = nx  # in Attention: n_state=768 (nx=n_embd)
        # [switch nx => n_state from Block to Attention to keep identical to TF implem]
        
        assert n_state % config.n_head == 0
        self.register_buffer("bias", torch.tril(torch.ones(n_ctx, n_ctx)).view(1, 1, n_ctx, n_ctx))
        self.n_head = config.n_head
        self.split_size = n_state
        self.scale = scale
        self.c_attn = lora.MergedLinear(
            nx, n_state * 3, 
            r=config.lora_attn_dim, 
            lora_alpha=config.lora_attn_alpha, 
            lora_dropout=config.lora_dropout, 
            enable_lora=[True, False, True], 
            fan_in_fan_out=True,
            merge_weights=False
        )
        self.c_proj = Conv1D(n_state, nx)

        self.config = config
    
    def _attn(self, q, k, v, len_kv=None):
        w = torch.matmul(q, k)
        if self.scale:
            w = w / math.sqrt(v.size(-1))
        nd, ns = w.size(-2), w.size(-1)
        b = self.bias[:, :, ns-nd:ns, :ns]
        w = w * b - 1e10 * (1 - b)

        # q : (batch, head, q_seq_length, head_features)
        # k : (batch, head, head_features, kv_seq_length)
        # w : (batch, head, q_seq_length, kv_seq_length)
        # v : (batch, head, kv_seq_length, head_features)
        if len_kv is not None:
            _len = torch.arange(k.size(-1), device=k.device)
            _input_msk =  _len[None, :] >= (len_kv)[:, None]
            w = w.masked_fill(_input_msk.unsqueeze(1).unsqueeze(2), -1.0e10) 

        w = nn.Softmax(dim=-1)(w)
        return torch.matmul(w, v)

    def merge_heads(self, x):
        x = x.permute(0, 2, 1, 3).contiguous()
        new_x_shape = x.size()[:-2] + (x.size(-2) * x.size(-1),)
        return x.view(*new_x_shape)  # in Tensorflow implem: fct merge_states

    def split_heads(self, x, k=False):
        new_x_shape = x.size()[:-1] + (self.n_head, x.size(-1) // self.n_head)
        x = x.view(*new_x_shape)  # in Tensorflow implem: fct split_states
        if k:
            return x.permute(0, 2, 3, 1).contiguous()  # (batch, head, head_features, seq_length)
        else:
            return x.permute(0, 2, 1, 3).contiguous()  # (batch, head, seq_length, head_features)

    def forward(self, x, history=None, layer_past=None, len_past=None):
        hidden_states = x

        x = self.c_attn(x)
        query, key, value = x.split(self.split_size, dim=2)

        query = self.split_heads(query)
        key = self.split_heads(key, k=True)
        value = self.split_heads(value)

        #_input_msk = None

        len_kv = None

        if layer_past is not None:
            # key : (batch, head, head_features, seq_length)
            # value : (batch, head, seq_length, head_features)
            # layer_past, key : (batch, head, seq_length, head_features)
            if len_past is None:
                past_key, past_value = layer_past[0].transpose(-2, -1), layer_past[1]  # transpose back cf below
                key = torch.cat((past_key, key), dim=-1)
                value = torch.cat((past_value, value), dim=-2)
            else:
                key_seq = key.shape[-1]
                assert key_seq == 1

                _batch = torch.arange(0, key.shape[0], dtype=torch.long, device=key.device)

                past_key, past_value = layer_past[0], layer_past[1]

                past_key[_batch,:,len_past,:] = key.squeeze(-1)
                past_value[_batch,:,len_past,:] = value.squeeze(-2)

                key = past_key.transpose(-2, -1)
                value = past_value

                len_kv = len_past + 1

        present = torch.stack((key.transpose(-2, -1), value))  # transpose to have same shapes for stacking
        a = self._attn(query, key, value, len_kv = len_kv)
        a = self.merge_heads(a)
        a = self.c_proj(a)
        return a, present


class MLP(nn.Module):
    def __init__(self, n_state, config):  # in MLP: n_state=3072 (4 * n_embd)
        super(MLP, self).__init__()
        nx = config.n_embd
        self.c_fc = Conv1D(n_state, nx)
        self.c_proj = Conv1D(nx, n_state)
        self.act = gelu

    def forward(self, x):
        h = self.act(self.c_fc(x))
        h2 = self.c_proj(h)
        return h2


class Block(nn.Module):
    def __init__(self, n_ctx, config, scale=False):
        super(Block, self).__init__()
        nx = config.n_embd
        self.ln_1 = LayerNorm(nx, eps=config.layer_norm_epsilon)
        self.attn = Attention(nx, n_ctx, config, scale)
        self.ln_2 = LayerNorm(nx, eps=config.layer_norm_epsilon)
        self.mlp = MLP(4 * nx, config)

    def forward(self, x, layer_past=None, len_past=None):
        a, present = self.attn(self.ln_1(x), layer_past=layer_past, len_past=len_past)
        x = x + a
        m = self.mlp(self.ln_2(x))
        x = x + m
        return x, present


class GPT2Model_Client(nn.Module):
    def __init__(self, config, split_layer: int = 3):
        super(GPT2Model_Client, self).__init__()
        self.n_layer = config.n_layer
        self.n_embd = config.n_embd
        self.n_vocab = config.vocab_size
        self.split_layer = split_layer  # Dynamic split point

        # Validate split layer
        assert 0 <= split_layer <= config.n_layer, \
            f"split_layer must be between 0 and {config.n_layer}, got {split_layer}"

        self.wte = nn.Embedding(config.vocab_size, config.n_embd)
        self.wpe = nn.Embedding(config.n_positions, config.n_embd)
        
        # Dynamic number of layers based on split point
        if split_layer > 0:
            block = Block(config.n_ctx, config, scale=True)
            self.h = nn.ModuleList([copy.deepcopy(block) for _ in range(split_layer)])
        else:
            self.h = nn.ModuleList([])  # No layers, embeddings only

        self.config = config

    def forward(
        self,
        input_ids,
        position_ids=None,
        token_type_ids=None,
        past=None,
        len_past=None
    ):
        if past is None:
            past_length = 0
            past = [None] * len(self.h)
        elif len_past is None:
            # equal size for past. []
            past_length = past[0][0].size(-2)

        if position_ids is None and len_past is None:
            position_ids = torch.arange(
                past_length, input_ids.size(-1) + past_length,
                dtype=torch.long, device=input_ids.device
            )
            position_ids = position_ids.unsqueeze(0).expand_as(input_ids)
        elif len_past is not None:
            position_ids = (len_past).unsqueeze(1) #.long()

        input_shape = input_ids.size()
    
        input_ids = input_ids.view(-1, input_ids.size(-1))
        position_ids = position_ids.view(-1, position_ids.size(-1))

        inputs_embeds = self.wte(input_ids)

        position_embeds = self.wpe(position_ids)

        if token_type_ids is not None:
            token_type_ids = token_type_ids.view(-1, token_type_ids.size(-1))
            token_type_embeds = self.wte(token_type_ids)
        else:
            token_type_embeds = 0
        hidden_states = inputs_embeds + position_embeds + token_type_embeds

        presents = []

        for i, (block, layer_past) in enumerate(zip(self.h, past)):
            hidden_states, present = block(hidden_states, layer_past=layer_past, len_past=len_past)
            presents.append(present)

        return hidden_states, presents

class GPT2Model_Server(nn.Module):
    def __init__(self, config, split_layer: int = 3):
        super(GPT2Model_Server, self).__init__()
        self.n_layer = config.n_layer
        self.n_embd = config.n_embd
        self.n_vocab = config.vocab_size
        self.split_layer = split_layer  # Dynamic split point
        
        # Calculate server layers
        server_layers = config.n_layer - split_layer

        self.wte = nn.Embedding(config.vocab_size, config.n_embd)
        
        # Dynamic number of layers based on split point
        if server_layers > 0:
            block = Block(config.n_ctx, config, scale=True)
            self.h = nn.ModuleList([copy.deepcopy(block) for _ in range(server_layers)])
        else:
            self.h = nn.ModuleList([])  # All layers on client
            
        self.ln_f = LayerNorm(config.n_embd, eps=config.layer_norm_epsilon)

        self.config = config


    def forward(
        self,
        hidden_states,
        presents,
        input_shape,
        past=None,
        len_past=None
    ):
        if past is None:
            past_length = 0
            past = [None] * len(self.h)
        elif len_past is None:
            past_length = past[0][0].size(-2)

        for i, (block, layer_past) in enumerate(zip(self.h, past)):
            hidden_states, present = block(hidden_states, layer_past=layer_past, len_past=len_past)
            presents.append(present)


        hidden_states = self.ln_f(hidden_states)
        
        # Derive shape from actual tensor dimensions (DataParallel-safe)
        # input_shape tuple doesn't get scattered, but hidden_states does
        batch_size = hidden_states.size(0)
        seq_len = hidden_states.size(1) if hidden_states.dim() == 3 else input_shape[1]
        output_shape = (batch_size, seq_len, hidden_states.size(-1))
        return hidden_states.view(*output_shape), presents


class GPT2LMHead(nn.Module):
    def __init__(self, model_embeddings_weights, config):
        super(GPT2LMHead, self).__init__()
        self.n_embd = config.n_embd
        self.set_embeddings_weights(model_embeddings_weights)

    def set_embeddings_weights(self, model_embeddings_weights):
        embed_shape = model_embeddings_weights.shape
        self.decoder = nn.Linear(embed_shape[1], embed_shape[0], bias=False)
        self.decoder.weight = model_embeddings_weights  # Tied weights

    def forward(self, hidden_state):
        # Truncated Language modeling logits (we remove the last token)
        # h_trunc = h[:, :-1].contiguous().view(-1, self.n_embd)
        lm_logits = self.decoder(hidden_state)
        return lm_logits


class GPT2Config(object):
    def __init__(
        self,
        vocab_size_or_config_json_file=50257,
        n_positions=1024,
        n_ctx=1024,
        n_embd=768,
        n_layer=12,
        n_head=12,
        layer_norm_epsilon=1e-5,
        initializer_range=0.02,
        lora_attn_dim=0,
        lora_attn_alpha=128,
        lora_dropout=0.0,
        lora_r_dropout=0.0,
        fix_dropout=0.0,
    ):
        self.vocab_size = vocab_size_or_config_json_file
        self.n_ctx = n_ctx
        self.n_positions = n_positions
        self.n_embd = n_embd
        self.n_layer = n_layer
        self.n_head = n_head
        self.layer_norm_epsilon = layer_norm_epsilon
        self.initializer_range = initializer_range
        self.lora_attn_dim = lora_attn_dim
        self.lora_attn_alpha = lora_attn_alpha
        self.lora_dropout = lora_dropout
        self.lora_r_dropout = lora_r_dropout

        self.fix_dropout = fix_dropout

        # Provide HF-like aliases so shared utilities (e.g. PrefixTuning) can
        # treat this config similarly to a standard GPT-2 config.
        self.hidden_size = n_embd
        self.num_attention_heads = n_head
        # Distinguish from HF "gpt2" but still recognizable as GPT-style.
        self.model_type = "gpt2_split"

    # ---- Minimal HF-style helpers so Trainer callbacks & integrations work ----
    def to_dict(self):
        """
        Return a JSON-serializable dictionary representation of this config.

        This mirrors the behavior of `transformers.PretrainedConfig.to_dict`
        well enough for logging/callback purposes.
        """
        output = {}
        for k, v in self.__dict__.items():
            if k.startswith("_"):
                continue
            # Best-effort conversion for tensors etc.
            if isinstance(v, torch.dtype):
                output[k] = str(v)
            else:
                try:
                    json.dumps(v)
                    output[k] = v
                except TypeError:
                    output[k] = str(v)
        # Include class name for easier debugging
        output["__class__"] = self.__class__.__name__
        return output

    def to_json_string(self, use_diff: bool = True) -> str:
        """
        HuggingFace integrations expect `config.to_json_string()`.
        We implement a lightweight version that just dumps `to_dict()`.
        """
        return json.dumps(self.to_dict(), indent=2, sort_keys=True) + "\n"



class GPT2LMModel_Client(nn.Module):
    """
    Client model for true split learning.
    
    This module is INDEPENDENT and can run on a separate machine.
    It only communicates via:
    - Input: input_ids
    - Output: hidden_states, presents (to be sent to server)
    
    Args:
        config: GPT2Config object
        split_layer: Layer index where to split (default: 3)
    """
    def __init__(self, config, split_layer: int = 3):
        super(GPT2LMModel_Client, self).__init__()
        self.transformer_Client = GPT2Model_Client(config, split_layer=split_layer)
        self.config = config
        self.split_layer = split_layer

    def set_tied(self):
        """ Make sure we are sharing the embeddings"""
        self.lm_head.set_embeddings_weights(self.transformer_Client.wte.weight)

    def forward(
            self,
            input_ids,
            past=None,
            len_past=None,
    ):
        _batch, _len = input_ids.shape
        hidden_states_client, presents_client = self.transformer_Client(input_ids, past=past,len_past=len_past)
        
        # Return hidden states and presents (to be sent to server)
        # Note: We do NOT return state_dict - that would violate split learning privacy
        return hidden_states_client, presents_client

    def _init_weights(self, module):
        if isinstance(module, (nn.Linear, nn.Embedding)):
            module.weight.data.normal_(mean=0.0, std=0.02)
        elif isinstance(module, nn.LayerNorm):
            module.bias.data.zero_()
            module.weight.data.fill_(1.0)
        if isinstance(module, nn.Linear) and module.bias is not None:
            module.bias.data.zero_()

    def load_weight(self, state_dict):
        if 'model_state_dict' in state_dict:
            state_dict = state_dict['model_state_dict']

        state_dict_tmp = copy.deepcopy(state_dict)
        old_keys = []
        new_keys = []
        for key in state_dict_tmp:
            new_key = None
            if key.endswith(".g"):
                new_key = key[:-2] + ".weight"
            elif key.endswith(".b"):
                new_key = key[:-2] + ".bias"
            elif key.endswith(".w"):
                new_key = key[:-2] + ".weight"

            if key.startswith("module.transformer."):
                new_key = key[len("module.transformer."):]

            if new_key:
                old_keys.append(key)
                new_keys.append(new_key)

        for old_key, new_key in zip(old_keys, new_keys):
            state_dict[new_key] = state_dict.pop(old_key)

        for n, p in self.transformer_Client.named_parameters():
            if n not in state_dict:
                state_dict[n] = p

        self.transformer_Client.load_state_dict(state_dict, strict=False)



class GPT2LMModel_Server(nn.Module):
    """
    Server model for true split learning.
    
    This module is INDEPENDENT and can run on a separate machine.
    It only communicates via:
    - Input: hidden_states, presents (from client)
    - Output: loss, gradients for hidden_states (to be sent back to client)
    
    Args:
        config: GPT2Config object
        split_layer: Layer index where to split (default: 3)
    """
    def __init__(self, config, split_layer: int = 3):
        super(GPT2LMModel_Server, self).__init__()

        self.transformer_Server = GPT2Model_Server(config, split_layer=split_layer)
        self.lm_head = GPT2LMHead(self.transformer_Server.wte.weight, config)
        self.apply(self._init_weights)
        self.config = config
        self.split_layer = split_layer

    def set_tied(self):
        """ Make sure we are sharing the embeddings"""
        self.lm_head.set_embeddings_weights(self.transformer_Server.wte.weight)

    def forward(
            self,
            input_ids_shape,
            hidden_states_client,
            presents_client,
            lm_labels=None,
            lm_mask=None,
            label_smooth=0.0,
            is_report_accuracy=False
    ):
        hidden_states_server, presents_server = self.transformer_Server(hidden_states_client,
                                                                                    presents_client, input_ids_shape)

        # batch, seq, vocab
        lm_logits = self.lm_head(hidden_states_server)
        
        # Derive actual batch/seq dims from tensors (DataParallel-safe)
        # input_ids_shape tuple doesn't get scattered, but tensors do
        _batch = lm_logits.size(0)
        _len = lm_logits.size(1)

        if lm_labels is not None:

            if is_report_accuracy:
                _pred_token = torch.argmax(lm_logits, dim=-1)
                _hit = (_pred_token == lm_labels) * lm_mask

                _t1_acc = torch.zeros(_batch, dtype=torch.float, device=lm_logits.device)
                _all_acc = torch.zeros(_batch, dtype=torch.float, device=lm_logits.device)

                for _b in range(0, _batch):
                    for _i in range(0, _len):
                        if lm_mask[_b, _i] >= 1.0:
                            if _hit[_b, _i] > 0:
                                _t1_acc[_b] = 1.0
                            break

                    _is_succ = True
                    for _i in range(0, _len):
                        if lm_mask[_b, _i] >= 1.0:
                            if _hit[_b, _i] <= 0:
                                _is_succ = False
                                break

                    if _is_succ:
                        _all_acc[_b] = 1.0

                # _t1_acc = _t1_acc * 1.0 / _batch
                # _all_acc = _all_acc * 1.0 / _batch

            if label_smooth > 0.0001:
                logprobs = torch.nn.functional.log_softmax(lm_logits.view(-1, lm_logits.size(-1)), dim=-1)
                nll_loss = -logprobs.gather(dim=-1, index=lm_labels.view(-1).unsqueeze(1))
                nll_loss = nll_loss.squeeze(1)
                smooth_loss = -logprobs.mean(dim=-1)
                loss = (1.0 - label_smooth) * nll_loss + label_smooth * smooth_loss
                loss = loss.view(_batch, _len)
            else:
                loss_fct = nn.CrossEntropyLoss(ignore_index=-1, reduce=False)
                loss = loss_fct(lm_logits.view(-1, lm_logits.size(-1)), lm_labels.view(-1)).view(_batch, _len)

            if lm_mask is None:
                lm_mask = torch.ones(loss.shape, dtype=loss.dtype, device=loss.device)
            loss = loss * lm_mask

            loss = loss.sum() / (lm_mask.sum() + 0.0001)

            if is_report_accuracy:
                return lm_logits, loss, _t1_acc, _all_acc
            else:
                return lm_logits, loss
        return lm_logits, presents_server

    def _init_weights(self, module):
        if isinstance(module, (nn.Linear, nn.Embedding)):
            module.weight.data.normal_(mean=0.0, std=0.02)
        elif isinstance(module, nn.LayerNorm):
            module.bias.data.zero_()
            module.weight.data.fill_(1.0)
        if isinstance(module, nn.Linear) and module.bias is not None:
            module.bias.data.zero_()

    def load_weight(self, state_dict):
        if 'model_state_dict' in state_dict:
            state_dict = state_dict['model_state_dict']

        state_dict_tmp = copy.deepcopy(state_dict)
        old_keys = []
        new_keys = []
        for key in state_dict_tmp:
            new_key = None
            if key.endswith(".g"):
                new_key = key[:-2] + ".weight"
            elif key.endswith(".b"):
                new_key = key[:-2] + ".bias"
            elif key.endswith(".w"):
                new_key = key[:-2] + ".weight"

            if key.startswith("module.transformer."):
                new_key = key[len("module.transformer."):]

            if new_key:
                old_keys.append(key)
                new_keys.append(new_key)

        for old_key, new_key in zip(old_keys, new_keys):
            state_dict[new_key] = state_dict.pop(old_key)

        for n, p in self.transformer_Server.named_parameters():
            if n not in state_dict:
                state_dict[n] = p

        new_state_dict = {}
        for key, value in state_dict.items():
            if key.startswith('h.'):
                parts = key.split('.')
                layer_idx = int(parts[1])
                # Use dynamic split_layer instead of hardcoded 3
                new_key = '.'.join(['h', str(layer_idx - self.split_layer)] + parts[2:])
                new_state_dict[new_key] = value
            else:
                new_state_dict[key] = value

        self.transformer_Server.load_state_dict(new_state_dict, strict=False)
        self.set_tied()


def verify_split_correctness(client_module, server_module):
    """
    Verify that the split model is correctly configured.
    
    Checks:
    1. Client and server are independent modules (no shared parameters)
    2. Both have trainable parameters
    
    Returns:
        Tuple of (is_valid, error_message)
    """
    # Check for shared parameters between client and server
    client_param_ids = {id(p) for p in client_module.parameters()}
    server_param_ids = {id(p) for p in server_module.parameters()}
    
    shared_params = client_param_ids & server_param_ids
    if shared_params:
        return False, f"Client and server share {len(shared_params)} parameters. They must be independent."
    
    # Check trainable parameters exist
    client_trainable = sum(1 for p in client_module.parameters() if p.requires_grad)
    server_trainable = sum(1 for p in server_module.parameters() if p.requires_grad)
    
    if client_trainable == 0:
        return False, "Client has no trainable parameters."
    if server_trainable == 0:
        return False, "Server has no trainable parameters."
    
    return True, f"Split verified: Client has {client_trainable} trainable params, Server has {server_trainable} trainable params."

        
class SplitGPT2(nn.Module):
    """
    True Split GPT-2 model for split learning with ZO support.
    
    Architecture:
    - Client: Contains the first N layers (0 to split_layer-1) and embeddings
    - Server: Contains the remaining layers (split_layer to n_layer-1) and LM head
    
    Key Features:
    1. Client and Server are truly INDEPENDENT (can run on separate machines)
    2. Only activations cross the network boundary (forward: h, backward: ∂L/∂h)
    3. For ZO: Perturbation seed AND RNG state are shared for coordinated ZO
    4. Split point is configurable via split_layer parameter
    
    Args:
        config: GPT2Config object with model hyperparameters
        split_layer: Layer index where to split (default: 3)
    """
    def __init__(self, config, split_layer: int = 3):
        super().__init__()
        
        # Validate split layer
        assert 0 <= split_layer <= config.n_layer, \
            f"split_layer must be between 0 and {config.n_layer}, got {split_layer}"
        
        self.split_layer = split_layer
        self.client = GPT2LMModel_Client(config, split_layer=split_layer)
        self.server = GPT2LMModel_Server(config, split_layer=split_layer)
        self.config = config
        
        # Storage for split learning gradients and activations
        self.client_output = None  # Activation at the cut on the client side
        self.server_input = None   # Detached clone that server sees
        
        # Peak memory tracking for client and server
        self.client_peak_memory_mb = 0.0
        self.server_peak_memory_mb = 0.0
        self._track_memory = False
        
        # Weight tying (only needed during initialization, then they're independent)
        self._initialize_weights()

    def enable_memory_tracking(self, enable=True):
        """Enable/disable memory tracking during forward passes."""
        self._track_memory = enable
        if enable and torch.cuda.is_available():
            torch.cuda.reset_peak_memory_stats()

    def _initialize_weights(self):
        """Initialize weights for client and server (they start the same, then diverge during training)"""
        # Copy embedding weights from server to client for initialization
        # Note: After this, they are INDEPENDENT copies
        self.client.transformer_Client.wte.weight.data.copy_(
            self.server.transformer_Server.wte.weight.data
        )
        # Ensure server LM head is tied to server embeddings
        self.server.set_tied()

    def forward(self, input_ids, labels=None, attention_mask=None, **kwargs):
        """
        Forward pass through split model.
        
        True split learning protocol:
        1. Client processes input through layers 0-2
        2. Client sends (hidden_states, presents) to server
        3. Server processes through layers 3-11 and computes loss
        4. For backward: Server sends gradients back to client
        
        Args:
            input_ids: Input token IDs (batch_size, seq_len)
            labels: Target labels for training (batch_size, seq_len)
            attention_mask: Attention mask (batch_size, seq_len)
            **kwargs: Additional arguments
        
        Returns:
            CausalLMOutput with loss (if labels provided) and logits
        """
        # Track memory before client forward
        if self._track_memory and torch.cuda.is_available():
            torch.cuda.reset_peak_memory_stats()
        
        # ============ CLIENT SIDE ============
        # Client forward - this would be on client machine
        hidden_states, presents = self.client(input_ids)
        
        # Track peak memory after client forward
        if self._track_memory and torch.cuda.is_available():
            client_peak = torch.cuda.max_memory_allocated() / (1024 ** 2)
            self.client_peak_memory_mb = max(self.client_peak_memory_mb, client_peak)
            torch.cuda.reset_peak_memory_stats()
        
        # ============ COMMUNICATION ============
        # In true split learning, only these tensors cross the network:
        # - hidden_states: activation at the split point
        # - presents: attention cache (if needed)
        # - perturbation_seed: for ZO coordination
        
        if self.training and labels is not None:
            # Store client output for backward pass
            self.client_output = hidden_states
            
            # Create detached copy for server (simulates network transmission)
            # The server receives this without connection to client's graph
            self.server_input = hidden_states.clone().detach().requires_grad_(True)
            hidden_states_to_server = self.server_input
            
            # Detach presents too (they should not carry gradients through network)
            presents_to_server = [p.clone().detach() if p is not None else None for p in presents]
        else:
            hidden_states_to_server = hidden_states
            presents_to_server = presents
        
        # ============ SERVER SIDE ============
        # Server forward - this would be on server machine
        server_outputs = self.server(
            input_ids_shape=input_ids.shape,
            hidden_states_client=hidden_states_to_server,
            presents_client=presents_to_server if isinstance(presents_to_server, list) else presents,
            lm_labels=labels,
            lm_mask=attention_mask,
            label_smooth=0.0
        )
        
        # Track peak memory after server forward
        if self._track_memory and torch.cuda.is_available():
            server_peak = torch.cuda.max_memory_allocated() / (1024 ** 2)
            self.server_peak_memory_mb = max(self.server_peak_memory_mb, server_peak)
        
        if labels is not None:
            logits, loss = server_outputs
            return CausalLMOutput(loss=loss, logits=logits)
        else:
            logits, _ = server_outputs
            return CausalLMOutput(logits=logits)

    def load_weight(self, state_dict_or_model_name):
        """
        Load weights into the split model.
        
        Supports two modes:
        1. From a state_dict (original format) - for loading checkpoints
        2. From HuggingFace model name (e.g., "gpt2") - for pretrained weights
        
        Args:
            state_dict_or_model_name: Either a state dict or HuggingFace model name
        
        Note: Uses self.split_layer that was set during __init__
        """
        if isinstance(state_dict_or_model_name, str):
            # Load from HuggingFace
            logging.info(f"Loading pretrained weights from HuggingFace: {state_dict_or_model_name}")
            self._load_from_huggingface(state_dict_or_model_name)
        else:
            # Load from state dict (original format)
            state_dict = state_dict_or_model_name
            self.client.load_weight(state_dict)
            self.server.load_weight(state_dict)
        
        # Re-tie server embeddings (but NOT client-server - they're independent)
        self.server.set_tied()
    
    def _load_from_huggingface(self, model_name):
        """Load and split HuggingFace GPT-2 weights using self.split_layer"""
        from transformers import GPT2LMHeadModel
        from collections import OrderedDict
        
        # Load full model from HuggingFace
        hf_model = GPT2LMHeadModel.from_pretrained(model_name)
        state_dict = hf_model.state_dict()
        
        split_layer = self.split_layer  # Use instance's split_layer
        
        client_state_dict = OrderedDict()
        server_state_dict = OrderedDict()
        
        for key, value in state_dict.items():
            # Remove "transformer." prefix
            if key.startswith("transformer."):
                key_without_prefix = key[len("transformer."):]
            else:
                key_without_prefix = key
            
            # Embeddings: each side gets its own copy (independent)
            if key_without_prefix.startswith("wte.") or key_without_prefix.startswith("wpe."):
                client_key = f"transformer_Client.{key_without_prefix}"
                server_key = f"transformer_Server.{key_without_prefix}"
                client_state_dict[client_key] = value.clone()
                server_state_dict[server_key] = value.clone()
            
            # Layer norm and LM head go to server
            elif key_without_prefix.startswith("ln_f."):
                server_key = f"transformer_Server.{key_without_prefix}"
                server_state_dict[server_key] = value.clone()
            elif key.startswith("lm_head."):
                server_state_dict[key] = value.clone()
            
            # Split transformer layers based on self.split_layer
            elif key_without_prefix.startswith("h."):
                parts = key_without_prefix.split(".")
                layer_idx = int(parts[1])
                rest_of_key = ".".join(parts[2:])
                
                if layer_idx < split_layer:
                    # Client layers
                    client_key = f"transformer_Client.h.{layer_idx}.{rest_of_key}"
                    client_state_dict[client_key] = value.clone()
                else:
                    # Server layers (adjust index)
                    new_layer_idx = layer_idx - split_layer
                    server_key = f"transformer_Server.h.{new_layer_idx}.{rest_of_key}"
                    server_state_dict[server_key] = value.clone()
        
        # Load into models
        missing_keys, unexpected_keys = self.client.load_state_dict(client_state_dict, strict=False)
        if missing_keys:
            logging.warning(f"Missing keys in client: {missing_keys}")
        
        missing_keys, unexpected_keys = self.server.load_state_dict(server_state_dict, strict=False)
        if missing_keys:
            logging.warning(f"Missing keys in server: {missing_keys}")
        
        # Set tied weights on server only
        self.server.set_tied()
        logging.info(f"Successfully loaded HuggingFace weights into split model (split_layer={split_layer})")

    @property
    def device(self):
        return next(self.parameters()).device
    
    def verify_split(self):
        """
        Verify that the model is correctly split for split learning.
        
        This checks:
        1. Client and server are independent (no shared parameters)
        2. Both have trainable parameters
        
        Returns:
            Tuple of (is_valid, message)
        """
        is_valid, msg = verify_split_correctness(self.client, self.server)
        if not is_valid:
            return is_valid, msg
        
        # Additional check: verify the split point creates proper forward flow
        if not hasattr(self.client, 'transformer_Client') or not hasattr(self.server, 'transformer_Server'):
            return False, "Model structure incorrect: missing transformer_Client or transformer_Server"
        
        return True, f"Model split verified. {msg}"
    
    def get_split_info(self):
        """
        Get information about the split configuration.
        
        Returns:
            Dict with split configuration details
        """
        client_params = sum(p.numel() for p in self.client.parameters())
        server_params = sum(p.numel() for p in self.server.parameters())
        client_trainable = sum(p.numel() for p in self.client.parameters() if p.requires_grad)
        server_trainable = sum(p.numel() for p in self.server.parameters() if p.requires_grad)
        
        total_params = client_params + server_params
        
        return {
            "split_type": "layer_split",
            "split_layer": self.split_layer,
            "client_layers": self.split_layer,
            "server_layers": self.config.n_layer - self.split_layer,
            "total_layers": self.config.n_layer,
            "client_params": client_params,
            "server_params": server_params,
            "client_trainable": client_trainable,
            "server_trainable": server_trainable,
            "total_params": total_params,
            "client_percentage": client_params / total_params * 100 if total_params > 0 else 0,
            "server_percentage": server_params / total_params * 100 if total_params > 0 else 0,
        }

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
            hidden_states: Activations at the split point
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
        

class OPTLMModel_Client(nn.Module):
    """
    Client for split OPT - computes embeddings + first N decoder layers.
    
    INDEPENDENT module that can run on a separate machine.
    Returns (hidden_states, presents) for transmission to server.
    
    Architecture (default split_layer=3):
      - Token embeddings
      - Position embeddings  
      - First 3 decoder layers (layers 0, 1, 2)
    """

    def __init__(self, config, client_layers: nn.ModuleList, embed_tokens: nn.Embedding, 
                 embed_positions: nn.Module, project_in: nn.Module = None,
                 split_layer: int = 3):
        super().__init__()
        self.config = config
        self.split_layer = split_layer
        
        # Embedding layers (independent copies)
        self.embed_tokens = embed_tokens
        self.embed_positions = embed_positions
        self.project_in = project_in  # Only for larger OPT models
        
        # First N decoder layers
        self.layers = client_layers
        
        # Dropout (matching OPT's dropout)
        self.dropout = nn.Dropout(config.dropout if hasattr(config, 'dropout') else 0.1)

    def forward(
        self,
        input_ids,
        attention_mask=None,
        position_ids=None,
        past_key_values=None,
        use_cache=False,
    ):
        """
        Forward pass through client layers.
        
        Args:
            input_ids: Input token IDs (batch_size, seq_len)
            attention_mask: Attention mask (batch_size, seq_len)
            position_ids: Position IDs (optional)
            past_key_values: Past key-value cache (for generation)
            use_cache: Whether to return past key-values
            
        Returns:
            hidden_states: Activations after client layers
            presents: Attention cache from client layers (if use_cache=True)
        """
        # Get input shape
        input_shape = input_ids.size()
        batch_size, seq_length = input_shape
        
        # Create position IDs if not provided
        if position_ids is None:
            # OPT uses offset of 2 for position embeddings
            past_key_values_length = 0
            if past_key_values is not None:
                past_key_values_length = past_key_values[0][0].shape[2]
            position_ids = torch.arange(
                past_key_values_length, 
                seq_length + past_key_values_length,
                dtype=torch.long, 
                device=input_ids.device
            )
            position_ids = position_ids.unsqueeze(0).expand(batch_size, -1)
        
        # Get embeddings
        inputs_embeds = self.embed_tokens(input_ids)
        
        # OPT-specific: position embeddings with offset
        pos_embeds = self.embed_positions(attention_mask, past_key_values_length=0)
        
        hidden_states = inputs_embeds + pos_embeds
        
        # Project in (only for larger models where word_embed_proj_dim != hidden_size)
        if self.project_in is not None:
            hidden_states = self.project_in(hidden_states)
        
        hidden_states = self.dropout(hidden_states)
        
        # Prepare attention mask for decoder layers
        # OPT uses causal mask
        if attention_mask is None:
            attention_mask = torch.ones(batch_size, seq_length, device=input_ids.device)
        
        # Create causal mask
        causal_mask = self._prepare_decoder_attention_mask(
            attention_mask, input_shape, inputs_embeds, past_key_values_length=0
        )
        
        # Process through client decoder layers
        presents = [] if use_cache else None
        
        for idx, decoder_layer in enumerate(self.layers):
            layer_past = past_key_values[idx] if past_key_values is not None else None
            
            layer_outputs = decoder_layer(
                hidden_states,
                attention_mask=causal_mask,
                layer_head_mask=None,
                past_key_value=layer_past,
                output_attentions=False,
                use_cache=use_cache,
            )
            
            hidden_states = layer_outputs[0]
            
            if use_cache:
                presents.append(layer_outputs[1])
        
        return hidden_states, presents
    
    def _prepare_decoder_attention_mask(self, attention_mask, input_shape, inputs_embeds, past_key_values_length):
        """Prepare causal attention mask for OPT decoder."""
        # Create causal mask: [batch_size, 1, seq_len, seq_len]
        batch_size, seq_length = input_shape
        
        # Create causal mask
        mask = torch.triu(
            torch.ones((seq_length, seq_length), device=inputs_embeds.device) * float('-inf'),
            diagonal=1
        )
        mask = mask.unsqueeze(0).unsqueeze(0)  # [1, 1, seq_len, seq_len]
        
        # Expand to batch size
        mask = mask.expand(batch_size, 1, seq_length, seq_length)
        
        # Apply attention mask if provided
        if attention_mask is not None:
            # Expand attention mask: [batch_size, seq_len] -> [batch_size, 1, 1, seq_len]
            expanded_attn_mask = attention_mask.unsqueeze(1).unsqueeze(2)
            expanded_attn_mask = (1.0 - expanded_attn_mask) * float('-inf')
            mask = mask + expanded_attn_mask
        
        return mask


class OPTLMModel_Server(nn.Module):
    """
    Server for split OPT - computes remaining decoder layers + LM head.
    
    INDEPENDENT module that can run on a separate machine.
    Receives hidden_states from client and computes loss.
    
    Architecture (default split_layer=3, for 12-layer OPT):
      - Decoder layers 3-11 (9 layers)
      - Final layer norm
      - LM head
    """

    def __init__(self, config, server_layers: nn.ModuleList, final_layer_norm: nn.Module,
                 lm_head: nn.Module, project_out: nn.Module = None, split_layer: int = 3):
        super().__init__()
        self.config = config
        self.split_layer = split_layer
        
        # Remaining decoder layers
        self.layers = server_layers
        
        # Final layer norm
        self.final_layer_norm = final_layer_norm
        
        # Project out (only for larger models)
        self.project_out = project_out
        
        # LM head for language modeling
        self.lm_head = lm_head
        
        # Dropout
        self.dropout = nn.Dropout(config.dropout if hasattr(config, 'dropout') else 0.1)

    def forward(
        self,
        input_ids_shape,
        hidden_states_client,
        attention_mask=None,
        presents_client=None,
        lm_labels=None,
        lm_mask=None,
        label_smooth: float = 0.0,
        is_report_accuracy: bool = False,
        use_cache: bool = False,
    ):
        """
        Forward pass through server layers.
        
        Args:
            input_ids_shape: Original input shape (batch_size, seq_len)
            hidden_states_client: Hidden states from client
            attention_mask: Attention mask
            presents_client: Past key-values from client layers
            lm_labels: Labels for loss computation
            lm_mask: Mask for loss computation
            label_smooth: Label smoothing factor
            is_report_accuracy: Whether to report accuracy metrics
            use_cache: Whether to return past key-values
            
        Returns:
            logits: Language model logits
            loss: Cross-entropy loss (if lm_labels provided)
        """
        # Derive actual batch/seq dims from tensors (DataParallel-safe)
        # input_ids_shape tuple doesn't get scattered, but tensors do
        batch_size = hidden_states_client.size(0)
        seq_length = hidden_states_client.size(1)
        hidden_states = hidden_states_client
        
        # Use lm_mask as attention_mask if provided
        if attention_mask is None and lm_mask is not None:
            attention_mask = lm_mask
        
        # Prepare causal attention mask
        if attention_mask is None:
            attention_mask = torch.ones(batch_size, seq_length, device=hidden_states.device)
        
        causal_mask = self._prepare_decoder_attention_mask(
            attention_mask, (batch_size, seq_length), hidden_states, past_key_values_length=0
        )
        
        # Process through server decoder layers
        presents = [] if use_cache else None
        
        for idx, decoder_layer in enumerate(self.layers):
            # Get past key-value from client if available
            layer_past = None
            if presents_client is not None and idx < len(presents_client):
                layer_past = presents_client[idx]
            
            layer_outputs = decoder_layer(
                hidden_states,
                attention_mask=causal_mask,
                layer_head_mask=None,
                past_key_value=layer_past,
                output_attentions=False,
                use_cache=use_cache,
            )
            
            hidden_states = layer_outputs[0]
            
            if use_cache:
                presents.append(layer_outputs[1])
        
        # Final layer norm
        hidden_states = self.final_layer_norm(hidden_states)
        
        # Project out (only for larger models)
        if self.project_out is not None:
            hidden_states = self.project_out(hidden_states)
        
        # LM head
        logits = self.lm_head(hidden_states)
        
        # Compute loss if labels provided
        if lm_labels is not None:
            # Shift for causal LM loss
            shift_logits = logits[..., :-1, :].contiguous()
            shift_labels = lm_labels[..., 1:].contiguous()
            
            if label_smooth > 0.0001:
                # Label smoothing
                logprobs = F.log_softmax(shift_logits.view(-1, shift_logits.size(-1)), dim=-1)
                nll_loss = -logprobs.gather(dim=-1, index=shift_labels.view(-1).unsqueeze(1))
                nll_loss = nll_loss.squeeze(1)
                smooth_loss = -logprobs.mean(dim=-1)
                loss = (1.0 - label_smooth) * nll_loss + label_smooth * smooth_loss
                loss = loss.view(batch_size, -1)
                
                if lm_mask is not None:
                    shift_mask = lm_mask[..., 1:].contiguous()
                    loss = loss * shift_mask
                    loss = loss.sum() / (shift_mask.sum() + 1e-8)
                else:
                    loss = loss.mean()
            else:
                # Standard cross-entropy
                loss_fct = nn.CrossEntropyLoss(ignore_index=-100)
                loss = loss_fct(shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1))
            
            if is_report_accuracy:
                # Compute accuracy metrics
                pred_tokens = torch.argmax(shift_logits, dim=-1)
                correct = (pred_tokens == shift_labels).float()
                if lm_mask is not None:
                    shift_mask = lm_mask[..., 1:].contiguous()
                    correct = correct * shift_mask
                    accuracy = correct.sum() / (shift_mask.sum() + 1e-8)
                else:
                    accuracy = correct.mean()
                return logits, loss, accuracy
            
            return logits, loss
        
        return logits, presents
    
    def _prepare_decoder_attention_mask(self, attention_mask, input_shape, inputs_embeds, past_key_values_length):
        """Prepare causal attention mask for OPT decoder."""
        batch_size, seq_length = input_shape
        
        # Create causal mask
        mask = torch.triu(
            torch.ones((seq_length, seq_length), device=inputs_embeds.device) * float('-inf'),
            diagonal=1
        )
        mask = mask.unsqueeze(0).unsqueeze(0)
        mask = mask.expand(batch_size, 1, seq_length, seq_length)
        
        # Apply attention mask if provided
        if attention_mask is not None:
            expanded_attn_mask = attention_mask.unsqueeze(1).unsqueeze(2)
            expanded_attn_mask = (1.0 - expanded_attn_mask) * float('-inf')
            mask = mask + expanded_attn_mask
        
        return mask


class SplitOPT(nn.Module):
    """
    True Split OPT model for split learning with ZO support.
    
    Architecture (default split_layer=3):
      - Client: Embeddings + position embeddings + first 3 decoder layers
      - Server: Remaining decoder layers (e.g., 9 for OPT-125M) + LM head
      
    Key features:
      - Client and server are truly INDEPENDENT modules (can run on separate machines)
      - Only activations cross the network boundary
      - Split point is configurable via split_layer parameter
      
    Args:
        model_name: HuggingFace model name (e.g., "facebook/opt-125m")
        split_layer: Layer index where to split (default: 3)
    """

    def __init__(self, model_name: str, split_layer: int = 3):
        super().__init__()
        
        # Load full OPT model from HuggingFace
        opt_model = OPTForCausalLM.from_pretrained(model_name)
        self.config = opt_model.config
        self.split_layer = split_layer
        
        # Get total number of layers
        num_layers = self.config.num_hidden_layers
        assert 0 <= split_layer <= num_layers, \
            f"split_layer must be between 0 and {num_layers}, got {split_layer}"
        
        decoder = opt_model.model.decoder
        
        # =========== BUILD CLIENT ===========
        # Create INDEPENDENT copies of embedding layers
        embed_tokens = nn.Embedding(
            decoder.embed_tokens.num_embeddings,
            decoder.embed_tokens.embedding_dim,
            padding_idx=decoder.embed_tokens.padding_idx,
        )
        embed_tokens.weight.data.copy_(decoder.embed_tokens.weight.data)
        
        # Copy position embeddings (OPTLearnedPositionalEmbedding)
        embed_positions = copy.deepcopy(decoder.embed_positions)
        
        # Project in (only for larger models)
        project_in = copy.deepcopy(decoder.project_in) if decoder.project_in is not None else None
        
        # Copy first N decoder layers for client
        client_layers = nn.ModuleList([
            copy.deepcopy(decoder.layers[i]) for i in range(split_layer)
        ])
        
        self.client = OPTLMModel_Client(
            config=self.config,
            client_layers=client_layers,
            embed_tokens=embed_tokens,
            embed_positions=embed_positions,
            project_in=project_in,
            split_layer=split_layer,
        )
        
        # =========== BUILD SERVER ===========
        # Copy remaining decoder layers for server
        server_layers = nn.ModuleList([
            copy.deepcopy(decoder.layers[i]) for i in range(split_layer, num_layers)
        ])
        
        # Copy final layer norm
        final_layer_norm = copy.deepcopy(decoder.final_layer_norm)
        
        # Project out (only for larger models)
        project_out = copy.deepcopy(decoder.project_out) if decoder.project_out is not None else None
        
        # Copy LM head
        lm_head = copy.deepcopy(opt_model.lm_head)
        
        self.server = OPTLMModel_Server(
            config=self.config,
            server_layers=server_layers,
            final_layer_norm=final_layer_norm,
            lm_head=lm_head,
            project_out=project_out,
            split_layer=split_layer,
        )
        
        # Storage for split-learning tensors
        self.client_output = None
        self.server_input = None
        
        # Peak memory tracking
        self.client_peak_memory_mb = 0.0
        self.server_peak_memory_mb = 0.0
        self._track_memory = False
        
        # Log split configuration
        self._log_split_info()

    def _log_split_info(self):
        """Log the split configuration."""
        num_layers = self.config.num_hidden_layers
        client_layers = self.split_layer
        server_layers = num_layers - self.split_layer
        
        logging.info(f"SplitOPT initialized:")
        logging.info(f"  - Total layers: {num_layers}")
        logging.info(f"  - Client: Embeddings + {client_layers} layers")
        logging.info(f"  - Server: {server_layers} layers + LM head")

    def enable_memory_tracking(self, enable=True):
        """Enable/disable memory tracking during forward passes."""
        self._track_memory = enable
        if enable and torch.cuda.is_available():
            torch.cuda.reset_peak_memory_stats()

    def forward(self, input_ids, labels=None, attention_mask=None, **kwargs):
        """
        Forward pass through split OPT.

        True split learning protocol:
        1. Client processes input through embeddings + first N layers
        2. Client sends hidden_states to server
        3. Server processes through remaining layers and computes loss
        4. For backward: Server sends gradients back to client
        
        Args:
            input_ids: Input token IDs (batch_size, seq_len)
            labels: Target labels for training
            attention_mask: Attention mask
            
        Returns:
            CausalLMOutput with loss and logits
        """
        # Track memory before client forward
        if self._track_memory and torch.cuda.is_available():
            torch.cuda.reset_peak_memory_stats()
        
        # ============ CLIENT SIDE ============
        hidden_states, presents = self.client(
            input_ids=input_ids,
            attention_mask=attention_mask,
        )
        
        # Track peak memory after client forward
        if self._track_memory and torch.cuda.is_available():
            client_peak = torch.cuda.max_memory_allocated() / (1024 ** 2)
            self.client_peak_memory_mb = max(self.client_peak_memory_mb, client_peak)
            torch.cuda.reset_peak_memory_stats()

        # ============ COMMUNICATION ============
        if self.training and labels is not None:
            # Store raw client activation for the manual backward step
            self.client_output = hidden_states
            # Break the graph for split learning; allow gradients on the server side
            self.server_input = hidden_states.clone().detach().requires_grad_(True)
            hidden_states_to_server = self.server_input
            # Detach presents too
            presents_to_server = None
            if presents is not None:
                presents_to_server = [
                    (p[0].clone().detach(), p[1].clone().detach()) if p is not None else None 
                    for p in presents
                ]
        else:
            hidden_states_to_server = hidden_states
            presents_to_server = presents

        # ============ SERVER SIDE ============
        server_outputs = self.server(
            input_ids_shape=input_ids.shape,
            hidden_states_client=hidden_states_to_server,
            attention_mask=attention_mask,
            presents_client=presents_to_server,
            lm_labels=labels,
            lm_mask=attention_mask,
            label_smooth=0.0,
        )
        
        # Track peak memory after server forward
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
        """
        Verify that the model is correctly split for split learning.
        
        Returns:
            Tuple of (is_valid, message)
        """
        is_valid, msg = verify_split_correctness(self.client, self.server)
        return is_valid, msg
    
    def get_split_info(self):
        """
        Get information about the split configuration.
        
        Returns:
            Dict with split configuration details
        """
        client_params = sum(p.numel() for p in self.client.parameters())
        server_params = sum(p.numel() for p in self.server.parameters())
        client_trainable = sum(p.numel() for p in self.client.parameters() if p.requires_grad)
        server_trainable = sum(p.numel() for p in self.server.parameters() if p.requires_grad)
        
        num_layers = self.config.num_hidden_layers
        
        return {
            "split_type": "layer_split",
            "split_layer": self.split_layer,
            "client_layers": self.split_layer,
            "server_layers": num_layers - self.split_layer,
            "total_layers": num_layers,
            "client_params": client_params,
            "server_params": server_params,
            "client_trainable": client_trainable,
            "server_trainable": server_trainable,
            "total_params": client_params + server_params,
            "client_percentage": client_params / (client_params + server_params) * 100,
            "server_percentage": server_params / (client_params + server_params) * 100,
        }

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
