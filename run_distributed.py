#!/usr/bin/env python3
"""
Distributed Split Learning with MeZO - DeComFL-style Implementation

This is a simplified multi-machine split learning implementation that:
1. Follows the DeComFL pattern (single entry point, role-based execution)
2. Reuses trainer.py logic for ZO optimization
3. Minimizes code duplication
4. Maintains all run.py functionality

Usage:
    # Server (machine 1):
    python run_distributed.py --role server --host 0.0.0.0 --port 50051 \
        --model_name facebook/opt-125m --task_name SST2 --trainer zo

    # Client (machine 2):
    python run_distributed.py --role client --server_host <server_ip> --port 50051 \
        --model_name facebook/opt-125m --task_name SST2 --trainer zo \
        --num_train 1000 --max_steps 500 --learning_rate 1e-4
"""

import logging
import os
import sys
import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import CrossEntropyLoss
import random
import numpy as np
from typing import Optional, List
from dataclasses import dataclass, field
from torch.utils.data import DataLoader, Dataset as TorchDataset
from tqdm import tqdm

# Note: Accelerate's DDP doesn't work for split learning (single TCP connection architecture)
# Multi-GPU is supported via DataParallel instead (single-process, multi-GPU)
# Keeping import for compatibility but not using Accelerator for DDP
try:
    from accelerate import Accelerator
    ACCELERATE_AVAILABLE = True
except ImportError:
    ACCELERATE_AVAILABLE = False

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


# =============================================================================
# GPU MEMORY TRACKING
# =============================================================================

class GPUMemoryTracker:
    """Track GPU memory usage and peak memory on a device."""
    
    def __init__(self, device=None, role="unknown"):
        self.role = role
        self.device = device if device is not None else torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.peak_memory_bytes = 0
        self.peak_memory_allocated_bytes = 0
        self.peak_memory_reserved_bytes = 0
        self.tracking = False
        
        if torch.cuda.is_available() and str(self.device).startswith('cuda'):
            # Reset peak stats at initialization
            torch.cuda.reset_peak_memory_stats(self.device)
    
    def start_tracking(self):
        """Start/reset memory tracking."""
        if torch.cuda.is_available() and str(self.device).startswith('cuda'):
            torch.cuda.reset_peak_memory_stats(self.device)
            self.tracking = True
    
    def update_peak(self):
        """Update peak memory stats by checking current peak values."""
        if not torch.cuda.is_available() or not str(self.device).startswith('cuda'):
            return
        
        # Get current peak memory stats
        current_allocated = torch.cuda.max_memory_allocated(self.device)
        current_reserved = torch.cuda.max_memory_reserved(self.device)
        
        # Update our tracked peaks
        if current_allocated > self.peak_memory_allocated_bytes:
            self.peak_memory_allocated_bytes = current_allocated
        if current_reserved > self.peak_memory_reserved_bytes:
            self.peak_memory_reserved_bytes = current_reserved
        
        # Track overall peak (using allocated as the primary metric)
        self.peak_memory_bytes = self.peak_memory_allocated_bytes
    
    def get_current_memory(self):
        """Get current memory usage."""
        if not torch.cuda.is_available() or not str(self.device).startswith('cuda'):
            return {'allocated': 0, 'reserved': 0}
        
        return {
            'allocated': torch.cuda.memory_allocated(self.device),
            'reserved': torch.cuda.memory_reserved(self.device),
        }
    
    def get_peak_memory(self):
        """Get peak memory usage tracked so far."""
        self.update_peak()
        return {
            'peak_allocated_bytes': self.peak_memory_allocated_bytes,
            'peak_reserved_bytes': self.peak_memory_reserved_bytes,
            'peak_allocated_gb': self.peak_memory_allocated_bytes / (1024**3),
            'peak_reserved_gb': self.peak_memory_reserved_bytes / (1024**3),
            'peak_allocated_mb': self.peak_memory_allocated_bytes / (1024**2),
            'peak_reserved_mb': self.peak_memory_reserved_bytes / (1024**2),
        }
    
    def get_summary(self):
        """Get a summary dict of memory usage."""
        peak = self.get_peak_memory()
        return {
            'role': self.role,
            'peak_gpu_memory_allocated_bytes': peak['peak_allocated_bytes'],
            'peak_gpu_memory_reserved_bytes': peak['peak_reserved_bytes'],
            'peak_gpu_memory_allocated_gb': peak['peak_allocated_gb'],
            'peak_gpu_memory_reserved_gb': peak['peak_reserved_gb'],
        }
    
    def print_summary(self):
        """Print memory summary."""
        peak = self.get_peak_memory()
        print(f"\n{'='*60}")
        print(f"GPU MEMORY USAGE - {self.role.upper()}")
        print(f"{'='*60}")
        print(f"Peak GPU Memory Allocated:   {peak['peak_allocated_gb']:.2f} GB ({peak['peak_allocated_mb']:.2f} MB)")
        print(f"Peak GPU Memory Reserved:    {peak['peak_reserved_gb']:.2f} GB ({peak['peak_reserved_mb']:.2f} MB)")
        print(f"{'='*60}")

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from transformers import AutoTokenizer, DataCollatorForTokenClassification, HfArgumentParser, TrainingArguments
from grpc_backend import TCPBackend, create_backend
from split_communication import ForwardPayload, BackwardPayload, ZOMetadata, CommunicationStats
from splitmodel import GPT2Config, GPT2LMModel_Client, GPT2LMModel_Server
from OPT_splitmodel import SplitOPT, OPTConfig, OPTLMModel_Client, OPTLMModel_Server
from utils import apply_lora_to_opt, mark_only_lora_as_trainable
from dataset import get_task
from utils import encode_prompt, DataCollatorWithPaddingAndNesting, Prediction
from metrics import calculate_metric


def set_seed(seed: int):
    """Set random seed for reproducibility - matching run.py"""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


@dataclass
class DistributedArguments(TrainingArguments):
    """
    Arguments for distributed split learning.
    Extends TrainingArguments to maintain compatibility with run.py
    """
    # Role selection (DeComFL-style single entry point)
    role: str = "client"  # "client" or "server"
    
    # Network settings
    server_host: str = "localhost"
    host: str = "0.0.0.0"
    port: int = 50051
    backend: str = "tcp"
    device: str = "cuda"
    
    # Model settings (matching run.py OurArguments)
    model_name: str = "facebook/opt-125m"
    task_name: str = "SST2"
    max_length: int = 2048
    model_card: str = "gpt2.sm"
    
    # Model loading
    load_float16: bool = False
    load_bfloat16: bool = False
    load_int8: bool = False
    no_auto_device: bool = False
    
    # Split layer configuration
    split_layer: int = 3  # layer index where to split the model (default: 3 for OPT/GPT-2)
    
    # Data settings
    num_train: int = 1000
    num_dev: int = 500
    num_eval: int = 1000
    num_train_sets: int = None
    train_set_seed: int = 0
    
    # Training settings (matching run.py)
    trainer: str = "zo"  # "zo", "regular", "none"
    client_optimizer: str = "auto"
    server_optimizer: str = "auto"
    
    # ZO settings
    zo_eps: float = 1e-3
    zo_continuous_rng: bool = False
    zo_variant: str = "central"
    zo_perturbation: str = "coordinate"
    num_pert: int = 1
    
    # Optimizer
    optimizer: str = "sgd"
    sgd_momentum: float = 0.0
    adam_beta1: float = 0.9
    adam_beta2: float = 0.999
    adam_epsilon: float = 1e-8
    client_learning_rate: Optional[float] = None
    server_learning_rate: Optional[float] = None
    
    # LoRA
    lora: bool = False
    lora_r: int = 8
    lora_alpha: int = 16
    
    # Evaluation settings (matching run.py / HuggingFace Trainer)
    eval_steps: int = 100  # Evaluate every N steps
    eval_strategy: str = "steps"  # "no", "steps", or "epoch"
    
    # Other run.py options
    only_train_option: bool = True
    train_as_classification: bool = False
    non_diff: bool = False
    prefix_tuning: bool = False
    num_prefix: int = 5
    no_reparam: bool = True
    prefix_init_by_real_act: bool = True
    head_tuning: bool = False
    init_checkpoint: str = None
    verbose: bool = False
    
    # Logging
    report_to: Optional[List[str]] = field(default_factory=lambda: [])


class NondiffCollator:
    """Collator for non-differentiable objective training - shared between run.py"""
    def __init__(self, tokenizer, pad_to_multiple_of=None):
        self.tokenizer = tokenizer
        self.pad_to_multiple_of = pad_to_multiple_of
    
    def __call__(self, features):
        golds = [f.pop("gold") if "gold" in f else None for f in features]
        max_length = max(len(f["input_ids"]) for f in features)
        if self.pad_to_multiple_of:
            max_length = ((max_length + self.pad_to_multiple_of - 1) // self.pad_to_multiple_of) * self.pad_to_multiple_of
        
        padded_input_ids, padded_labels, option_lens = [], [], []
        padding_side = getattr(self.tokenizer, 'padding_side', 'left')
        pad_token_id = self.tokenizer.pad_token_id or 0
        
        attention_masks = []
        
        for f in features:
            input_ids = f["input_ids"]
            labels = f.get("labels", input_ids)
            option_len = f.get("option_len", 0)
            
            seq_len = len(input_ids)
            pad_len = max_length - seq_len
            
            if padding_side == "left":
                padded = [pad_token_id] * pad_len + list(input_ids)
                padded_lab = [-100] * pad_len + list(labels)
                # attention_mask: 0 for padding, 1 for real tokens
                attn_mask = [0] * pad_len + [1] * seq_len
            else:
                padded = list(input_ids) + [pad_token_id] * pad_len
                padded_lab = list(labels) + [-100] * pad_len
                attn_mask = [1] * seq_len + [0] * pad_len
            
            padded_input_ids.append(padded)
            padded_labels.append(padded_lab)
            attention_masks.append(attn_mask)
            option_lens.append(option_len)
        
        batch = {
            "input_ids": torch.tensor(padded_input_ids, dtype=torch.long),
            "labels": torch.tensor(padded_labels, dtype=torch.long),
            "attention_mask": torch.tensor(attention_masks, dtype=torch.long),
            "option_len": option_lens,
        }
        if any(g is not None for g in golds):
            batch["gold"] = golds
        return batch


class HFDataset(TorchDataset):
    """Simple dataset wrapper - matching run.py"""
    def __init__(self, data):
        self.data = data
    def __len__(self):
        return len(self.data)
    def __getitem__(self, idx):
        return self.data[idx]


# =============================================================================
# SERVER IMPLEMENTATION
# =============================================================================

class DistributedServer:
    """
    Server for distributed split learning.
    Loads server portion of model and processes forward passes from client.
    
    Multi-GPU Support:
    - Split learning uses single TCP connection, so DDP (multi-process) doesn't work
    - Instead, we support DataParallel (single-process, multi-GPU) for batch parallelism
    - For large models that don't fit on one GPU, use model sharding via device placement
    """
    
    def __init__(self, args: DistributedArguments):
        self.args = args
        self.server_lr = args.server_learning_rate or args.learning_rate
        
        # Multi-GPU support via DataParallel (single-process)
        # Note: DDP/Accelerate with multiple processes doesn't work for split learning
        # due to single TCP connection architecture
        self.n_gpu = torch.cuda.device_count() if torch.cuda.is_available() else 0
        self.device = torch.device(args.device if torch.cuda.is_available() else 'cpu')
        self.use_data_parallel = (self.n_gpu > 1)
        
        if self.n_gpu > 1:
            logger.info(f"Multi-GPU detected: {self.n_gpu} GPUs available")
            logger.info(f"Using DataParallel for batch parallelism (single-process)")
        
        # Resolve optimizer mode
        global_trainer = args.trainer
        self.server_mode = args.server_optimizer
        if self.server_mode == "auto":
            self.server_mode = "zo" if global_trainer == "zo" else "fo"
        
        self.server_model = None
        self._server_module = None  # Underlying module (for DataParallel)
        self.backend = None
        self._sorted_params = None
        self._fo_optimizer = None  # FO optimizer (initialized if server_mode == "fo")
        
        # GPU memory tracking
        self.memory_tracker = GPUMemoryTracker(device=self.device, role="server")
        
    def load_model(self):
        """Load server portion of split model - matching run.py Framework.load_model()"""
        logger.info(f"Loading server model: {self.args.model_name}")
        
        if "opt" in self.args.model_name.lower():
            # Use OPT_splitmodel.py implementation with fixes
            split_layer = getattr(self.args, 'split_layer', 3)
            logger.info(f"Creating OPT split model with split_layer={split_layer}")
            
            # Create config from pretrained model
            opt_config = OPTConfig.from_pretrained(self.args.model_name)
            
            # Add LoRA config if enabled
            if self.args.lora:
                opt_config.lora_attn_dim = self.args.lora_r
                opt_config.lora_attn_alpha = self.args.lora_alpha
                opt_config.lora_dropout = 0.0
            
            # Create split model and load weights
            self.server_model = SplitOPT(opt_config, split_layer=split_layer)
            self.server_model.load_weight(self.args.model_name)

            if self.args.lora:
                logger.info(f"Applying LoRA (r={self.args.lora_r}, alpha={self.args.lora_alpha})")
                apply_lora_to_opt(self.server_model.server, self.args.lora_r, self.args.lora_alpha, 0.0)
                mark_only_lora_as_trainable(self.server_model.server)
        else:
            # GPT-2 style models
            if self.args.model_card == "gpt2.md":
                config = GPT2Config(n_embd=1024, n_layer=24, n_head=16,
                    lora_attn_dim=self.args.lora_r if self.args.lora else 0,
                    lora_attn_alpha=self.args.lora_alpha, lora_dropout=0.0)
            elif self.args.model_card == "gpt2.lg":
                config = GPT2Config(n_embd=1280, n_layer=36, n_head=20,
                    lora_attn_dim=self.args.lora_r if self.args.lora else 0,
                    lora_attn_alpha=self.args.lora_alpha, lora_dropout=0.0)
            else:
                config = GPT2Config(n_embd=768, n_layer=12, n_head=12,
                    lora_attn_dim=self.args.lora_r if self.args.lora else 0,
                    lora_attn_alpha=self.args.lora_alpha, lora_dropout=0.0)
            
            self.server_model = GPT2LMModel_Server(config)
            
            if "gpt2" in self.args.model_name.lower():
                self._load_gpt2_pretrained_weights()
            
            if self.args.lora and self.args.lora_r > 0:
                mark_only_lora_as_trainable(self.server_model)
        
        self.server_model.eval()
        
        if self.args.load_float16:
            self.server_model.half()
        elif self.args.load_bfloat16:
            self.server_model.bfloat16()
        
        self.server_model = self.server_model.to(self.device)
        
        # Store reference to underlying module before wrapping
        self._server_module = self.server_model
        
        # Multi-GPU: Use DataParallel for batch parallelism (single-process)
        # Note: ZO perturbations are applied to self._server_module (the unwrapped model)
        # DataParallel replicates the model during forward, so perturbations are consistent
        if self.use_data_parallel and self.n_gpu > 1:
            self.server_model = torch.nn.DataParallel(self.server_model)
            logger.info(f"Model wrapped with DataParallel for {self.n_gpu} GPU(s)")
        
        # Initialize FO optimizer if needed
        if self.server_mode == "fo":
            # Use parameters from the underlying module (not DataParallel wrapper)
            trainable_params = [p for p in self._server_module.parameters() if p.requires_grad]
            if self.args.optimizer.lower() == "adam":
                self._fo_optimizer = torch.optim.Adam(
                    trainable_params, lr=self.server_lr,
                    betas=(self.args.adam_beta1, self.args.adam_beta2),
                    eps=self.args.adam_epsilon,
                    weight_decay=self.args.weight_decay
                )
                logger.info(f"Initialized Adam optimizer for FO server (lr={self.server_lr}, betas=({self.args.adam_beta1}, {self.args.adam_beta2}))")
            else:
                self._fo_optimizer = torch.optim.SGD(
                    trainable_params, lr=self.server_lr,
                    momentum=self.args.sgd_momentum,
                    weight_decay=self.args.weight_decay
                )
                logger.info(f"Initialized SGD optimizer for FO server (lr={self.server_lr}, momentum={self.args.sgd_momentum})")
        
        # Cache sorted params for ZO (use underlying module, not DataParallel wrapper)
        server_module = self._get_server_module()
        self._sorted_params = sorted(
            [(name, param) for name, param in server_module.named_parameters() 
             if param.requires_grad],
            key=lambda x: x[0]
        )
        
        trainable = sum(p.numel() for p in self._server_module.parameters() if p.requires_grad)
        total = sum(p.numel() for p in self._server_module.parameters())
        logger.info(f"Server model: {total:,} params, {trainable:,} trainable")
        logger.info(f"Server mode: {self.server_mode.upper()}")
        
    def _load_gpt2_pretrained_weights(self):
        """Load pretrained GPT-2 weights for server (layers 3+)"""
        try:
            from transformers import GPT2LMHeadModel
            from collections import OrderedDict
            
            hf_model = GPT2LMHeadModel.from_pretrained(self.args.model_name)
            state_dict = hf_model.state_dict()
            server_state_dict = OrderedDict()
            split_layer = getattr(self.args, 'split_layer', 3)
            
            for key, value in state_dict.items():
                key_clean = key.replace("transformer.", "") if key.startswith("transformer.") else key
                
                if key_clean.startswith("wte."):
                    server_state_dict[f"transformer_Server.{key_clean}"] = value.clone()
                elif key_clean.startswith("ln_f."):
                    server_state_dict[f"transformer_Server.{key_clean}"] = value.clone()
                elif key.startswith("lm_head."):
                    server_state_dict[key] = value.clone()
                elif key_clean.startswith("h."):
                    parts = key_clean.split(".")
                    layer_idx = int(parts[1])
                    if layer_idx >= split_layer:
                        new_idx = layer_idx - split_layer
                        rest = ".".join(parts[2:])
                        server_state_dict[f"transformer_Server.h.{new_idx}.{rest}"] = value.clone()
                        
            self.server_model.load_state_dict(server_state_dict, strict=False)
            self.server_model.set_tied()
            logger.info(f"Loaded pretrained GPT-2 server weights (split_layer={split_layer})")
        except Exception as e:
            logger.warning(f"Could not load pretrained weights: {e}")
    
    def _load_opt_server_weights(self, hf_model, split_layer):
        """Load pretrained OPT weights for server (layers split_layer+)"""
        try:
            from collections import OrderedDict
            
            hf_state_dict = hf_model.state_dict()
            server_state_dict = OrderedDict()
            loaded_layers = set()
            
            for key, value in hf_state_dict.items():
                # Remove HF prefixes
                new_key = key
                if new_key.startswith("model.decoder."):
                    new_key = new_key[len("model.decoder."):]
                
                # Embeddings go to server for weight tying
                if new_key.startswith("embed_tokens."):
                    server_key = f"transformer_Server.{new_key}"
                    server_state_dict[server_key] = value.clone()
                
                # Final layer norm goes to server
                elif new_key.startswith("final_layer_norm."):
                    server_key = f"transformer_Server.{new_key}"
                    server_state_dict[server_key] = value.clone()
                
                # LM head
                elif key.startswith("lm_head."):
                    server_state_dict[key] = value.clone()
                
                # Split layers: server gets layers >= split_layer
                elif new_key.startswith("layers."):
                    parts = new_key.split(".")
                    layer_idx = int(parts[1])
                    rest = ".".join(parts[2:])
                    
                    # Map HF structure to our structure (fc1/fc2 -> mlp.fc1/mlp.fc2)
                    if rest.startswith("fc1") or rest.startswith("fc2"):
                        mapped_rest = "mlp." + rest
                    else:
                        mapped_rest = rest
                    
                    if layer_idx >= split_layer:
                        new_layer_idx = layer_idx - split_layer
                        server_key = f"transformer_Server.h.{new_layer_idx}.{mapped_rest}"
                        server_state_dict[server_key] = value.clone()
                        loaded_layers.add(layer_idx)
            
            missing, unexpected = self.server_model.load_state_dict(server_state_dict, strict=False)
            self.server_model.set_tied()
            
            # Verify embeddings loaded correctly (for weight tying)
            if hasattr(self.server_model, 'transformer_Server'):
                embed_weight = self.server_model.transformer_Server.embed_tokens.weight
                embed_sum = embed_weight.abs().sum().item()
                if embed_sum == 0:
                    logger.error("CRITICAL: Server embeddings not loaded properly!")
                else:
                    logger.info(f"Server embeddings verified (sum={embed_sum:.2f})")
                
                # Verify final layer norm
                ln_weight = self.server_model.transformer_Server.final_layer_norm.weight
                ln_sum = ln_weight.abs().sum().item()
                if ln_sum == 0:
                    logger.error("CRITICAL: Server final_layer_norm not loaded properly!")
                else:
                    logger.info(f"Server final_layer_norm verified (sum={ln_sum:.2f})")
            
            # Log any missing keys (excluding LoRA)
            if missing:
                real_missing = [k for k in missing if 'lora' not in k.lower()]
                if real_missing:
                    logger.warning(f"Server missing keys: {real_missing}")
            
            logger.info(f"Loaded pretrained OPT server weights (split_layer={split_layer})")
            logger.info(f"  Loaded layers: {sorted(loaded_layers)}")
        except Exception as e:
            logger.error(f"Could not load OPT pretrained weights: {e}")
            import traceback
            traceback.print_exc()
    
    def _generate_perturbation(self, param):
        """Generate perturbation vector - matching trainer.py"""
        if self.args.zo_perturbation == "layer":
            z = torch.normal(mean=0, std=1, size=param.data.size(),
                           device=param.data.device, dtype=param.data.dtype)
            z_norm = z.norm()
            if z_norm > 0:
                z = z / z_norm
            z = z * (param.numel() ** 0.5)
        else:
            z = torch.normal(mean=0, std=1, size=param.data.size(),
                           device=param.data.device, dtype=param.data.dtype)
        return z
    
    def _get_server_module(self):
        """Get the server module (handles DataParallel wrapping)"""
        # Use stored reference to underlying module
        if self._server_module is not None:
            return self._server_module.server
        # Fallback: unwrap DataParallel if needed
        model = self.server_model
        if hasattr(model, 'module'):
            model = model.module
        return model.server
    
    def _perturb_parameters(self, seed: int, scaling_factor: float):
        """Perturb server parameters - matching trainer.py"""
        torch.manual_seed(seed)
        for name, param in self._sorted_params:
            z = self._generate_perturbation(param)
            param.data = param.data + scaling_factor * z * self.args.zo_eps
            
    def _update_parameters(self, seed: int, projected_grad: float):
        """Update server parameters - matching trainer.py"""
        torch.manual_seed(seed)
        for name, param in self._sorted_params:
            z = self._generate_perturbation(param)
            if "bias" not in name and "layer_norm" not in name and "layernorm" not in name:
                param.data = param.data - self.server_lr * (
                    projected_grad * z + self.args.weight_decay * param.data)
            else:
                param.data = param.data - self.server_lr * (projected_grad * z)
    
    def _process_forward(self, forward_payload: ForwardPayload) -> BackwardPayload:
        """Process forward pass from client
        
        Handles both ZO and FO modes:
        - ZO mode: inference_mode, returns only loss
        - FO mode: gradient tracking, returns loss + grad_activations
        """
        hidden_states = forward_payload.activations.to(self.device)
        presents = forward_payload.presents
        if presents:
            presents = [p.to(self.device) if p is not None else None for p in presents]
        input_shape = forward_payload.input_shape
        mode = getattr(forward_payload, 'mode', 'train')
        phase = getattr(forward_payload, 'phase', None)
        
        labels = getattr(forward_payload, 'labels', None)
        if labels is not None:
            labels = labels.to(self.device)
        lm_mask = getattr(forward_payload, 'attention_mask', None)
        if lm_mask is not None:
            lm_mask = lm_mask.to(self.device)
        option_len = getattr(forward_payload, 'option_len', None)
        
        # Determine processing mode based on phase and server_mode
        # ZO/FO hybrid phases:
        #   - zo_fo_compute_grad: compute gradients, STORE them (don't update)
        #   - zo_fo_inference: inference only
        # FO/ZO hybrid phases:
        #   - perturb_pos, perturb_neg: server uses ZO (perturbation)
        #   - fo_backward: client needs gradients back
        # Pure FO mode: always compute gradients and update
        
        if phase == 'zo_fo_compute_grad':
            # ZO/FO hybrid: compute and store gradients, don't update
            return self._process_forward_fo_no_update(
                hidden_states, presents, input_shape, labels, lm_mask,
                option_len, forward_payload.batch_id, mode
            )
        elif phase == 'zo_fo_inference':
            # ZO/FO hybrid: inference only
            return self._process_forward_zo(
                hidden_states, presents, input_shape, labels, lm_mask,
                option_len, forward_payload.batch_id, mode
            )
        elif phase == 'fo_backward':
            # FO/ZO hybrid or pure FO: compute gradients and return them
            return self._process_forward_fo(
                hidden_states, presents, input_shape, labels, lm_mask, 
                option_len, forward_payload.batch_id, mode
            )
        elif self.server_mode == "fo" and phase not in ['perturb_pos', 'perturb_neg', 'restore']:
            # Pure FO mode: compute gradients and update
            return self._process_forward_fo(
                hidden_states, presents, input_shape, labels, lm_mask, 
                option_len, forward_payload.batch_id, mode
            )
        else:
            # ZO mode (including FO/ZO hybrid with perturbation phases): no gradients needed
            return self._process_forward_zo(
                hidden_states, presents, input_shape, labels, lm_mask,
                option_len, forward_payload.batch_id, mode
            )
    
    def _process_forward_zo(self, hidden_states, presents, input_shape, labels, lm_mask,
                            option_len, batch_id, mode):
        """ZO mode forward: inference only, no gradients"""
        server_module = self._get_server_module()
        with torch.inference_mode():
            outputs = server_module(
                input_ids_shape=input_shape,
                hidden_states_client=hidden_states,
                presents_client=presents,
                lm_labels=labels if mode == "train" else None,
                lm_mask=lm_mask,
                attention_mask=lm_mask,
            )
            
            if mode == "inference":
                logits = outputs[0] if isinstance(outputs, tuple) else outputs
                return BackwardPayload(logits=logits.cpu(), batch_id=batch_id)
            
            if labels is not None:
                logits, loss = outputs
                
                # Apply option_len masking if needed (matching run.py)
                if option_len is not None and self.args.only_train_option:
                    shift_logits = logits[..., :-1, :].contiguous()
                    shift_labels = labels[..., 1:].contiguous().clone()
                    
                    for _i, _len in enumerate(option_len):
                        if _len > 0:
                            shift_labels[_i, :-_len] = -100
                    
                    loss_fct = CrossEntropyLoss(ignore_index=-100)
                    vocab_size = logits.size(-1)
                    loss = loss_fct(shift_logits.view(-1, vocab_size), shift_labels.view(-1))
                
                # Ensure loss is a scalar (average if batch dimension present)
                if isinstance(loss, torch.Tensor) and loss.ndim > 0:
                    loss = loss.mean()
                
                loss_value = loss.item() if isinstance(loss, torch.Tensor) else loss
            else:
                loss_value = None
                
        return BackwardPayload(loss=loss_value, batch_id=batch_id)
    
    def _process_forward_fo_no_update(self, hidden_states, presents, input_shape, labels, lm_mask,
                                       option_len, batch_id, mode):
        """ZO/FO hybrid: compute and store gradients but DON'T update
        
        Gradients are accumulated in .grad and applied later when update signal received.
        """
        # Enable gradient tracking on hidden_states
        hidden_states = hidden_states.clone().detach().requires_grad_(True)
        
        server_module = self._get_server_module()
        server_module.train()
        # Note: Don't zero gradients - we'll accumulate and update later

        outputs = server_module(
            input_ids_shape=input_shape,
            hidden_states_client=hidden_states,
            presents_client=presents,
            lm_labels=labels if mode == "train" else None,
            lm_mask=lm_mask,
            attention_mask=lm_mask,
        )
        
        if labels is not None:
            logits, loss = outputs
            
            # Apply option_len masking
            if option_len is not None and self.args.only_train_option:
                shift_logits = logits[..., :-1, :].contiguous()
                shift_labels = labels[..., 1:].contiguous().clone()
                
                for _i, _len in enumerate(option_len):
                    if _len > 0:
                        shift_labels[_i, :-_len] = -100
                
                loss_fct = CrossEntropyLoss(ignore_index=-100)
                vocab_size = logits.size(-1)
                loss = loss_fct(shift_logits.view(-1, vocab_size), shift_labels.view(-1))
            
            # Ensure loss is a scalar (average if batch dimension present)
            if isinstance(loss, torch.Tensor) and loss.ndim > 0:
                loss = loss.mean()
            
            # Backward to compute gradients (stored in .grad, but don't update)
            # DataParallel handles gradient aggregation across GPUs automatically
            loss.backward()
            
            loss_value = loss.item()
        else:
            loss_value = None
        
        self.server_model.eval()
        return BackwardPayload(loss=loss_value, batch_id=batch_id)
    
    def _process_forward_fo(self, hidden_states, presents, input_shape, labels, lm_mask,
                            option_len, batch_id, mode):
        """FO mode forward: compute gradients and return grad_activations"""
        # Enable gradient tracking on hidden_states (the split point)
        hidden_states = hidden_states.clone().detach().requires_grad_(True)
        
        server_module = self._get_server_module()
        server_module.train()
        if self._fo_optimizer:
            self._fo_optimizer.zero_grad()

        outputs = server_module(
            input_ids_shape=input_shape,
            hidden_states_client=hidden_states,
            presents_client=presents,
            lm_labels=labels if mode == "train" else None,
            lm_mask=lm_mask,
            attention_mask=lm_mask,
        )
        
        if mode == "inference":
            logits = outputs[0] if isinstance(outputs, tuple) else outputs
            return BackwardPayload(logits=logits.cpu(), batch_id=batch_id)
        
        if labels is not None:
            logits, loss = outputs
            
            # Apply option_len masking if needed (matching run.py)
            if option_len is not None and self.args.only_train_option:
                shift_logits = logits[..., :-1, :].contiguous()
                shift_labels = labels[..., 1:].contiguous().clone()
                
                for _i, _len in enumerate(option_len):
                    if _len > 0:
                        shift_labels[_i, :-_len] = -100
                
                loss_fct = CrossEntropyLoss(ignore_index=-100)
                vocab_size = logits.size(-1)
                loss = loss_fct(shift_logits.view(-1, vocab_size), shift_labels.view(-1))
            
            # Ensure loss is a scalar (average if batch dimension present)
            if isinstance(loss, torch.Tensor) and loss.ndim > 0:
                loss = loss.mean()

            # Backward pass to compute gradients
            # DataParallel handles gradient aggregation across GPUs automatically
            loss.backward()
            
            # Get gradient at split point (to send back to client)
            grad_activations = hidden_states.grad.clone().detach().cpu() if hidden_states.grad is not None else None
            
            # Update server parameters
            if self._fo_optimizer:
                self._fo_optimizer.step()
            
            loss_value = loss.item()
        else:
            loss_value = None
            grad_activations = None
        
        self.server_model.eval()
        return BackwardPayload(loss=loss_value, grad_activations=grad_activations, batch_id=batch_id)
    
    def run(self):
        """Main server loop"""
        logger.info("Starting Distributed Server...")
        self.load_model()
        
        # Start GPU memory tracking
        self.memory_tracker.start_tracking()
        
        self.backend = create_backend(
            backend_type=self.args.backend,
            mode='server',
            host=self.args.host,
            port=self.args.port,
            device=str(self.device),
        )
        
        self.backend.start()
        step = 0
        received_data = False  # Track if we received actual training data
        
        logger.info("Server ready, waiting for client...")
        
        try:
            while True:
                try:
                    forward_payload = self.backend.recv_forward()
                    received_data = True  # We got real data, this is a real client
                except ConnectionError:
                    if not received_data:
                        # Connection closed before any data - likely a readiness probe
                        logger.info("Connection closed without data (probe?), waiting for real client...")
                        self.backend.wait_for_client()
                        continue
                    else:
                        # Real client disconnected after training
                        logger.info("Client disconnected")
                        break
                except Exception as e:
                    logger.error(f"Error receiving: {e}")
                    break
                
                # Handle ZO perturbation
                seed = getattr(forward_payload, 'seed', None)
                phase = getattr(forward_payload, 'phase', None)
                
                if seed is not None and self.server_mode == "zo":
                    if phase == "perturb_pos":
                        self._perturb_parameters(seed, scaling_factor=1)
                    elif phase == "perturb_neg":
                        self._perturb_parameters(seed, scaling_factor=-2)
                    elif phase == "restore":
                        self._perturb_parameters(seed, scaling_factor=1)
                
                # Process forward and send result
                backward_payload = self._process_forward(forward_payload)
                self.backend.send_backward(backward_payload)
                
                # Update GPU memory peak tracking
                self.memory_tracker.update_peak()
                
                # Check for update signal based on phase
                phase = getattr(forward_payload, 'phase', None)
                
                # Determine if we should wait for an update signal
                # - ZO/ZO mode: after perturb_neg (central) or perturb_pos (forward)
                # - ZO/FO hybrid: after zo_fo_inference (second phase)
                wait_for_update = False
                if self.server_mode == "zo" and seed is not None:
                    update_phase = "perturb_pos" if self.args.zo_variant == "forward" else "perturb_neg"
                    wait_for_update = (phase == update_phase)
                elif self.server_mode == "fo" and phase == "zo_fo_inference":
                    # ZO/FO hybrid: client finished both forwards, waiting for update
                    wait_for_update = True
                
                if wait_for_update:
                    try:
                        zo_metadata = self.backend.recv_zo_metadata()
                        if zo_metadata and zo_metadata.step_phase == "update":
                            if self.server_mode == "zo":
                                # ZO mode: restore and update using perturbation
                                projected_grad = getattr(zo_metadata, 'projected_grad', None)
                                if projected_grad is not None and zo_metadata.seed is not None:
                                    restore_factor = getattr(zo_metadata, 'restore_scaling_factor', 1)
                                    self._perturb_parameters(zo_metadata.seed, restore_factor)
                                    self._update_parameters(zo_metadata.seed, projected_grad)
                            elif self.server_mode == "fo" and self._fo_optimizer:
                                # FO mode (ZO/FO hybrid): apply stored gradients
                                self._fo_optimizer.step()
                                self._fo_optimizer.zero_grad()
                    except Exception as e:
                        logger.warning(f"Failed to receive update signal: {e}")
                
                if step % self.args.logging_steps == 0:
                    loss_str = f"loss={backward_payload.loss:.4f}" if backward_payload.loss else ""
                    logger.info(f"Step {step}: {loss_str}")
                    
                step += 1
                
        except KeyboardInterrupt:
            logger.info("Server interrupted")
        finally:
            # Print GPU memory summary
            self.memory_tracker.print_summary()
            
            if self.backend:
                self.backend.close()
            logger.info("Server stopped")


# =============================================================================
# CLIENT IMPLEMENTATION  
# =============================================================================

class DistributedClient:
    """
    Client for distributed split learning.
    Loads client portion, handles data, and drives training loop.
    
    Multi-GPU Support:
    - Split learning uses single TCP connection, so DDP (multi-process) doesn't work
    - Instead, we support DataParallel (single-process, multi-GPU) for batch parallelism
    """
    
    def __init__(self, args: DistributedArguments, task):
        self.args = args
        self.task = task
        self.client_lr = args.client_learning_rate or args.learning_rate
        
        # Multi-GPU support via DataParallel (single-process)
        self.n_gpu = torch.cuda.device_count() if torch.cuda.is_available() else 0
        self.device = torch.device(args.device if torch.cuda.is_available() else 'cpu')
        self.use_data_parallel = (self.n_gpu > 1)
        
        if self.n_gpu > 1:
            logger.info(f"Multi-GPU detected: {self.n_gpu} GPUs available")
            logger.info(f"Using DataParallel for batch parallelism (single-process)")
        
        # Resolve optimizer mode
        global_trainer = args.trainer
        self.client_mode = args.client_optimizer
        if self.client_mode == "auto":
            self.client_mode = "zo" if global_trainer == "zo" else "fo"
        
        self.client_model = None
        self._client_module = None  # Underlying module (for DataParallel)
        self.tokenizer = None
        self.backend = None
        self._sorted_params = None
        self.communication_rounds = 0  # Counter for actual communication exchanges
        self._client_output = None  # For FO mode backward pass
        self._fo_optimizer = None  # FO optimizer (initialized if client_mode == "fo")
        
        # GPU memory tracking
        self.memory_tracker = GPUMemoryTracker(device=self.device, role="client")
        
        # Resolve server mode for hybrid modes
        self.server_mode = args.server_optimizer
        if self.server_mode == "auto":
            self.server_mode = "zo" if global_trainer == "zo" else "fo"
        
    def load_model(self):
        """Load client portion of split model"""
        logger.info(f"Loading client model: {self.args.model_name}")
        
        if "opt" in self.args.model_name.lower():
            # Use OPT_splitmodel.py implementation with fixes
            split_layer = getattr(self.args, 'split_layer', 3)
            logger.info(f"Creating OPT split model with split_layer={split_layer}")
            
            # Create config from pretrained model
            opt_config = OPTConfig.from_pretrained(self.args.model_name)
            
            # Add LoRA config if enabled
            if self.args.lora:
                opt_config.lora_attn_dim = self.args.lora_r
                opt_config.lora_attn_alpha = self.args.lora_alpha
                opt_config.lora_dropout = 0.0
            
            # Create split model and load weights
            self.client_model = SplitOPT(opt_config, split_layer=split_layer)
            self.client_model.load_weight(self.args.model_name)

            if self.args.lora:
                apply_lora_to_opt(self.client_model.client, self.args.lora_r, self.args.lora_alpha, 0.0)
                mark_only_lora_as_trainable(self.client_model.client)
        else:
            # GPT-2 style models
            if self.args.model_card == "gpt2.md":
                config = GPT2Config(n_embd=1024, n_layer=24, n_head=16,
                    lora_attn_dim=self.args.lora_r if self.args.lora else 0,
                    lora_attn_alpha=self.args.lora_alpha, lora_dropout=0.0)
            elif self.args.model_card == "gpt2.lg":
                config = GPT2Config(n_embd=1280, n_layer=36, n_head=20,
                    lora_attn_dim=self.args.lora_r if self.args.lora else 0,
                    lora_attn_alpha=self.args.lora_alpha, lora_dropout=0.0)
            else:
                config = GPT2Config(n_embd=768, n_layer=12, n_head=12,
                    lora_attn_dim=self.args.lora_r if self.args.lora else 0,
                    lora_attn_alpha=self.args.lora_alpha, lora_dropout=0.0)
            
            from splitmodel import GPT2LMModel_Client
            self.client_model = GPT2LMModel_Client(config)
            
            if "gpt2" in self.args.model_name.lower():
                self._load_gpt2_pretrained_weights()
            
            if self.args.lora and self.args.lora_r > 0:
                mark_only_lora_as_trainable(self.client_model)
        
        self.client_model.eval()
        
        if self.args.load_float16:
            self.client_model.half()
        elif self.args.load_bfloat16:
            self.client_model.bfloat16()
        
        self.client_model = self.client_model.to(self.device)
        
        # Store reference to underlying module before wrapping
        self._client_module = self.client_model
        
        # Multi-GPU: Use DataParallel for batch parallelism (single-process)
        if self.use_data_parallel and self.n_gpu > 1:
            self.client_model = torch.nn.DataParallel(self.client_model)
            logger.info(f"Model wrapped with DataParallel for {self.n_gpu} GPU(s)")
        
        # Initialize FO optimizer if needed
        if self.client_mode == "fo":
            # Use parameters from the underlying module (not DataParallel wrapper)
            trainable_params = [p for p in self._client_module.parameters() if p.requires_grad]
            if len(trainable_params) > 0:
                if self.args.optimizer.lower() == "adam":
                    self._fo_optimizer = torch.optim.Adam(
                        trainable_params, lr=self.client_lr,
                        betas=(self.args.adam_beta1, self.args.adam_beta2),
                        eps=self.args.adam_epsilon,
                        weight_decay=self.args.weight_decay
                    )
                    logger.info(f"Initialized Adam optimizer for FO client (lr={self.client_lr}, betas=({self.args.adam_beta1}, {self.args.adam_beta2}))")
                else:
                    self._fo_optimizer = torch.optim.SGD(
                        trainable_params, lr=self.client_lr, 
                        momentum=self.args.sgd_momentum,
                        weight_decay=self.args.weight_decay
                    )
                    logger.info(f"Initialized SGD optimizer for FO client (lr={self.client_lr}, momentum={self.args.sgd_momentum})")
            else:
                logger.warning("Client has 0 trainable parameters - no client optimizer created")
                logger.warning("Client gradients will be computed but not applied (only server trains)")
        
        # Cache sorted params for ZO (use underlying module, not DataParallel wrapper)
        client_module = self._get_client_module()
        self._sorted_params = sorted(
            [(name, param) for name, param in client_module.named_parameters() 
             if param.requires_grad],
            key=lambda x: x[0]
        )
        
        trainable = sum(p.numel() for p in self._client_module.parameters() if p.requires_grad)
        total = sum(p.numel() for p in self._client_module.parameters())
        logger.info(f"Client model: {total:,} params, {trainable:,} trainable")
        logger.info(f"Client mode: {self.client_mode.upper()}")
        logger.info(f"Server mode: {self.server_mode.upper()}")
        
        # Load tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(self.args.model_name, use_fast=False)
        if "opt" in self.args.model_name.lower():
            self.tokenizer.bos_token_id = 0
        if "gpt2" in self.args.model_name.lower() and self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        
    def _load_gpt2_pretrained_weights(self):
        """Load pretrained GPT-2 weights for client (layers 0 to split_layer-1)"""
        try:
            from transformers import GPT2LMHeadModel
            from collections import OrderedDict
            
            hf_model = GPT2LMHeadModel.from_pretrained(self.args.model_name)
            state_dict = hf_model.state_dict()
            client_state_dict = OrderedDict()
            split_layer = getattr(self.args, 'split_layer', 3)
            
            for key, value in state_dict.items():
                key_clean = key.replace("transformer.", "") if key.startswith("transformer.") else key
                
                if key_clean.startswith("wte.") or key_clean.startswith("wpe."):
                    client_state_dict[f"transformer_Client.{key_clean}"] = value.clone()
                elif key_clean.startswith("h."):
                    parts = key_clean.split(".")
                    layer_idx = int(parts[1])
                    if layer_idx < split_layer:
                        rest = ".".join(parts[2:])
                        client_state_dict[f"transformer_Client.h.{layer_idx}.{rest}"] = value.clone()
                        
            self.client_model.load_state_dict(client_state_dict, strict=False)
            logger.info(f"Loaded pretrained GPT-2 client weights (split_layer={split_layer})")
        except Exception as e:
            logger.warning(f"Could not load pretrained weights: {e}")
    
    def _load_opt_client_weights(self, hf_model, split_layer):
        """Load pretrained OPT weights for client (layers 0 to split_layer-1)"""
        try:
            from collections import OrderedDict
            
            hf_state_dict = hf_model.state_dict()
            client_state_dict = OrderedDict()
            loaded_layers = set()
            
            for key, value in hf_state_dict.items():
                # Remove HF prefixes
                new_key = key
                if new_key.startswith("model.decoder."):
                    new_key = new_key[len("model.decoder."):]
                
                # Embeddings go to client
                if new_key.startswith("embed_tokens.") or new_key.startswith("embed_positions."):
                    client_key = f"transformer_Client.{new_key}"
                    client_state_dict[client_key] = value.clone()
                
                # Split layers: client gets layers < split_layer
                elif new_key.startswith("layers."):
                    parts = new_key.split(".")
                    layer_idx = int(parts[1])
                    rest = ".".join(parts[2:])
                    
                    # Map HF structure to our structure (fc1/fc2 -> mlp.fc1/mlp.fc2)
                    if rest.startswith("fc1") or rest.startswith("fc2"):
                        mapped_rest = "mlp." + rest
                    else:
                        mapped_rest = rest
                    
                    if layer_idx < split_layer:
                        client_key = f"transformer_Client.h.{layer_idx}.{mapped_rest}"
                        client_state_dict[client_key] = value.clone()
                        loaded_layers.add(layer_idx)
            
            missing, unexpected = self.client_model.load_state_dict(client_state_dict, strict=False)
            
            # Verify embeddings loaded correctly
            if hasattr(self.client_model, 'transformer_Client'):
                embed_weight = self.client_model.transformer_Client.embed_tokens.weight
                embed_sum = embed_weight.abs().sum().item()
                if embed_sum == 0:
                    logger.error("CRITICAL: Client embeddings not loaded properly!")
                else:
                    logger.info(f"Client embeddings verified (sum={embed_sum:.2f})")
                
                pos_embed_weight = self.client_model.transformer_Client.embed_positions.weight
                pos_sum = pos_embed_weight.abs().sum().item()
                if pos_sum == 0:
                    logger.error("CRITICAL: Client position embeddings not loaded properly!")
                else:
                    logger.info(f"Client position embeddings verified (sum={pos_sum:.2f})")
            
            # Log any missing keys (excluding LoRA)
            if missing:
                real_missing = [k for k in missing if 'lora' not in k.lower()]
                if real_missing:
                    logger.warning(f"Client missing keys: {real_missing}")
            
            logger.info(f"Loaded pretrained OPT client weights (split_layer={split_layer})")
            logger.info(f"  Loaded layers: {sorted(loaded_layers) if loaded_layers else 'embeddings only'}")
        except Exception as e:
            logger.error(f"Could not load OPT pretrained weights: {e}")
            import traceback
            traceback.print_exc()
    
    def connect(self):
        """Connect to server"""
        logger.info(f"Connecting to server at {self.args.server_host}:{self.args.port}...")
        self.backend = create_backend(
            backend_type=self.args.backend,
            mode='client',
            host=self.args.server_host,
            port=self.args.port,
            device=str(self.device),
        )
        self.backend.connect()
        logger.info("Connected to server")
    
    def _get_client_module(self):
        """Get the client module (handles DataParallel wrapping)"""
        # Use stored reference to underlying module
        if self._client_module is not None:
            return self._client_module.client
        # Fallback: unwrap DataParallel if needed
        model = self.client_model
        if hasattr(model, 'module'):
            model = model.module
        return model.client
    
    def _generate_perturbation(self, param):
        """Generate perturbation vector - matching trainer.py"""
        if self.args.zo_perturbation == "layer":
            z = torch.normal(mean=0, std=1, size=param.data.size(),
                           device=param.data.device, dtype=param.data.dtype)
            z_norm = z.norm()
            if z_norm > 0:
                z = z / z_norm
            z = z * (param.numel() ** 0.5)
        else:
            z = torch.normal(mean=0, std=1, size=param.data.size(),
                           device=param.data.device, dtype=param.data.dtype)
        return z
    
    def _perturb_parameters(self, seed: int, scaling_factor: float):
        """Perturb client parameters - matching trainer.py"""
        torch.manual_seed(seed)
        for name, param in self._sorted_params:
            z = self._generate_perturbation(param)
            param.data = param.data + scaling_factor * z * self.args.zo_eps
            
    def _update_parameters(self, seed: int, projected_grad: float):
        """Update client parameters - matching trainer.py"""
        torch.manual_seed(seed)
        for name, param in self._sorted_params:
            z = self._generate_perturbation(param)
            if "bias" not in name and "layer_norm" not in name and "layernorm" not in name:
                param.data = param.data - self.client_lr * (
                    projected_grad * z + self.args.weight_decay * param.data)
            else:
                param.data = param.data - self.client_lr * (projected_grad * z)
    
    def _forward_to_server(self, input_ids, labels, attention_mask, step, 
                          seed=None, phase=None, option_len=None, mode="train",
                          requires_grad=False):
        """Send forward through client to server, return loss/logits (and optionally grad_activations)
        
        Args:
            requires_grad: If True, enable gradient tracking for FO mode.
                          Client will store output for backward pass.
        """
        # Get the actual client module (handles Accelerate/DDP wrapping)
        client_module = self._get_client_module()
        
        if requires_grad:
            # FO mode: Need gradients, don't use inference_mode
            client_module.train()
            hidden_states, presents = client_module(input_ids, attention_mask=attention_mask)
            # Ensure gradients are enabled for the output
            if not hidden_states.requires_grad:
                hidden_states = hidden_states.clone().detach().requires_grad_(True)
            # Store for backward pass
            self._client_output = hidden_states
        else:
            with torch.inference_mode():
                hidden_states, presents = client_module(input_ids, attention_mask=attention_mask)
        
        forward_payload = ForwardPayload(
            activations=hidden_states.detach().cpu() if not requires_grad else hidden_states.cpu(),
            presents=[p.detach().cpu() if p is not None else None for p in presents] if presents else None,
            input_shape=input_ids.shape,
            seed=seed,
            batch_id=step,
            labels=labels.cpu(),
            attention_mask=attention_mask.cpu() if attention_mask is not None else None,
            phase=phase,
            option_len=option_len,
            mode=mode,
        )
        
        self.backend.send_forward(forward_payload)
        self.communication_rounds += 1  # Count send_forward
        backward_payload = self.backend.recv_backward()
        self.communication_rounds += 1  # Count recv_backward
        
        if mode == "inference":
            return backward_payload.logits
        
        if requires_grad:
            # FO mode: return both loss and grad_activations
            return backward_payload.loss, backward_payload.grad_activations
        return backward_payload.loss
    
    def _send_update_signal(self, seed: int, projected_grad: float, restore_scaling_factor: int = 1):
        """Send update signal to server with restore info
        
        Args:
            seed: The perturbation seed
            projected_grad: The computed gradient estimate
            restore_scaling_factor: Scaling factor to restore parameters before update
                - For central variant (after -2ε): use +1 to go back to θ
                - For forward variant (after +ε): use -1 to go back to θ
        """
        zo_metadata = ZOMetadata(
            seed=seed,
            zo_eps=self.args.zo_eps,
            scaling_factor=restore_scaling_factor,
            step_phase='update',
            projected_grad=projected_grad,
            restore_scaling_factor=restore_scaling_factor,
        )
        self.backend.send_zo_metadata(zo_metadata)
        self.communication_rounds += 1  # Count send_zo_metadata
    
    def _zo_training_step(self, inputs, step):
        """
        ZO training step with multi-perturbation support.
        Matching trainer.py zo_step_split_coordinated logic.
        """
        input_ids = inputs['input_ids'].to(self.device)
        labels = inputs.get('labels', input_ids.clone()).to(self.device)
        attention_mask = inputs.get('attention_mask')
        if attention_mask is not None:
            attention_mask = attention_mask.to(self.device)
        option_len = inputs.get('option_len', None)
        
        # Generate seeds for all perturbations
        seeds = [np.random.randint(1000000000) for _ in range(self.args.num_pert)]
        
        if self.args.zo_variant == 'forward':
            # RGE-forward: baseline + perturbed
            loss0 = self._forward_to_server(input_ids, labels, attention_mask, step,
                                           seed=None, phase=None, option_len=option_len)
            loss_for_log = loss0
            
            for seed in seeds:
                self._perturb_parameters(seed, scaling_factor=1)
                loss1 = self._forward_to_server(input_ids, labels, attention_mask, step,
                                               seed=seed, phase='perturb_pos', option_len=option_len)
                
                # Restore client parameters
                self._perturb_parameters(seed, scaling_factor=-1)
                
                projected_grad = (loss1 - loss0) / (self.args.zo_eps * self.args.num_pert)
                self._update_parameters(seed, projected_grad)
                # For forward variant: after +1 perturbation, restore with -1
                self._send_update_signal(seed, projected_grad, restore_scaling_factor=-1)
        else:
            # RGE-central (default)
            loss_for_log = None
            
            for seed in seeds:
                self._perturb_parameters(seed, scaling_factor=1)
                loss1 = self._forward_to_server(input_ids, labels, attention_mask, step,
                                               seed=seed, phase='perturb_pos', option_len=option_len)
                if loss_for_log is None:
                    loss_for_log = loss1
                
                self._perturb_parameters(seed, scaling_factor=-2)
                loss2 = self._forward_to_server(input_ids, labels, attention_mask, step,
                                               seed=seed, phase='perturb_neg', option_len=option_len)
                
                # Restore client parameters (no forward pass needed - just local restore)
                self._perturb_parameters(seed, scaling_factor=1)
                # Tell server to restore via update signal (not forward pass)
                
                projected_grad = (loss1 - loss2) / (2 * self.args.zo_eps * self.args.num_pert)
                self._update_parameters(seed, projected_grad)
                # For central variant: after -2ε perturbation (at -ε), restore with +1
                self._send_update_signal(seed, projected_grad, restore_scaling_factor=1)
        
        return loss_for_log
    
    def _fo_training_step(self, inputs, step):
        """
        FO (first-order) training step with backpropagation.
        
        Protocol:
        1. Forward pass through client with gradient tracking
        2. Send activations to server
        3. Server computes loss and backward, returns grad_activations
        4. Client backward using grad_activations (if client has trainable params)
        5. Update client parameters with optimizer (if available)
        
        Note: If client has no trainable params (e.g., LoRA only on server),
        only the server will be updated. This is still valid FO split learning.
        """
        input_ids = inputs['input_ids'].to(self.device)
        labels = inputs.get('labels', input_ids.clone()).to(self.device)
        attention_mask = inputs.get('attention_mask')
        if attention_mask is not None:
            attention_mask = attention_mask.to(self.device)
        option_len = inputs.get('option_len', None)
        
        # Zero gradients
        if self._fo_optimizer:
            self._fo_optimizer.zero_grad()
        
        # Check if client has trainable params that need gradients
        has_trainable = any(p.requires_grad for p in self.client_model.parameters())
        
        if has_trainable:
            # Forward with gradient tracking
            loss, grad_activations = self._forward_to_server(
                input_ids, labels, attention_mask, step,
                option_len=option_len, requires_grad=True
            )
            
            # Backward through client using gradient from server
            # DataParallel handles gradient aggregation across GPUs automatically
            if grad_activations is not None and self._client_output is not None:
                grad_activations = grad_activations.to(self.device)
                self._client_output.backward(grad_activations)
                self._client_output = None
        else:
            # No trainable params on client, just forward (server still trains via FO)
            loss = self._forward_to_server(
                input_ids, labels, attention_mask, step,
                option_len=option_len, requires_grad=False
            )
        
        # Update client parameters (if optimizer exists)
        if self._fo_optimizer:
            self._fo_optimizer.step()
        
        return loss
    
    def _hybrid_zo_fo_step(self, inputs, step):
        """
        Hybrid mode: Client uses ZO (zeroth-order), Server uses FO (first-order).
        
        Protocol (matching trainer.py _zo_fo_step):
        1. Perturb client (+ε*z)
        2. Forward → loss1; Server computes gradients but does NOT update yet
        3. Perturb client to (-ε*z)
        4. Forward → loss2; Server does inference only (no update)
        5. Restore client, compute ZO gradient, update client
        6. Signal server to update with stored gradients
        
        IMPORTANT: Server must NOT update between loss1 and loss2!
        """
        input_ids = inputs['input_ids'].to(self.device)
        labels = inputs.get('labels', input_ids.clone()).to(self.device)
        attention_mask = inputs.get('attention_mask')
        if attention_mask is not None:
            attention_mask = attention_mask.to(self.device)
        option_len = inputs.get('option_len', None)
        
        seeds = [np.random.randint(1000000000) for _ in range(self.args.num_pert)]
        
        loss_for_log = None
        
        for seed in seeds:
            # Phase 1: Perturb client +ε, server computes gradients (no update)
            self._perturb_parameters(seed, scaling_factor=1)
            loss1 = self._forward_to_server(input_ids, labels, attention_mask, step,
                                           seed=seed, phase='zo_fo_compute_grad', option_len=option_len)
            if loss_for_log is None:
                loss_for_log = loss1
            
            # Phase 2: Perturb client to -ε, server inference only
            self._perturb_parameters(seed, scaling_factor=-2)
            loss2 = self._forward_to_server(input_ids, labels, attention_mask, step,
                                           seed=seed, phase='zo_fo_inference', option_len=option_len)
            
            # Restore client
            self._perturb_parameters(seed, scaling_factor=1)
            
            # Update client with ZO gradient
            projected_grad = (loss1 - loss2) / (2 * self.args.zo_eps * self.args.num_pert)
            self._update_parameters(seed, projected_grad)
            
            # Signal server to update with stored gradients
            self._send_update_signal(seed, projected_grad, restore_scaling_factor=1)
        
        return loss_for_log
    
    def _hybrid_fo_zo_step(self, inputs, step):
        """
        Hybrid mode: Client uses FO (first-order), Server uses ZO (zeroth-order).
        Client will compute gradients via backprop, server uses ZO.
        """
        input_ids = inputs['input_ids'].to(self.device)
        labels = inputs.get('labels', input_ids.clone()).to(self.device)
        attention_mask = inputs.get('attention_mask')
        if attention_mask is not None:
            attention_mask = attention_mask.to(self.device)
        option_len = inputs.get('option_len', None)
        
        seeds = [np.random.randint(1000000000) for _ in range(self.args.num_pert)]
        loss_for_log = None
        
        for seed in seeds:
            # Server perturbs +ε (indicated by phase)
            loss1 = self._forward_to_server(input_ids, labels, attention_mask, step,
                                           seed=seed, phase='perturb_pos', option_len=option_len)
            if loss_for_log is None:
                loss_for_log = loss1
            
            # Server perturbs to -ε
            loss2 = self._forward_to_server(input_ids, labels, attention_mask, step,
                                           seed=seed, phase='perturb_neg', option_len=option_len)
            
            # Send update signal to server for ZO update
            projected_grad = (loss1 - loss2) / (2 * self.args.zo_eps * self.args.num_pert)
            self._send_update_signal(seed, projected_grad, restore_scaling_factor=1)
        
        # Now do FO update for client (with updated server)
        has_trainable = any(p.requires_grad for p in self.client_model.parameters())
        
        if has_trainable:
            if self._fo_optimizer:
                self._fo_optimizer.zero_grad()
            
            # Forward with gradient tracking for client FO
            loss, grad_activations = self._forward_to_server(
                input_ids, labels, attention_mask, step,
                option_len=option_len, requires_grad=True, phase='fo_backward'
            )
            
            # Backward through client using gradient from server
            # DataParallel handles gradient aggregation across GPUs automatically
            if grad_activations is not None and self._client_output is not None:
                grad_activations = grad_activations.to(self.device)
                self._client_output.backward(grad_activations)
                self._client_output = None
            
            if self._fo_optimizer:
                self._fo_optimizer.step()
        
        return loss_for_log
    
    def train(self, train_samples, eval_samples):
        """Training loop - matching run.py Framework.train()"""
        if self.client_model is None:
            self.load_model()
        
        # Start GPU memory tracking
        self.memory_tracker.start_tracking()
        
        # Store eval_samples for periodic evaluation
        self._eval_samples = eval_samples
        
        self.tokenizer.padding_side = "left"
        
        def _convert(samples):
            data = []
            for sample in samples:
                encoded_candidates, option_lens = encode_prompt(
                    self.task, self.task.get_template(), [], sample, self.tokenizer,
                    max_length=self.args.max_length)
                
                if isinstance(sample.correct_candidate, list):
                    correct_candidate_id = sample.candidates.index(sample.correct_candidate[0])
                else:
                    correct_candidate_id = sample.candidates.index(sample.correct_candidate)
                
                if self.args.train_as_classification:
                    data.append([{
                        "input_ids": encoded_candidates[i],
                        "labels": correct_candidate_id,
                        "option_len": option_lens[i],
                        "num_options": len(sample.candidates)
                    } for i in range(len(encoded_candidates))])
                elif self.args.only_train_option:
                    if self.args.non_diff:
                        data.append({
                            "input_ids": encoded_candidates[correct_candidate_id],
                            "labels": encoded_candidates[correct_candidate_id],
                            "option_len": option_lens[correct_candidate_id],
                            "gold": sample.correct_candidate
                        })
                    else:
                        data.append({
                            "input_ids": encoded_candidates[correct_candidate_id],
                            "labels": encoded_candidates[correct_candidate_id],
                            "option_len": option_lens[correct_candidate_id]
                        })
                else:
                    data.append({
                        "input_ids": encoded_candidates[correct_candidate_id],
                        "labels": encoded_candidates[correct_candidate_id]
                    })
            return data
        
        logger.info(f"Converting {len(train_samples)} training samples...")
        train_dataset = HFDataset(_convert(train_samples))
        
        if self.args.train_as_classification:
            collator = DataCollatorWithPaddingAndNesting(self.tokenizer, pad_to_multiple_of=8)
        elif self.args.non_diff:
            collator = NondiffCollator(self.tokenizer, pad_to_multiple_of=8)
        else:
            collator = DataCollatorForTokenClassification(self.tokenizer, pad_to_multiple_of=8)
        
        train_dataloader = DataLoader(
            train_dataset,
            batch_size=self.args.per_device_train_batch_size,
            shuffle=True,
            collate_fn=collator,
        )
        
        steps_per_epoch = len(train_dataloader)
        num_epochs = max(1, (self.args.max_steps + steps_per_epoch - 1) // steps_per_epoch)
        
        logger.info(f"Training: {len(train_samples)} samples, {steps_per_epoch} steps/epoch")
        logger.info(f"Running for {num_epochs} epochs (max_steps={self.args.max_steps})")
        
        # Check for periodic evaluation settings
        eval_steps = getattr(self.args, 'eval_steps', 0)
        eval_strategy = getattr(self.args, 'eval_strategy', 'no')
        if eval_strategy == "steps" and eval_steps > 0:
            logger.info(f"Periodic evaluation every {eval_steps} steps")
        
        # Track best accuracy for logging
        best_accuracy = 0.0
        
        global_step = 0
        total_loss = 0.0
        log_loss = 0.0
        log_steps = 0
        
        for epoch in range(num_epochs):
            logger.info(f"Epoch {epoch + 1}/{num_epochs}")
            
            for batch in tqdm(train_dataloader, desc=f"Epoch {epoch+1}"):
                if global_step >= self.args.max_steps:
                    break
                
                # Select training step based on optimizer modes
                if self.client_mode == "zo" and self.server_mode == "zo":
                    loss = self._zo_training_step(batch, global_step)
                elif self.client_mode == "fo" and self.server_mode == "fo":
                    loss = self._fo_training_step(batch, global_step)
                elif self.client_mode == "zo" and self.server_mode == "fo":
                    loss = self._hybrid_zo_fo_step(batch, global_step)
                elif self.client_mode == "fo" and self.server_mode == "zo":
                    loss = self._hybrid_fo_zo_step(batch, global_step)
                else:
                    raise ValueError(f"Unsupported mode: client={self.client_mode}, server={self.server_mode}")
                
                total_loss += loss
                log_loss += loss
                log_steps += 1
                global_step += 1
                
                # Update GPU memory peak tracking
                self.memory_tracker.update_peak()
                
                if global_step % self.args.logging_steps == 0:
                    avg_loss = log_loss / log_steps
                    logger.info(f"Step {global_step}/{self.args.max_steps}: loss = {avg_loss:.4f}")
                    log_loss = 0.0
                    log_steps = 0
                
                # Periodic evaluation - matching run.py eval_strategy
                if (eval_strategy == "steps" and 
                    eval_steps > 0 and 
                    global_step % eval_steps == 0):
                    logger.info(f"Running evaluation at step {global_step}...")
                    eval_metrics = self.evaluate([], self._eval_samples)
                    accuracy = eval_metrics.get('accuracy', 0.0)
                    if accuracy > best_accuracy:
                        best_accuracy = accuracy
                    logger.info(f"***** Eval Accuracy at step {global_step}: {accuracy:.4f} (best: {best_accuracy:.4f}) *****")
                    
            if global_step >= self.args.max_steps:
                break
        
        avg_loss = total_loss / global_step if global_step > 0 else 0
        logger.info(f"Training completed. Steps: {global_step}, Avg loss: {avg_loss:.4f}")
        logger.info(f"Communication rounds during training: {self.communication_rounds}")
        if best_accuracy > 0:
            logger.info(f"Best evaluation accuracy during training: {best_accuracy:.4f}")
    
    def forward(self, input_ids, option_len=None):
        """Inference forward - matching run.py Framework.forward()"""
        input_ids_tensor = torch.tensor([input_ids]).to(self.device)
        
        logits = self._forward_to_server(
            input_ids_tensor, input_ids_tensor, None, 0, mode="inference"
        )
        
        logits = logits.to(self.device)
        labels = input_ids_tensor[0, 1:]
        logits = logits[0, :-1]
        log_probs = F.log_softmax(logits, dim=-1)
        
        selected_log_probs = log_probs[torch.arange(len(labels)).to(labels.device), labels]
        selected_log_probs = selected_log_probs.cpu().detach()
        
        return selected_log_probs[-option_len:]
    
    def one_step_pred(self, train_samples, eval_sample, verbose=False):
        """Single prediction - matching run.py Framework.one_step_pred()"""
        encoded_candidates, option_lens = encode_prompt(
            self.task, self.task.get_template(), train_samples, eval_sample, 
            self.tokenizer, max_length=self.args.max_length
        )
        
        outputs = []
        for candidate_id, encoded_candidate in enumerate(encoded_candidates):
            selected_log_probs = self.forward(encoded_candidate, option_len=option_lens[candidate_id])
            outputs.append(selected_log_probs)
        
        scores = [x.mean().item() for x in outputs]
        
        if isinstance(eval_sample.correct_candidate, list):
            correct_candidate_id = [eval_sample.candidates.index(c) for c in eval_sample.correct_candidate]
        else:
            correct_candidate_id = eval_sample.candidates.index(eval_sample.correct_candidate)
        
        return Prediction(correct_candidate=correct_candidate_id, predicted_candidate=int(np.argmax(scores)))
    
    def evaluate(self, train_samples, eval_samples):
        """Evaluation loop - matching run.py Framework.evaluate()"""
        logger.info(f"Evaluating on {len(eval_samples)} samples...")
        
        predictions = []
        for eval_id, eval_sample in enumerate(tqdm(eval_samples, desc="Evaluating")):
            predictions.append(
                self.one_step_pred(train_samples, eval_sample, verbose=(eval_id < 3))
            )
        
        metrics = {"accuracy": calculate_metric(predictions)}
        return metrics
    
    def close(self):
        """Close connection"""
        if self.backend:
            self.backend.close()
            logger.info("Disconnected from server")


# =============================================================================
# MAIN ENTRY POINT (DeComFL-style)
# =============================================================================

def parse_args():
    """Parse arguments using HfArgumentParser for run.py compatibility"""
    parser = HfArgumentParser(DistributedArguments)
    args = parser.parse_args_into_dataclasses()[0]
    return args


def main():
    args = parse_args()
    print(args)
    set_seed(args.seed)
    
    logger.info("=" * 60)
    logger.info(f"Distributed Split Learning - Role: {args.role.upper()}")
    logger.info("=" * 60)
    logger.info(f"Model: {args.model_name}")
    logger.info(f"Task: {args.task_name}")
    logger.info(f"Split Layer: {args.split_layer}")
    logger.info(f"Trainer: {args.trainer}")
    logger.info(f"Optimizer: {args.optimizer}")
    if args.optimizer.lower() == "sgd":
        logger.info(f"SGD Momentum: {args.sgd_momentum}")
    else:
        logger.info(f"Adam Betas: ({args.adam_beta1}, {args.adam_beta2})")
    logger.info(f"ZO: variant={args.zo_variant}, pert={args.zo_perturbation}, eps={args.zo_eps}")
    logger.info(f"Num Pert: {args.num_pert}")
    logger.info(f"LR: {args.learning_rate}")
    logger.info(f"LoRA: {args.lora} (r={args.lora_r}, alpha={args.lora_alpha})")
    logger.info("=" * 60)
    
    if args.role == "server":
        server = DistributedServer(args)
        server.run()
    else:
        # Load task and data (client handles data)
        task = get_task(args.task_name)
        
        logger.info(f"Loading {args.task_name} dataset...")
        train_sets = task.sample_train_sets(
            num_train=args.num_train,
            num_dev=args.num_dev,
            num_eval=args.num_eval,
            num_train_sets=1,
            seed=args.train_set_seed
        )
        train_samples = train_sets[0]
        
        if args.num_dev:
            dev_samples = train_samples[-args.num_dev:]
            train_samples = train_samples[:-args.num_dev]
        else:
            dev_samples = None
        
        eval_samples = task.sample_subset(data_split="valid", seed=args.train_set_seed, num=args.num_eval)
        
        logger.info(f"Loaded {len(train_samples)} train, {len(dev_samples) if dev_samples else 0} dev, {len(eval_samples)} eval")
        
        client = DistributedClient(args, task)
        client.load_model()
        
        try:
            client.connect()
            
            if args.trainer != "none":
                client.train(train_samples, dev_samples if dev_samples else eval_samples)
            
            logger.info("Running evaluation...")
            metrics = client.evaluate([], eval_samples)
            
            if dev_samples:
                dev_metrics = client.evaluate([], dev_samples)
                for m in dev_metrics:
                    metrics["dev_" + m] = dev_metrics[m]
            
            logger.info("=" * 60)
            logger.info("RESULTS")
            logger.info("=" * 60)
            logger.info(f"Accuracy: {metrics.get('accuracy', 'N/A')}")
            if 'dev_accuracy' in metrics:
                logger.info(f"Dev Accuracy: {metrics.get('dev_accuracy', 'N/A')}")
            logger.info(f"Total communication rounds: {client.communication_rounds}")
            logger.info("=" * 60)
            metrics['num_communication_rounds'] = client.communication_rounds
            
            # Print client GPU memory summary
            client.memory_tracker.print_summary()
            
            # Add GPU memory metrics
            client_memory = client.memory_tracker.get_summary()
            metrics.update({
                'client_peak_gpu_memory_allocated_bytes': client_memory['peak_gpu_memory_allocated_bytes'],
                'client_peak_gpu_memory_allocated_gb': client_memory['peak_gpu_memory_allocated_gb'],
                'client_peak_gpu_memory_reserved_bytes': client_memory['peak_gpu_memory_reserved_bytes'],
                'client_peak_gpu_memory_reserved_gb': client_memory['peak_gpu_memory_reserved_gb'],
            })
            
            # Get model parameter counts for communication cost comparison
            client_params = sum(p.numel() for p in client.client_model.parameters())
            trainable_params = sum(p.numel() for p in client.client_model.parameters() if p.requires_grad)
            
            # Add communication cost metrics (DeComFL-style)
            if hasattr(client.backend, 'comm_stats'):
                comm_summary = client.backend.comm_stats.get_summary(
                    model_params=client_params,
                    num_perturbations=args.num_pert
                )
                
                # Print detailed communication summary
                client.backend.comm_stats.print_summary(
                    model_params=client_params,
                    num_perturbations=args.num_pert
                )
                
                # Add to metrics dict
                metrics.update({
                    'total_bytes_transferred': comm_summary['total_bytes'],
                    'total_bytes_formatted': comm_summary['total_bytes_formatted'],
                    'total_bytes_sent': comm_summary['total_bytes_sent'],
                    'total_bytes_received': comm_summary['total_bytes_received'],
                    'avg_bytes_per_round': comm_summary['avg_bytes_per_round'],
                    'forward_payload_bytes': comm_summary['forward_payload_bytes'],
                    'backward_payload_bytes': comm_summary['backward_payload_bytes'],
                    'zo_metadata_bytes': comm_summary['zo_metadata_bytes'],
                    'total_mb': comm_summary['total_mb'],
                })
                
                if 'traditional_fl_bytes' in comm_summary:
                    metrics.update({
                        'traditional_fl_bytes': comm_summary['traditional_fl_bytes'],
                        'savings_vs_traditional_fl': comm_summary['savings_vs_traditional_fl'],
                        'compression_ratio_vs_fl': comm_summary['compression_ratio_vs_fl'],
                    })
            
            print("results:", metrics)
            
        except KeyboardInterrupt:
            logger.info("Training interrupted")
        except Exception as e:
            logger.error(f"Error: {e}")
            import traceback
            traceback.print_exc()
        finally:
            client.close()


if __name__ == '__main__':
    main()

