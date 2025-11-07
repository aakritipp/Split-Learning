import socket
import pickle
import torch
from torch import nn
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from transformers import AutoTokenizer, AutoModelForCausalLM
import os
try:
    from transformers.cache_utils import DynamicCache, StaticCache
except Exception:
    DynamicCache = None
    StaticCache = None
from tqdm import tqdm
import numpy as np
import argparse
import sys
import traceback
from SGDGradientEst import StochasticGradientApproximator
import torch.nn.functional as F
from lora import (
    apply_lora_to_opt, 
    iter_lora_parameters, 
    get_lora_state_dict, 
    load_lora_state_dict
)
from dataset import (
    get_enhanced_dataloaders as get_task_dataloaders,
    get_squad_dataloaders
)
from metrics import (
    calculate_squad_metrics,
    calculate_generation_f1_em, 
    test_generation_simple,
    normalize_answer,
    squad_f1_score,
    squad_exact_match,
    calculate_metrics
)
# Compute/memory & communication tracker
from compute_tracker import ComputeTracker
# Ensure merge_past_key_values is available for prefix-aware eval
try:
    from prefix_kv import (
        merge_past_key_values,
        PrefixKV, 
        load_grad_state_into
    )
except Exception:
    merge_past_key_values = None

def right_trim(input_ids, attention_mask, labels=None):
    """Remove right padding for efficiency"""
    L = attention_mask.sum(dim=1).max().item()
    input_ids = input_ids[:, :int(L)]
    attention_mask = attention_mask[:, :int(L)]
    if labels is not None: 
        labels = labels[:, :int(L)]
    return input_ids, attention_mask, labels

def _right_trim(input_ids, attention_mask, labels):
    with torch.no_grad():
        seq_lens = attention_mask.sum(dim=1)
        max_len = int(seq_lens.max().item())
    return (
        input_ids[:, :max_len],
        attention_mask[:, :max_len],
        labels[:, :max_len] if labels is not None else None,
    )

def _get_hf_token_from_env():
    token = (
        os.environ.get("HF_TOKEN")
        or os.environ.get("HUGGINGFACE_API_TOKEN")
        or os.environ.get("HUGGINGFACEHUB_API_TOKEN")
        or os.environ.get("HUGGINGFACE_TOKEN")
    )
    if token:
        token = token.strip()
    return token or None

def _hf_auth_kwargs():
    token = _get_hf_token_from_env()
    return {"token": token} if token else {}

# === Verification and diagnostics utilities for ZOO ===
def verify_parameter_restoration(model, stage="unknown"):
    """
    Verify that model parameters are restored after ZOO estimation.

    This should be called BEFORE optimizer.step() to ensure we're not
    accidentally updating from a perturbed state.
    """
    import hashlib

    # Compute hash of current parameters
    param_str = ""
    for param in model.parameters():
        if param.requires_grad:
            param_str += str(param.data.cpu().numpy().tobytes())

    current_hash = hashlib.md5(param_str.encode()).hexdigest()

    # Store original hash at start of training
    if not hasattr(verify_parameter_restoration, 'original_hashes'):
        verify_parameter_restoration.original_hashes = {}
        verify_parameter_restoration.original_hashes[id(model)] = current_hash
        print(f"[{stage}] Stored original parameter hash: {current_hash[:8]}")
        return True

    # Check if parameters have been modified unintentionally
    original_hash = verify_parameter_restoration.original_hashes.get(id(model))

    if original_hash != current_hash:
        print(f"[{stage}] WARNING: Parameters changed unexpectedly!")
        print(f"  Original hash: {original_hash[:8]}")
        print(f"  Current hash: {current_hash[:8]}")
        return False

    return True

class ZOOLogger:
    """
    Comprehensive logging for ZOO training to diagnose issues.
    """
    def __init__(self, log_dir, model_name="client"):
        self.log_dir = log_dir
        self.model_name = model_name
        self.history = {
            'step': [],
            'loss': [],
            'grad_norm': [],
            'param_norm': [],
            'param_change': [],
            'lr': [],
            'max_grad': [],
            'min_grad': [],
            'grad_variance': [],
            'predictions': [],
        }

    def log_step(self, step, loss, model, optimizer, predictions=None):
        # Compute norms
        params = [p for p in model.parameters() if p.requires_grad]
        grads = [p.grad for p in params if getattr(p, 'grad', None) is not None]

        param_norm = sum((p.norm().item() ** 2) for p in params) ** 0.5 if params else 0.0
        grad_norm = sum((g.norm().item() ** 2) for g in grads) ** 0.5 if grads else 0.0

        if len(self.history['param_norm']) > 0:
            param_change = abs(param_norm - self.history['param_norm'][-1])
        else:
            param_change = 0.0

        if grads:
            all_grads = torch.cat([g.detach().flatten() for g in grads])
            max_grad = all_grads.max().item()
            min_grad = all_grads.min().item()
            grad_var = all_grads.var().item()
        else:
            max_grad = min_grad = grad_var = 0.0

        try:
            lr = optimizer.param_groups[0]['lr']
        except Exception:
            lr = 0.0

        self.history['step'].append(int(step))
        self.history['loss'].append(float(loss))
        self.history['grad_norm'].append(float(grad_norm))
        self.history['param_norm'].append(float(param_norm))
        self.history['param_change'].append(float(param_change))
        self.history['lr'].append(float(lr))
        self.history['max_grad'].append(float(max_grad))
        self.history['min_grad'].append(float(min_grad))
        self.history['grad_variance'].append(float(grad_var))
        if predictions is not None:
            try:
                self.history['predictions'].append(predictions.detach().cpu().tolist())
            except Exception:
                pass

        print(f"[{self.model_name}] Step {int(step)}:")
        print(f"  Loss: {float(loss):.4f}")
        print(f"  Grad norm: {float(grad_norm):.2f}")
        print(f"  Param norm: {float(param_norm):.2f}")
        print(f"  Param change: {float(param_change):.2e}")
        print(f"  LR: {float(lr):.2e}")
        print(f"  Grad range: [{float(min_grad):.2f}, {float(max_grad):.2f}]")

        if grad_norm < 1e-3:
            print(f"  ⚠️  WARNING: Gradient norm very small ({grad_norm:.2e})")
        if grad_norm > 1e4:
            print(f"  ⚠️  WARNING: Gradient norm very large ({grad_norm:.2e})")
        if param_change < 1e-6 and int(step) > 10:
            print(f"  ⚠️  WARNING: Parameters barely changing ({param_change:.2e})")

    def save(self):
        import json
        os.makedirs(self.log_dir, exist_ok=True)
        filepath = os.path.join(self.log_dir, f'{self.model_name}_history.json')
        with open(filepath, 'w') as f:
            json.dump(self.history, f, indent=2)
        print(f"Saved {self.model_name} history to {filepath}")

    def plot(self):
        try:
            import matplotlib.pyplot as plt
        except Exception:
            print("matplotlib not available; skipping plot")
            return
        fig, axes = plt.subplots(3, 2, figsize=(12, 10))
        fig.suptitle(f'{self.model_name} Training Diagnostics')
        # Loss
        axes[0, 0].plot(self.history['step'], self.history['loss'])
        axes[0, 0].set_xlabel('Step')
        axes[0, 0].set_ylabel('Loss')
        axes[0, 0].set_title('Loss over time')
        axes[0, 0].grid(True)
        # Gradient norm
        axes[0, 1].plot(self.history['step'], self.history['grad_norm'])
        axes[0, 1].set_xlabel('Step')
        axes[0, 1].set_ylabel('Gradient Norm')
        axes[0, 1].set_title('Gradient Norm over time')
        axes[0, 1].set_yscale('log')
        axes[0, 1].grid(True)
        # Parameter norm
        axes[1, 0].plot(self.history['step'], self.history['param_norm'])
        axes[1, 0].set_xlabel('Step')
        axes[1, 0].set_ylabel('Parameter Norm')
        axes[1, 0].set_title('Parameter Norm over time')
        axes[1, 0].grid(True)
        # Parameter change
        axes[1, 1].plot(self.history['step'], self.history['param_change'])
        axes[1, 1].set_xlabel('Step')
        axes[1, 1].set_ylabel('Parameter Change')
        axes[1, 1].set_title('Parameter Change over time')
        axes[1, 1].set_yscale('log')
        axes[1, 1].grid(True)
        # Learning rate
        axes[2, 0].plot(self.history['step'], self.history['lr'])
        axes[2, 0].set_xlabel('Step')
        axes[2, 0].set_ylabel('Learning Rate')
        axes[2, 0].set_title('Learning Rate schedule')
        axes[2, 0].set_yscale('log')
        axes[2, 0].grid(True)
        # Gradient variance
        axes[2, 1].plot(self.history['step'], self.history['grad_variance'])
        axes[2, 1].set_xlabel('Step')
        axes[2, 1].set_ylabel('Gradient Variance')
        axes[2, 1].set_title('Gradient Variance over time')
        axes[2, 1].set_yscale('log')
        axes[2, 1].grid(True)
        plt.tight_layout()
        os.makedirs(self.log_dir, exist_ok=True)
        filepath = os.path.join(self.log_dir, f'{self.model_name}_diagnostics.png')
        plt.savefig(filepath, dpi=150)
        print(f"Saved {self.model_name} diagnostics to {filepath}")

def get_conservative_zoo_args():
    """Return conservative ZOO hyperparameters that should work."""
    return {
        'zoo_lr': 1e-5,
        'mu': 1e-3,
        'num_pert': 1,
        'train_batch_size': 16,
        'lora_r': 8,
        'lora_alpha': 16,
        'weight_decay': 0.001,
        'warmup_steps': 200,
        'scheduler': 'constant_with_warmup',
        'zoo_max_grad_value': 0.0,
        'estimator': 'central',
        'zoo_perturbation_type': 'gaussian',
        'max_steps': 2000,
        'eval_steps': 50,
    }

def try_hf_login():
    token = _get_hf_token_from_env()
    if not token:
        return False
    try:
        from huggingface_hub import login as hf_login
        # Avoid modifying git credential store on CI
        hf_login(token=token, add_to_git_credential=False)
        print("✅ Hugging Face login succeeded via HF_TOKEN")
        return True
    except Exception as _e:
        # Non-fatal; we'll rely on passing token directly to loaders
        print(f"⚠️ Hugging Face login skipped/failed: {_e}")
        return False
def adapt_batch(batch, device):
    input_ids = batch["input_ids"].to(device)
    attention_mask = batch["attention_mask"].to(device)
    labels = batch.get("labels", None)
    if isinstance(labels, torch.Tensor):
        labels = labels.to(device)
    prompt_text = batch.get("prompt_text", None) or []
    text_target = batch.get("text_target", None) or []
    meta = batch.get("meta", None) or [{} for _ in range(input_ids.size(0))]
    return input_ids, attention_mask, labels, prompt_text, text_target, meta

def _assert_only_expected_trainables(module: nn.Module, mode: str, layer_range=None, side: str = None):
    for n, p in module.named_parameters():
        if mode == "prefix":
            if side == "client":
                # Only client KV prefixes should be trainable
                is_allowed = n.startswith("kv.")
            elif side == "server":
                # Only server prefixes should be trainable
                is_allowed = n.startswith("server_kv.")
            else:
                # Generic fallback: any KV/prefix-like params are allowed
                is_allowed = (n.startswith("kv.") or n.startswith("server_kv.") or ("prefix" in n))

            ok = ("lora_A" not in n and "lora_B" not in n) and ((is_allowed) == p.requires_grad)

        elif mode == "lora":
            # Only LoRA adapters should be trainable
            is_lora = ("lora_A" in n) or ("lora_B" in n)
            ok = (is_lora == p.requires_grad) and ("kv." not in n) and ("server_kv." not in n)

        elif mode == "full":
            # Full fine-tuning: disallow LoRA/prefix params from being trainable; base params may be either
            is_disallowed = (n.startswith("kv.") or n.startswith("server_kv.") or ("prefix" in n) or ("lora_A" in n) or ("lora_B" in n))
            ok = not (is_disallowed and p.requires_grad)

        else:  # none
            ok = (p.requires_grad is False)

        assert ok, f"Unexpected trainable param in {mode} mode{f' ({side})' if side else ''}: {n} requires_grad={p.requires_grad}"


def _neg_inf(dtype: torch.dtype) -> float:
    # Use the representable minimum as the additive mask value
    return torch.finfo(dtype).min

def _get_current_lr(optimizer: optim.Optimizer) -> float:
    """Return the learning rate of the first param group for logging."""
    try:
        return float(optimizer.param_groups[0].get("lr", 0.0))
    except Exception:
        return 0.0

# --- Scheduler helpers (support warmup + non-plateau schedulers) ---
def _compute_warmup_steps(max_steps: int, warmup_steps: int = 0, warmup_ratio: float = 0.0) -> int:
    try:
        if warmup_steps and warmup_steps > 0:
            return int(warmup_steps)
        if warmup_ratio and warmup_ratio > 0.0:
            return max(0, int(float(max_steps) * float(warmup_ratio)))
    except Exception:
        pass
    return 0

def _build_lambda_scheduler(optimizer: optim.Optimizer, kind: str, max_steps: int, warmup_steps: int) -> optim.lr_scheduler.LambdaLR:
    import math
    kind = str(kind).lower()
    def lr_lambda(step: int) -> float:
        s = int(step)
        if warmup_steps > 0 and s < warmup_steps:
            return float(s) / float(max(1, warmup_steps))
        progress = (s - warmup_steps) / float(max(1, max_steps - warmup_steps))
        progress = max(0.0, min(1.0, progress))
        if kind == 'linear':
            return max(0.0, 1.0 - progress)
        if kind == 'cosine':
            return 0.5 * (1.0 + math.cos(math.pi * progress))
        # none: constant after warmup
        return 1.0
    return optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)

class _SchedAdapter:
    """Adapter that lets us call step(metric) for Plateau and step() for LambdaLR transparently."""
    def __init__(self, scheduler, plateau: bool):
        self.scheduler = scheduler
        self.is_plateau = bool(plateau)
    def step(self, metric: float = None):
        try:
            if self.is_plateau:
                if metric is None:
                    # Fallback when metric not provided
                    self.scheduler.step(0.0)
                else:
                    self.scheduler.step(metric)
            else:
                self.scheduler.step()
        except Exception:
            # Be forgiving if the underlying scheduler signature differs
            try:
                self.scheduler.step()
            except Exception:
                pass

## Deprecated helpers removed in favor of unified --precision

def _refresh_eval_prefixes(full_model, client_model, trainer, args):
    """
    Pull the latest server prefix snapshot for eval,
    attach live client prefixes, and enable prefix-aware eval.
    Safe fallback to legacy eval if anything fails.
    """
    # LoRA mode: do not use prefix-aware eval or request server KV
    if args.tuning == "lora":
        # In LoRA mode, skip split eval handshake entirely; use local frozen eval only
        full_model.enable_prefix_eval(False)
        return False
    
    full_model.attach_live_client_kv(client_model.kv)
    try:
        trainer.send_data({"type": "get_server_kv_state"})
        resp = trainer.receive_data()
        if isinstance(resp, dict) and resp.get("type") == "server_kv_state":
            state = resp["state"]
            # Guard against LoRA/empty state
            if state and state.get("k") is not None and state.get("v") is not None:
                full_model.load_server_kv_state(state)
                full_model.enable_prefix_eval(True)
                return True
        
    except Exception as e:
        print(f"⚠️ Eval prefixes not refreshed: {e}")
    full_model.enable_prefix_eval(False)   # fallback to legacy eval
    return False

def _build_self_attn_mask(attention_mask: torch.Tensor,
                          tgt_len: int,
                          prefix_len: int,
                          dtype: torch.dtype,
                          device: torch.device) -> torch.Tensor:
    """
    Construct OPT-style additive attention mask with:
      - causal masking over current tokens
      - left prefix KV of length P (always visible)
      - padding masking from `attention_mask` (0 -> masked)
    Returns shape [B, 1, tgt_len, prefix_len + tgt_len] with 0 for allowed, -inf for masked.
    """
    bsz = attention_mask.size(0)

    # Causal part over current tokens (S x S): 0 on/below diag, -inf above
    causal = torch.triu(
        torch.full((tgt_len, tgt_len), _neg_inf(dtype), device=device),
        diagonal=1
    )
    # Prepend prefix block (always visible -> zeros)
    if prefix_len > 0:
        prefix_block = torch.zeros((tgt_len, prefix_len), dtype=dtype, device=device)
        base = torch.cat([prefix_block, causal], dim=-1)  # [S, P+S]
    else:
        base = causal  # [S, S]

    # Expand to [B,1,S,P+S]
    attn = base.unsqueeze(0).unsqueeze(1).expand(bsz, 1, tgt_len, prefix_len + tgt_len)

    # Source-side padding: broadcast to [B,1,1,P+S] and add
    pad = (1.0 - attention_mask.to(dtype))  # [B,S], 1 where pad
    if prefix_len > 0:
        src_pad = torch.cat([torch.zeros((bsz, prefix_len), dtype=dtype, device=device), pad], dim=-1)
    else:
        src_pad = pad
    attn = attn + src_pad.view(bsz, 1, 1, prefix_len + tgt_len) * _neg_inf(dtype)
    return attn

def _client_forward_to_cut_payload(
    client_wrap,                    # clientKVOnly instance
    input_ids: torch.Tensor,
    attention_mask: torch.Tensor,
    labels: torch.Tensor,
    send_fp16: bool = True,):
    """
    Run embeddings + layers [0..cut-1] with client prefixes as per-layer past_key_value to produce h_cut.
    Returns (h_cut_live, payload_to_server).
    """
    base_model = client_wrap.base_model
    decoder    = base_model.model.decoder
    cut        = client_wrap.cut_layer
    dtype      = next(base_model.parameters()).dtype

    input_ids, attention_mask, labels = _right_trim(input_ids, attention_mask, labels)

    # Embeddings + positions (OPT expects mask-based positions)
    # x = decoder.embed_tokens(input_ids) * decoder.embed_scale
    x = decoder.embed_tokens(input_ids)
    scale = getattr(decoder, "embed_scale", None)
    if scale is not None:
        x = x * scale  # only scale when attribute exists

    pos = None
    embed_pos = getattr(decoder, "embed_positions", None)
    if embed_pos is not None:
        try:
            # some forks accept attention_mask (your earlier ZeroPositionalEmbedding matched this)
            pos = embed_pos(attention_mask, position_ids=None, past_key_values_length=0)
        except TypeError:
            # stock HF OPT uses (input_shape, past_key_values_length)
            pos = embed_pos(input_ids.shape, past_key_values_length=0)


    # pos = decoder.embed_positions(attention_mask, position_ids=None, past_key_values_length=0)
    if pos is not None:
        x = x + pos
    if getattr(decoder, "layernorm_embedding", None) is not None:
        x = decoder.layernorm_embedding(x)
    # x = decoder.dropout(x)
    # Dropout can be either a module or a float p in some OPT forks
    drop_attr = getattr(decoder, "dropout", None)
    if callable(drop_attr):
        # Standard HF: decoder.dropout is nn.Dropout
        x = drop_attr(x)
    else:
        # Some forks store p as a float (e.g., decoder.dropout == 0.1)
        if isinstance(drop_attr, (float, int)):
            p = float(drop_attr)
        else:
            # Fallbacks: try decoder.dropout_p or config.dropout, else 0.0
            p = getattr(decoder, "dropout_p", None)
            if p is None:
                p = float(getattr(getattr(client_wrap.base_model, "config", object()), "dropout", 0.0))
        if p and p > 0.0:
            x = F.dropout(
                x,
                p=p,
                training=decoder.training if hasattr(decoder, "training") else client_wrap.base_model.training,
            )
        # else: no-op if p == 0.0

    # Build attention mask per-layer to match the actual prefix length of that layer
    tgt_len = x.shape[1]

    # Build per-layer past_kv from client prefixes
    bsz = input_ids.size(0)
    client_past = client_wrap.kv.get_local_past(bsz)  # {layer_idx: (k,v)} with [B,H,P,D]

    # Run client side layers [0..cut-1]
    for li in range(cut):
        layer = decoder.layers[li]
        pkv   = client_past.get(li, None)
        # pkv = client_past.get(li, None)
        if pkv is not None:
            # pkv is (k,v) shaped [B,H,P,D] each — wrap to cache-like object
            pkv = _PrefixConcatCache(pkv[0], pkv[1])

        # Determine this layer's prefix length from client_past to avoid mask/prefix mismatch
        try:
            prefix_len_cur = int(client_past.get(li, (None, None))[0].shape[2]) if client_past.get(li, None) is not None else 0
        except Exception:
            prefix_len_cur = 0

        attn_mask_4d = _build_self_attn_mask(
            attention_mask=attention_mask,
            tgt_len=tgt_len,
            prefix_len=prefix_len_cur,
            dtype=x.dtype,
            device=x.device,
        )

        layer_out = layer(
            x,
            attention_mask=attn_mask_4d,
            layer_head_mask=None,
            past_key_value=pkv,        # _PrefixConcatCache or None
            output_attentions=False,
            use_cache=False,
        )
        # HF may return Tensor or a tuple; when use_cache=False & output_attentions=False it's often a Tensor
        x = layer_out[0] if isinstance(layer_out, (tuple, list)) else layer_out


    # Keep live tensor for SGD backprop
    h_cut_live = x
    if not h_cut_live.requires_grad:
        # if prefixes aren’t yet in graph, still make it trainable to unblock
        h_cut_live = x.detach().requires_grad_(True)

    # Detached payload (wire-friendly)
    h_cut_send = h_cut_live.detach()
    h_cut_send = (h_cut_send.to(torch.float16) if send_fp16 else h_cut_send.to(dtype)).cpu()

    payload = {
        "h_cut": h_cut_send,
        "attention_mask": attention_mask.cpu(),
        "labels": labels.cpu() if labels is not None else None,
        "cut_layer": cut,
    }
    return h_cut_live, payload

class _PrefixConcatCache:
    """
    Minimal cache adapter for HF SelfAttention that expects an object with .update(...)
    It concatenates the stored prefix (k,v) along the sequence length dimension.
    """
    def __init__(self, k_prefix: torch.Tensor, v_prefix: torch.Tensor):
        # Expect [B, H, P, D]
        self.kp = k_prefix
        self.vp = v_prefix

    def update(self, *args, **kwargs):
        # Works with either positional or keyword signatures used by HF
        if len(args) >= 2:
            key_states, value_states = args[0], args[1]
        else:
            key_states  = kwargs.get("key_states")
            value_states = kwargs.get("value_states")

        # Move/cast prefix to match current states
        kp = self.kp.to(device=key_states.device, dtype=key_states.dtype)
        vp = self.vp.to(device=value_states.device, dtype=value_states.dtype)

        # Concat along seq-len axis (dim=2): [B,H,P+S,D]
        k_cat = torch.cat([kp, key_states], dim=2)
        v_cat = torch.cat([vp, value_states], dim=2)
        return (k_cat, v_cat)

class LoRAclientModel(nn.Module):
    """
    client-side when tuning=LoRA. Keeps full base_model, injects LoRA only into layers [0..cut-1].
    Presents no-prefix stubs so forward/masks pipeline stays unified.
    """
    def __init__(self, model_name: str, cut_layer: int,
                 r: int = 8, alpha: int = 16, dropout: float = 0.0,
                 targets=("q_proj","v_proj"), torch_dtype: torch.dtype = torch.float32):
        super().__init__()
        self.base_model = AutoModelForCausalLM.from_pretrained(
            model_name, torch_dtype=torch_dtype, device_map=None
        )
        for p in self.base_model.parameters():
            p.requires_grad = False
        self.total_layers = self.base_model.config.num_hidden_layers
        self.cut_layer = cut_layer

        apply_lora_to_opt(
            self.base_model,
            targets=tuple(targets),
            layer_range=(0, cut_layer - 1),
            r=r, lora_alpha=alpha, lora_dropout=dropout
        )

        # keep interface consistent with prefix path
        class _NoPrefixStub:
            def get_local_past(self, bsz): return {}
            def set_requires_grad(self, flag: bool): pass
        self.client_kv = _NoPrefixStub()
        self.server_kv_mirror = _NoPrefixStub()

        # Provide a minimal KV stub so shared code paths can reference client_model.kv safely
        class _EmptyKV:
            def __init__(self):
                # tensors with 4 dims so shape[-2] exists; P dimension is 0
                self.k = torch.zeros((0, 0, 0, 0))
                self.v = torch.zeros((0, 0, 0, 0))
            def get_local_past(self, bsz):
                return {}
            def parameters(self):
                return iter(())
            def set_requires_grad(self, flag: bool):
                return None
            def state_dict(self):
                return {"k": self.k, "v": self.v}

        self.kv = _EmptyKV()

    def state_dict_kv(self):
        """Return an empty KV state to satisfy interfaces expecting a client KV snapshot."""
        try:
            # Prefer the stub tensors so shapes/types are tensors
            return {"k": self.kv.k.detach().cpu(), "v": self.kv.v.detach().cpu()}
        except Exception:
            # Ultimate fallback
            return {"k": torch.zeros(0), "v": torch.zeros(0)}

    def trainable_parameters(self):
        return iter_lora_parameters(self.base_model, layer_range=(0, self.cut_layer-1))

class clientFullModel(nn.Module):
    """
    client-side when tuning=full. Keeps full base_model; trains embeddings + layers [0..cut-1].
    Presents empty KV stubs so shared forward-to-cut path can be reused.
    """
    def __init__(self, model_name: str, cut_layer: int, torch_dtype: torch.dtype = torch.float32):
        super().__init__()
        self.base_model = AutoModelForCausalLM.from_pretrained(
            model_name, torch_dtype=torch_dtype, device_map=None
        )
        # Full fine-tuning: enable grads only for embeddings + layers [0..cut-1]
        for p in self.base_model.parameters():
            p.requires_grad = False
        self.total_layers = self.base_model.config.num_hidden_layers
        self.cut_layer = cut_layer

        # keep interface consistent with prefix/LoRA paths
        class _NoPrefixStub:
            def get_local_past(self, bsz): return {}
            def set_requires_grad(self, flag: bool): pass
        self.client_kv = _NoPrefixStub()
        self.server_kv_mirror = _NoPrefixStub()

        class _EmptyKV:
            def __init__(self):
                self.k = torch.zeros((0, 0, 0, 0))
                self.v = torch.zeros((0, 0, 0, 0))
            def get_local_past(self, bsz):
                return {}
            def parameters(self):
                return iter(())
            def set_requires_grad(self, flag: bool):
                return None
            def state_dict(self):
                return {"k": self.k, "v": self.v}

        self.kv = _EmptyKV()

    def state_dict_kv(self):
        try:
            return {"k": self.kv.k.detach().cpu(), "v": self.kv.v.detach().cpu()}
        except Exception:
            return {"k": torch.zeros(0), "v": torch.zeros(0)}

    def trainable_parameters(self):
        # Return only parameters that require grad
        return (p for p in self.base_model.parameters() if p.requires_grad)

class clientKVOnly(nn.Module):
    """
    Minimal client-side holder for KV prefixes on the first `cut_layer` layers.
    (No forward compute here to keep changes minimal; server runs the full model and uses these prefixes.)
    """
    def __init__(self, model_name, cut_layer, num_prefix=10, torch_dtype: torch.dtype = torch.float32):
        super().__init__()
        # load config to size params correctly
        tmp = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype=torch_dtype,
            device_map=None,
            **_hf_auth_kwargs(),
        )
        self.total_layers = tmp.config.num_hidden_layers
        self.cut_layer = cut_layer
        self.kv = PrefixKV(tmp.config, list(range(0, cut_layer)), num_prefix=num_prefix, device=tmp.device)
        for p in self.kv.parameters():
            p.requires_grad = True
        self.attach_partial_model(model_name)
        # we do not keep the full model in memory here to save RAM
        del tmp
        torch.cuda.empty_cache()
    
    def attach_partial_model(self, model_name: str):
        """
        Load an OPT-style LM and keep only embeddings + first `cut_layer` decoder blocks.
        Enough to produce h_cut on the client.
        """

        base = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype=next(tmp.parameters()).dtype if hasattr(tmp, 'parameters') else torch_dtype,
            device_map=None,
            **_hf_auth_kwargs(),
        )
        dec = base.model.decoder
        # keep only [0..cut_layer-1]
        dec.layers = nn.ModuleList(list(dec.layers[: self.cut_layer]))
        self.base_model = base.eval()
        # Freeze all base model params on client in prefix mode; only KV prefixes train
        for p in self.base_model.parameters():
            p.requires_grad = False

    def state_dict_kv(self):
        # minimal state dict to send to server
        return {"k": self.kv.k.detach().cpu(), "v": self.kv.v.detach().cpu()}

def safe_get_hf_tokenizer(model_name):
    """Safe tokenizer loading with error handling"""
    try:
        tokenizer = AutoTokenizer.from_pretrained(model_name, **_hf_auth_kwargs())
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
        return tokenizer
    except Exception as e:
        print(f"Failed to load tokenizer for {model_name}: {e}")
        raise


class FullLLMModel(nn.Module):
    """Frozen full model used only for monitoring/evaluation."""
    def __init__(self, model_name, cut_layer, num_prefix=5, torch_dtype: torch.dtype = torch.float32):
        super(FullLLMModel, self).__init__()
        self.base_model = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype=torch_dtype,
            device_map=None,
            **_hf_auth_kwargs(),
        )
        self.total_layers = self.base_model.config.num_hidden_layers
        self.cut_layer    = int(cut_layer)
        self.num_prefix = num_prefix
        for p in self.base_model.parameters():
            p.requires_grad = False
        self._use_prefix_eval = False
        self._client_kv_live  = None            # live reference (client side)
        self._server_kv_eval  = None            # local PrefixKV snapshot for server side

        print("Full model ready; prefix-aware eval OFF by default (legacy behavior).")

    def attach_live_client_kv(self, client_kv_module: PrefixKV):
        """Point eval model to the live client PrefixKV (no copy)."""
        self._client_kv_live = client_kv_module

    @torch.no_grad()
    def load_server_kv_state(self, state: dict):
        """
        Load a snapshot of server prefixes into a local PrefixKV for eval.
        Expects {'k': tensor[Lc,H,P,D], 'v': tensor[Lc,H,P,D]} for layers [cut..L-1].
        """
        device = next(self.base_model.parameters()).device
        dtype  = next(self.base_model.parameters()).dtype

        if self._server_kv_eval is None:
            # Build eval-side PrefixKV container for server half
            self._server_kv_eval = PrefixKV(
                self.base_model.config,
                list(range(self.cut_layer, self.total_layers)),
                num_prefix=state["k"].shape[-2],
                device=device,
                dtype=dtype,
            )
        # Copy weights
        self._server_kv_eval.k.copy_(state["k"].to(device=device, dtype=self._server_kv_eval.k.dtype))
        self._server_kv_eval.v.copy_(state["v"].to(device=device, dtype=self._server_kv_eval.v.dtype))

    def enable_prefix_eval(self, flag: bool = True):
        """Turn prefix-aware eval on/off."""
        self._use_prefix_eval = bool(flag)

    def forward(self, input_ids, attention_mask, labels=None):
        """
        If prefix-aware eval is enabled and both halves are present,
        evaluate with merged past_key_values + explicit position_ids.
        Otherwise, fall back to the legacy frozen no-prefix path.
        """
        # Happy-path: prefix-aware eval
        if self._use_prefix_eval and self._client_kv_live is not None and self._server_kv_eval is not None:
            bsz, seq_len = input_ids.size(0), input_ids.size(1)

            # Build per-layer K/V caches from both halves (legacy list of tuples)
            client_past = self._client_kv_live.get_local_past(bsz)   # {layer_idx: (k,v)}
            server_past = self._server_kv_eval.get_local_past(bsz)   # {layer_idx: (k,v)}
            legacy_cache = merge_past_key_values(self.total_layers, client_past, server_past)

            # Normalize per-layer past lengths to a uniform max P across layers
            # This avoids width mismatches inside attention when some layers have different P.
            try:
                cfg = getattr(self.base_model, 'config', None)
                num_heads = int(getattr(cfg, 'num_attention_heads', 1)) if cfg is not None else legacy_cache[0][0].shape[1] if legacy_cache and legacy_cache[0] is not None else 1
                head_dim = int(getattr(cfg, 'hidden_size', num_heads)) // int(max(1, num_heads)) if cfg is not None else legacy_cache[0][0].shape[-1] if legacy_cache and legacy_cache[0] is not None else 1
            except Exception:
                num_heads = 1
                head_dim = 1

            # Determine maximum past length across layers
            past_len_max = 0
            for kv in legacy_cache:
                if kv is not None and isinstance(kv, (tuple, list)) and isinstance(kv[0], torch.Tensor):
                    try:
                        past_len_max = max(past_len_max, int(kv[0].shape[-2]))
                    except Exception:
                        pass

            # Pad each layer's K/V to past_len_max; fill missing layers with zeros so mask width matches everywhere
            if past_len_max > 0:
                device = next(self.base_model.parameters()).device
                dtype  = next(self.base_model.parameters()).dtype
                padded_legacy = []
                for kv in legacy_cache:
                    if kv is None:
                        k_pad = torch.zeros((bsz, num_heads, past_len_max, head_dim), dtype=dtype, device=device)
                        v_pad = torch.zeros((bsz, num_heads, past_len_max, head_dim), dtype=dtype, device=device)
                        padded_legacy.append((k_pad, v_pad))
                    else:
                        k, v = kv
                        try:
                            cur_len = int(k.shape[-2])
                        except Exception:
                            cur_len = past_len_max
                        if cur_len == past_len_max:
                            padded_legacy.append((k, v))
                        elif cur_len < past_len_max:
                            pad_len = past_len_max - cur_len
                            z_k = torch.zeros((k.shape[0], k.shape[1], pad_len, k.shape[3]), dtype=k.dtype, device=k.device)
                            z_v = torch.zeros((v.shape[0], v.shape[1], pad_len, v.shape[3]), dtype=v.dtype, device=v.device)
                            # Prepend zeros so real prefixes remain the most recent segment
                            padded_legacy.append((torch.cat([z_k, k], dim=2), torch.cat([z_v, v], dim=2)))
                        else:
                            # Truncate if somehow longer (shouldn't happen in practice)
                            padded_legacy.append((k[:, :, -past_len_max:, :], v[:, :, -past_len_max:, :]))
            else:
                padded_legacy = list(legacy_cache)

            # Use the uniform max past length for positions
            past_len = int(past_len_max)

            # Use explicit positions (matches server forward_full)
            position_ids = torch.arange(
                past_len, past_len + seq_len, device=input_ids.device, dtype=torch.long
            ).unsqueeze(0).expand(bsz, -1)

            # Convert to HF Cache object if available for newer transformers
            cache_obj = padded_legacy
            try:
                if StaticCache is not None and hasattr(StaticCache, "from_legacy_cache"):
                    cache_obj = StaticCache.from_legacy_cache(tuple(padded_legacy))
                elif DynamicCache is not None and hasattr(DynamicCache, "from_legacy_cache"):
                    cache_obj = DynamicCache.from_legacy_cache(tuple(padded_legacy))
            except Exception:
                cache_obj = padded_legacy

            return self.base_model(
                input_ids=input_ids,
                attention_mask=None,            # use explicit positions instead
                position_ids=position_ids,
                labels=labels,
                past_key_values=cache_obj,
                use_cache=False,
            )

        # Legacy fallback: frozen, no prefixes
        return self.base_model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            labels=labels,
            use_cache=False,
        )

    # --- client.py (inside class FullLLMModel) ---
    def generate(self, input_ids, attention_mask=None, **kwargs):
        # Prefix-aware generate (matches your prefix-aware forward)
        if self._use_prefix_eval and self._client_kv_live is not None and self._server_kv_eval is not None:
            bsz, seq_len = input_ids.shape
            client_past = self._client_kv_live.get_local_past(bsz)
            server_past = self._server_kv_eval.get_local_past(bsz)
            legacy_cache = merge_past_key_values(self.total_layers, client_past, server_past)

            # Normalize per-layer past lengths to uniform max P
            try:
                cfg = getattr(self.base_model, 'config', None)
                num_heads = int(getattr(cfg, 'num_attention_heads', 1)) if cfg is not None else legacy_cache[0][0].shape[1] if legacy_cache and legacy_cache[0] is not None else 1
                head_dim = int(getattr(cfg, 'hidden_size', num_heads)) // int(max(1, num_heads)) if cfg is not None else legacy_cache[0][0].shape[-1] if legacy_cache and legacy_cache[0] is not None else 1
            except Exception:
                num_heads = 1
                head_dim = 1

            past_len_max = 0
            for kv in legacy_cache:
                if kv is not None and isinstance(kv, (tuple, list)) and isinstance(kv[0], torch.Tensor):
                    try:
                        past_len_max = max(past_len_max, int(kv[0].shape[-2]))
                    except Exception:
                        pass

            if past_len_max > 0:
                device = next(self.base_model.parameters()).device
                dtype  = next(self.base_model.parameters()).dtype
                padded_legacy = []
                for kv in legacy_cache:
                    if kv is None:
                        k_pad = torch.zeros((bsz, num_heads, past_len_max, head_dim), dtype=dtype, device=device)
                        v_pad = torch.zeros((bsz, num_heads, past_len_max, head_dim), dtype=dtype, device=device)
                        padded_legacy.append((k_pad, v_pad))
                    else:
                        k, v = kv
                        try:
                            cur_len = int(k.shape[-2])
                        except Exception:
                            cur_len = past_len_max
                        if cur_len == past_len_max:
                            padded_legacy.append((k, v))
                        elif cur_len < past_len_max:
                            pad_len = past_len_max - cur_len
                            z_k = torch.zeros((k.shape[0], k.shape[1], pad_len, k.shape[3]), dtype=k.dtype, device=k.device)
                            z_v = torch.zeros((v.shape[0], v.shape[1], pad_len, v.shape[3]), dtype=v.dtype, device=v.device)
                            padded_legacy.append((torch.cat([z_k, k], dim=2), torch.cat([z_v, v], dim=2)))
                        else:
                            padded_legacy.append((k[:, :, -past_len_max:, :], v[:, :, -past_len_max:, :]))
            else:
                padded_legacy = list(legacy_cache)

            past_len = int(past_len_max)

            position_ids = torch.arange(past_len, past_len + seq_len, device=input_ids.device, dtype=torch.long)
            position_ids = position_ids.unsqueeze(0).expand(bsz, -1)

            # Let HF generate with the prefilled cache; attention_mask becomes unnecessary
            cache_obj = padded_legacy
            try:
                if StaticCache is not None and hasattr(StaticCache, "from_legacy_cache"):
                    cache_obj = StaticCache.from_legacy_cache(tuple(padded_legacy))
                elif DynamicCache is not None and hasattr(DynamicCache, "from_legacy_cache"):
                    cache_obj = DynamicCache.from_legacy_cache(tuple(padded_legacy))
            except Exception:
                cache_obj = padded_legacy  # fallback

            return self.base_model.generate(
                input_ids=input_ids,
                position_ids=position_ids,
                past_key_values=cache_obj,
                **kwargs
            )

        # Fallback: no prefixes known
        return self.base_model.generate(input_ids=input_ids, attention_mask=attention_mask, **kwargs)


class Trainer:
    """Handles communication with the server"""
    def __init__(self, conn, tracker=None):
        self.conn = conn
        self.tracker = tracker
    
    def send_data(self, data):
        try:
            # If message carries an explicit mode, honor it for this send
            if getattr(self, 'tracker', None) is not None:
                try:
                    if isinstance(data, dict) and ('mode' in data):
                        self.tracker.set_phase('eval' if str(data.get('mode')).lower() == 'eval' else 'train')
                    # Track logical payload size and a communication round
                    self.tracker.track_send(data)
                except Exception:
                    pass
            serialized = pickle.dumps(data, protocol=pickle.HIGHEST_PROTOCOL)
            self.conn.sendall(len(serialized).to_bytes(4, 'big'))
            self.conn.sendall(serialized)
        except Exception as e:
            print(f"❌ Failed to send data: {e}")
            raise
    
    def receive_data(self):
        try:
            length = int.from_bytes(self.conn.recv(4), 'big')
            data = b''
            while len(data) < length:
                data += self.conn.recv(length - len(data))
            obj = pickle.loads(data)
            if getattr(self, 'tracker', None) is not None:
                try:
                    self.tracker.track_receive(obj)
                except Exception:
                    pass
            return obj
        except Exception as e:
            print(f"❌ Failed to receive data: {e}")
            raise
            
def evaluate_model(model, test_loader, device, tokenizer, args, client_model=None, trainer=None, tracker=None):
    """Task-aware evaluation; generation metrics for QA/summary, class acc for SST-2."""
    task_name = str(getattr(args, 'task', 'squad')).upper()
    print(f"  Starting {task_name} evaluation...")
    
    # Only test generation if not a pure classification task
    if getattr(args, 'task', 'squad') != 'sst2':
        print("  Testing generation capability...")
        gen_works = test_generation_simple(model, tokenizer, device)
        print(f"  Generation test result: {'✅ PASS' if gen_works else '❌ FAIL'}")
    
    model.eval()
    total_loss = 0.0
    total_answer_accuracy = 0.0
    total_f1 = 0.0
    total_em = 0.0
    num_batches = 0

    with torch.no_grad():
        # Try to refresh prefix-aware eval when split context is available
        # IMPORTANT: In LoRA/Full, we will build a combined model ONCE per eval to avoid per-batch fetch.
        split_eval = False
        if client_model is not None and trainer is not None:
            if getattr(args, 'tuning', 'prefix') == 'lora':
                # Avoid split-eval; combined eval used instead
                split_eval = False
            elif getattr(args, 'tuning', 'prefix') == 'full':
                # Combined eval for full finetune
                split_eval = False
            else:
                _refreshed = _refresh_eval_prefixes(model, client_model, trainer, args)
                if not _refreshed:
                    print("⚠️ Prefix-aware eval unavailable; falling back to frozen no-prefix eval.")
                else:
                    split_eval = True

        # Build a single combined eval model for LoRA/Full to reduce communication to once per eval
        combined_eval = None
        combined_dev = device
        if (client_model is not None and trainer is not None) and (getattr(args, 'tuning', 'prefix') in ('lora', 'full')):
            try:
                # Honor eval_on_cpu for LoRA combined eval to reduce GPU memory
                dev_for_combined = (torch.device('cpu') if (bool(getattr(args, 'eval_on_cpu', False)) and str(getattr(args, 'tuning', 'prefix')) == 'lora') else device)
                full_eval = AutoModelForCausalLM.from_pretrained(
                    args.model_name,
                    torch_dtype=(torch.float16 if str(getattr(args, 'precision', 'fp32')).lower() == 'fp16' else torch.float32),
                    device_map=None
                ).to(dev_for_combined)
                for p in full_eval.parameters():
                    p.requires_grad = False

                total_layers_local = full_eval.config.num_hidden_layers
                if getattr(args, 'tuning', 'prefix') == 'lora':
                    # Apply LoRA to both halves and load client+server once
                    apply_lora_to_opt(full_eval, targets=tuple(args.lora_targets.split(',')), layer_range=(0, args.cut_layer-1), r=args.lora_r, lora_alpha=args.lora_alpha, lora_dropout=args.lora_dropout)
                    apply_lora_to_opt(full_eval, targets=tuple(args.lora_targets.split(',')), layer_range=(args.cut_layer, total_layers_local-1), r=args.lora_r, lora_alpha=args.lora_alpha, lora_dropout=args.lora_dropout)

                    client_state = get_lora_state_dict(getattr(client_model, 'base_model', client_model), layer_range=(0, args.cut_layer-1))
                    _ = load_lora_state_dict(full_eval, client_state)

                    trainer.send_data({"type": "get_server_lora_state", "mode": "eval"})
                    resp = trainer.receive_data()
                    if isinstance(resp, dict) and resp.get("type") == "server_lora_state" and resp.get("ok", False):
                        server_state = resp.get("state", {})
                        _ = load_lora_state_dict(full_eval, server_state)
                    else:
                        print("⚠️ Could not fetch server LoRA state; will fall back to split-eval if available")
                        split_eval = True
                        full_eval = None
                else:
                    # Full finetune: stitch client+server weights once
                    fe_sd = full_eval.state_dict()
                    srv_sd = getattr(client_model, 'base_model', client_model).state_dict()
                    # Copy client half: embeddings + layers [0..cut-1]
                    for k, v in srv_sd.items():
                        take = False
                        if k.startswith('model.decoder.embed_'):
                            take = True
                        elif '.model.decoder.layers.' in ('.' + k):
                            try:
                                after = k.split('model.decoder.layers.')[1]
                                idx = int(after.split('.')[0])
                                if idx < int(args.cut_layer):
                                    take = True
                            except Exception:
                                pass
                        if take and k in fe_sd:
                            fe_sd[k] = v.to(device=fe_sd[k].device, dtype=fe_sd[k].dtype)

                    # Fetch server half ONCE
                    trainer.send_data({"type": "get_server_full_state", "mode": "eval"})
                    resp = trainer.receive_data()
                    if isinstance(resp, dict) and resp.get("type") == "server_full_state" and resp.get("ok", False):
                        cl_state = resp.get("state", {})
                        for k, v in cl_state.items():
                            if k in fe_sd:
                                tv = v
                                if not torch.is_tensor(tv):
                                    tv = torch.as_tensor(tv)
                                fe_sd[k] = tv.to(device=fe_sd[k].device, dtype=fe_sd[k].dtype)
                        _ = full_eval.load_state_dict(fe_sd, strict=False)
                    else:
                        print("⚠️ Could not fetch server full state; will fall back to split-eval if available")
                        split_eval = True
                        full_eval = None

                combined_eval = full_eval
                combined_dev = dev_for_combined
            except Exception as e:
                print(f"⚠️ Combined eval assembly failed: {e}")
                combined_eval = None

        for batch_idx, batch in enumerate(tqdm(test_loader, desc=f"{task_name} Evaluation")):
            try:
                input_ids, attention_mask, labels, prompt_text, text_target, meta = adapt_batch(batch, device)
                input_ids, attention_mask, labels = right_trim(input_ids, attention_mask, labels)

                # If we assembled a combined eval model, reuse it across all batches (no per-batch comm)
                if combined_eval is not None:
                    try:
                        ii = input_ids.to(combined_dev)
                        am = attention_mask.to(combined_dev) if attention_mask is not None else None
                        lb = labels.to(combined_dev) if labels is not None else None
                        outputs = combined_eval(input_ids=ii, attention_mask=am, labels=lb)
                        # Update eval-phase memory snapshot
                        try:
                            if tracker is not None:
                                tracker.set_phase('eval')
                                tracker.update_memory()
                        except Exception:
                            pass
                        loss, answer_acc, f1, em = calculate_squad_metrics(outputs, lb if lb is not None else labels, batch, tokenizer, combined_eval, combined_dev)
                        total_loss += loss
                        total_answer_accuracy += answer_acc
                        total_f1 += f1
                        total_em += em
                        num_batches += 1
                        if batch_idx < 3:
                            print(f"\nBatch {batch_idx}: Loss={loss:.4f}, Acc={answer_acc:.6f}, F1={f1:.6f}, EM={em:.6f}")
                        continue
                    except Exception as e:
                        print(f"⚠️ Combined eval forward failed; falling back to legacy path: {e}")

                # Default eval: compute local outputs for metrics
                outputs = model(input_ids, attention_mask, labels)
                # Update eval-phase memory snapshot
                try:
                    if tracker is not None:
                        tracker.set_phase('eval')
                        tracker.update_memory()
                except Exception:
                    pass

                # Debug: Check data structure for first batch
                if batch_idx == 0:
                    print(f"\n  First batch debug:")
                    print(f"   Input shape: {input_ids.shape}")
                    print(f"   Has formatted_text: {'formatted_text' in batch}")
                    print(f"   Has original_example: {'original_example' in batch}")
                    if 'formatted_text' in batch:
                        print(f"   Formatted text count: {len(batch['formatted_text'])}")
                        print(f"   Sample: {batch['formatted_text'][0][:100]}...")

                # If split eval context exists, ask server to compute loss on its half once
                if split_eval:
                    # Use unified precision flag
                    _use_fp16 = (str(getattr(args, 'precision', 'fp32')).lower() == 'fp16')
                    h_cut_live, pkg = _client_forward_to_cut_payload(
                        client_model,
                        input_ids, attention_mask, labels,
                        send_fp16=_use_fp16
                    )

                    trainer.send_data({
                        "type": "forward_cut",
                        "mode": "eval",
                        "data": {"h_cut": pkg["h_cut"], "attention_mask": pkg["attention_mask"], "labels": pkg["labels"], "cut_layer": pkg["cut_layer"]},
                        "meta": {
                            "task_type": getattr(args, "task", None),
                            "max_new_tokens": getattr(args, "max_new_tokens", 20),
                        }
                    })

                    resp = trainer.receive_data()  # server returns eval stats
                    if not (isinstance(resp, dict) and resp.get("type") == "eval_stats"):
                        raise RuntimeError(f"Bad eval resp: {type(resp)}")

                    # server responded; we rely on local metrics for aggregation

                # Calculate task-aware metrics locally (authoritative)
                # Choose generation max tokens dynamically per task
                task = getattr(args, 'task', 'squad')
                gen_max = 5
                if task == 'squad':
                    gen_max = 24
                elif task == 'drop':
                    gen_max = 8
                elif task == 'xsum':
                    gen_max = 20
                elif task in ('boolq','copa','multirc','cb','wic','wsc','rte'):
                    gen_max = 4
                loss, answer_acc, f1, em = calculate_squad_metrics(
                    outputs, labels, batch, tokenizer, model, device,
                    generation_max_new_tokens=gen_max
                )

                total_loss += loss
                total_answer_accuracy += answer_acc
                total_f1 += f1
                total_em += em
                num_batches += 1

                if batch_idx < 3:
                    print(f"\nBatch {batch_idx}: Loss={loss:.4f}, Acc={answer_acc:.6f}, F1={f1:.6f}, EM={em:.6f}")

            except Exception as e:
                print(f"\n⚠️ Error in evaluation batch {batch_idx}: {e}")
                continue
    
    # Calculate averages
    avg_loss = total_loss / num_batches if num_batches > 0 else 0.0
    avg_answer_accuracy = total_answer_accuracy / num_batches if num_batches > 0 else 0.0
    avg_f1 = total_f1 / num_batches if num_batches > 0 else 0.0
    avg_em = total_em / num_batches if num_batches > 0 else 0.0
    
    print(f"\n  {task_name} Evaluation Complete:")
    print(f"   Average Loss: {avg_loss:.4f}")
    print(f"   Answer Token Accuracy: {avg_answer_accuracy:.6f}")
    print(f"   F1 Score: {avg_f1:.6f}")
    print(f"   Exact Match: {avg_em:.6f}")
    
    return avg_loss, avg_answer_accuracy, avg_f1, avg_em
    
def train_split_learning_zoo(client_model, full_model, train_loader, eval_loader,
                             optimizer, scheduler, trainer, device, tokenizer, args):
    """Train using split learning with Zeroth-Order Optimization (client-side ZOO)."""
    print("  Starting split learning training (ZOO)...")
    print(f"   ZOO Training configuration:")
    print(f"   Max steps: {args.max_steps}")
    print(f"   Batch size: {args.train_batch_size}")
    print(f"   Learning rate: {args.zoo_lr}")
    print(f"   Perturbation scale (eps): {args.mu}")
    print(f"   Number of perturbations: {args.num_pert}")
    try:
        print(f"   server SGD warmup: {getattr(args, 'server_sgd_warmup_steps', 0)} steps | interval: {getattr(args, 'server_sgd_every', 1)}")
    except Exception:
        pass
    print(f"   Eval every: {args.eval_steps} steps")
    # Initialize diagnostics
    try:
        _log_dir = os.path.join(os.getcwd(), "ZOO_logs")
        os.makedirs(_log_dir, exist_ok=True)
        logger_client = ZOOLogger(_log_dir, "client")
    except Exception:
        logger_client = None

    # --- FIX 1: define params to perturb depending on tuning mode ---
    # In prefix mode, perturb client KV prefixes; in LoRA mode, perturb LoRA adapters.
    try:
        if getattr(args, 'tuning', 'prefix') == 'lora':
            # LoRA client wraps base model; only LoRA params are trainable
            try:
                from lora import iter_lora_parameters
                client_params = list(iter_lora_parameters(getattr(client_model, 'base_model', client_model), layer_range=(0, getattr(args, 'cut_layer', 0)-1)))
                if not client_params:
                    # Fallback: collect any requires_grad params (LoRA A/B)
                    client_params = [p for p in getattr(client_model, 'base_model', client_model).parameters() if p.requires_grad]
            except Exception:
                client_params = [p for p in getattr(client_model, 'base_model', client_model).parameters() if p.requires_grad]
        elif getattr(args, 'tuning', 'prefix') == 'full':
            # Full finetuning: train embeddings + client layers [0..cut-1]
            base = getattr(client_model, 'base_model', client_model)
            client_params = []
            for n, p in base.named_parameters():
                take = False
                if n.startswith('model.decoder.embed_'):
                    take = True
                elif '.model.decoder.layers.' in ('.' + n):
                    try:
                        after = n.split('model.decoder.layers.')[1]
                        idx = int(after.split('.')[0])
                        if idx < int(getattr(args, 'cut_layer', 0)):
                            take = True
                    except Exception:
                        pass
                if take and p.requires_grad:
                    client_params.append(p)
            if not client_params:
                client_params = [p for p in base.parameters() if p.requires_grad]
        else:
            # Prefix mode
            client_params = list(client_model.kv.parameters())
    except Exception:
        # Ultimate fallback: any trainable parameter
        client_params = [p for p in client_model.parameters() if p.requires_grad]

    # Create gradient estimator on the EXACT params optimized by 'optimizer'
    # This guarantees grads land on the same Parameter objects that optimizer.step() updates.
    try:
        _opt_params = []
        for _group in getattr(optimizer, 'param_groups', []):
            for _p in _group.get('params', []):
                if isinstance(_p, torch.nn.Parameter) and _p.requires_grad:
                    _opt_params.append(_p)
        if not _opt_params:
            _opt_params = client_params
    except Exception:
        _opt_params = client_params
    grad_estimator = StochasticGradientApproximator(
        model_params=_opt_params,
        perturbation_scale=args.mu,
        sample_count=args.num_pert,
        compute_device=device,
        data_type=(torch.float16 if str(getattr(args, 'precision', 'fp32')).lower() == 'fp16' else torch.float32),
        estimator_type=str(getattr(args, 'estimator', 'central')),
        perturbation_type=str(getattr(args, 'zoo_perturbation_type', 'gaussian')),
        max_grad_value=float(getattr(args, 'zoo_max_grad_value', 500.0))
    )

    client_model.train()
    # Ensure tracker is in train phase and reset phase peak at start
    try:
        if _srv_tracker is not None:
            _srv_tracker.set_phase('train')
    except Exception:
        pass
    # Diagnostic: snapshot initial trainable parameters to measure changes
    _client_initial_params = {}
    try:
        _base_mod = getattr(client_model, 'base_model', client_model)
        for _name, _param in _base_mod.named_parameters():
            if _param.requires_grad:
                _client_initial_params[_name] = _param.detach().clone()
    except Exception:
        _client_initial_params = None
    losses, accs = [], []
    global_step = 0
    stop_training = False


    try:
        is_plateau_sched = (str(getattr(args, 'scheduler', 'plateau')).lower() == 'plateau')
        pbar = tqdm(total=args.max_steps, desc="Training (ZOO)",
                    bar_format='{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}] Loss: {postfix}')

        epoch = 0
        while global_step < args.max_steps:
            epoch += 1
            print(f"\nEpoch {epoch} (Steps {global_step}/{args.max_steps})")

            for batch_idx, batch in enumerate(train_loader):
                if global_step >= args.max_steps:
                    break

                try:
                    input_ids = batch['input_ids'].to(device)
                    attention_mask = batch['attention_mask'].to(device)

                    # Prefer dataset-provided labels (answer-only), fallback to pad-masked copy
                    labels = batch.get('labels', None)
                    if isinstance(labels, torch.Tensor):
                        labels = labels.to(device)
                    else:
                        labels = input_ids.clone()
                        labels[attention_mask == 0] = -100
                        pad_id = getattr(tokenizer, 'pad_token_id', None)
                        if pad_id is not None:
                            labels[input_ids == pad_id] = -100
                        labels = labels.long()

                    optimizer.zero_grad(set_to_none=True)

                    # SOLUTION 1: Generate synchronized seed for coordinated perturbations
                    # This ensures client and server use the same random seed, eliminating cross-terms
                    # that cause gradient explosion in split learning ZOO-ZOO
                    sync_seed = global_step * 1000 + batch_idx + args.seed
                    if global_step in (0, 10):
                        print(f"Client sync_seed @step {global_step}: {sync_seed}")
                        # Debug Phase-2 flags and cadence at key steps
                        try:
                            warm_dbg = int(getattr(args, 'server_sgd_warmup_steps', 0))
                            every_dbg = max(1, int(getattr(args, 'server_sgd_every', 1)))
                            print(f"[phase2-debug] seq={bool(getattr(args,'sequential_zoo', False))}, "
                                  f"srvZOO={bool(getattr(args,'use_zeroth_order_server', False))}, "
                                  f"warm={warm_dbg}, every={every_dbg}")
                        except Exception:
                            pass

                    # EARLY PHASE-2 PROBE: run immediately for first N steps (diagnostic)
                    try:
                        _force_n = max(0, int(getattr(args, 'force_phase2_probe', 0)))
                    except Exception:
                        _force_n = 0
                    if _force_n > 0 and global_step < _force_n and bool(getattr(args, 'sequential_zoo', False)) and bool(getattr(args, 'use_zeroth_order_server', False)):
                        try:
                            _phase2_id = int(global_step)
                            print(f"\n[phase2-enter] gstep={global_step}, phase2_id={_phase2_id} [forced-early]", flush=True)
                            print(f"PHASE 2: Updating SERVER parameters (client FIXED)...", flush=True)
                            _was_training = client_model.training
                            client_model.eval()
                            with torch.no_grad():
                                _use_fp16 = False
                                _h_cut_fix, _pkg_fix = _client_forward_to_cut_payload(
                                    client_model, input_ids, attention_mask, labels, send_fp16=_use_fp16
                                )
                                _pkg_fix["tgt_len"] = int(_h_cut_fix.shape[1])
                            _base_seed_server = int(sync_seed) + int(getattr(args, 'num_pert', 10))
                            trainer.send_data({
                                "type": "zoo_server_update",
                                "mode": "train",
                                "data": _pkg_fix,
                                "meta": {
                                    "zoo_phase": "server_update",
                                    "num_perturbations": int(getattr(args, 'num_pert', 10)),
                                    "base_seed": int(_base_seed_server),
                                    "mu": float(getattr(args, 'mu', 1e-3)),
                                    "step": int(global_step),
                                    "phase2_id": int(_phase2_id)
                                }
                            })
                            _srv_resp = trainer.receive_data()
                            print(f"[phase2-exit] gstep={global_step}, resp_type={_srv_resp.get('type') if isinstance(_srv_resp, dict) else type(_srv_resp)}, phase2_id={(isinstance(_srv_resp, dict) and _srv_resp.get('phase2_id'))}", flush=True)
                            if _was_training:
                                client_model.train()
                        except Exception as _probe_e:
                            print(f"⚠️ Phase2 forced-early probe failed: {_probe_e}", flush=True)

                    def objective_fn():
                        # Run ZOO evaluations with dropout disabled to remove extra stochasticity
                        was_training = client_model.training
                        try:
                            client_model.eval()
                            with torch.no_grad():
                                _use_fp16 = (str(getattr(args, 'precision', 'fp32')).lower() == 'fp16')
                                h_cut_live, pkg = _client_forward_to_cut_payload(
                                    client_model,
                                    input_ids, attention_mask, labels,
                                    send_fp16=_use_fp16
                                )
                            pkg["tgt_len"] = int(h_cut_live.shape[1])
                            trainer.send_data({
                                "type": "forward_cut",
                                "mode": "train",
                                "data": pkg,
                                "meta": {"zoo_eval": True, "need_g_cut": False, "sync_seed": sync_seed, "step": int(global_step)}
                            })
                            resp = trainer.receive_data()
                            try:
                                if _srv_tracker is not None:
                                    _srv_tracker.set_phase('train')
                                    _srv_tracker.update_memory()
                            except Exception:
                                pass
                            loss_val = float(resp.get("loss", 0.0))
                            if loss_val > 20.0 or loss_val < 0.0 or not np.isfinite(loss_val):
                                print(f"WARNING: Suspicious loss value detected: {loss_val:.6f}. This may indicate:")
                                print(f"  - Loss computation error")
                                print(f"  - Model divergence")
                                print(f"  - Label/input mismatch")
                            return torch.tensor(float(loss_val), dtype=next(client_model.base_model.parameters()).dtype)
                        finally:
                            if was_training:
                                client_model.train()

                    def objective_fn_c(*_args,**_kwargs):
                        return objective_fn()

                    # Optional: request g_cut once and use a variance-reduced objective ⟨g_cut, h_cut(θ)⟩
                    use_gcut = bool(getattr(args, 'zoo_use_gcut', False))
                    g_cut_tensor = None
                    pkg_probe = None
                    if use_gcut:
                        try:
                            # CRITICAL FIX: Keep model in training mode (consistent with objective_fn fix)
                            with torch.no_grad():
                                _use_fp16 = (str(getattr(args, 'precision', 'fp32')).lower() == 'fp16')
                                h_probe, pkg_probe = _client_forward_to_cut_payload(
                                    client_model, input_ids, attention_mask, labels, send_fp16=_use_fp16
                                )
                                pkg_probe["tgt_len"] = int(h_probe.shape[1])
                            trainer.send_data({
                                "type": "forward_cut",
                                "mode": "train",
                                "data": pkg_probe,
                                "meta": {"zoo_eval": False, "need_g_cut": True, "local_sgd": False, "step": int(global_step)}
                            })
                            resp_gc = trainer.receive_data()
                            if isinstance(resp_gc, dict) and ("g_cut" in resp_gc):
                                g_cut_tensor = torch.as_tensor(resp_gc["g_cut"], dtype=torch.float32, device=h_probe.device)
                        except Exception as _e:
                            print(f"⚠️ zoo_use_gcut failed, falling back to loss objective: {_e}")
                            g_cut_tensor = None
                            pkg_probe = None

                    def objective_fn_gcut(*_args, **_kwargs):
                        # f(θ) = ⟨g_cut, h_cut(θ)⟩ with fixed g_cut from current server
                        if g_cut_tensor is None:
                            return objective_fn()
                        was_training = client_model.training
                        try:
                            client_model.eval()
                            with torch.no_grad():
                                _use_fp16 = (str(getattr(args, 'precision', 'fp32')).lower() == 'fp16')
                                h_tmp, _ = _client_forward_to_cut_payload(
                                    client_model, input_ids, attention_mask, labels, send_fp16=_use_fp16
                                )
                            v = (h_tmp.to(dtype=g_cut_tensor.dtype) * g_cut_tensor).sum().item()
                            return torch.tensor(float(v), dtype=torch.float32)
                        finally:
                            if was_training:
                                client_model.train()

                    # Estimate gradients for client KV via ZOO with configurable μ scaling
                    pbar.set_description("Training (ZOO) | Computing ZOO gradients...")
                    # Always estimate grads on the EXACT Parameter objects the optimizer will step
                    try:
                        _opt_params_step = []
                        for _grp in getattr(optimizer, 'param_groups', []):
                            for _pp in _grp.get('params', []):
                                if isinstance(_pp, torch.nn.Parameter) and _pp.requires_grad:
                                    _opt_params_step.append(_pp)
                        if _opt_params_step:
                            grad_estimator.model_params = _opt_params_step
                        else:
                            grad_estimator.model_params = client_params
                    except Exception:
                        grad_estimator.model_params = client_params

                    # Choose mu scaling method
                    # CRITICAL FIX: Use 'fixed' by default (like server) - RMS scaling makes mu WAY too large
                    # RMS scaling multiplies mu by parameter RMS (often 0.1-1.0), making perturbations 100-1000x larger!
                    base_mu = float(getattr(args, 'mu', 1e-3))
                    mu_scaling = str(getattr(args, 'zoo_mu_scaling', 'fixed')).lower()

                    if mu_scaling == 'fixed':
                        # Fixed mu (like MeZO) - use base_mu directly
                        scaled_mu = base_mu
                    else:  # mu_scaling == 'rms'
                        # RMS scale μ per step (current implementation)
                        with torch.no_grad():
                            squares, count = 0.0, 0
                            for p in client_params:
                                squares += (p.data.float()**2).sum().item()
                                count   += int(p.numel())
                            rms = (squares / max(1, count))**0.5
                        scaled_mu = max(1e-5, base_mu) * (rms if rms > 0 else 1.0)

                    old_mu = getattr(grad_estimator, 'perturbation_scale', scaled_mu)
                    grad_estimator.perturbation_scale = scaled_mu
                    try:
                        # Parameter restoration safety check (before ZOO)
                        try:
                            _ = verify_parameter_restoration(getattr(client_model, 'base_model', client_model), "before_ZOO")
                        except Exception:
                            pass
                        # SOLUTION 1: Use synchronized seed for client perturbations
                        # This ensures client and server perturbations are correlated, eliminating cross-terms
                        grad_estimator.estimate_gradients(
                            input_ids, labels,
                            (objective_fn_gcut if use_gcut else objective_fn_c),
                            random_seed=sync_seed
                        )
                    finally:
                        grad_estimator.perturbation_scale = old_mu
                    # Parameter restoration safety check (after ZOO, before step)
                    try:
                        _ = verify_parameter_restoration(getattr(client_model, 'base_model', client_model), "after_ZOO")
                    except Exception:
                        pass

                    # Track memory after gradient estimation
                    try:
                        if _srv_tracker is not None:
                            _srv_tracker.set_phase('train')
                            _srv_tracker.update_memory()
                    except Exception:
                        pass

                    # Step every iteration (no ZOO-specific accumulation)
                    # Apply clipping if explicitly requested (>0)
                        try:
                            clip_value = getattr(args, 'clip_grad_norm', 0.0)
                            if clip_value > 0.0:
                                params_to_clip = []
                                for group in optimizer.param_groups:
                                    for p in group.get('params', []):
                                        if p.grad is not None:
                                            params_to_clip.append(p)
                                if params_to_clip:
                                    torch.nn.utils.clip_grad_norm_(params_to_clip, max_norm=clip_value)
                        except Exception:
                            pass
                        # Debug: ensure grads exist on optimizer params
                        try:
                            _num_opt_grads = sum(1 for _g in optimizer.param_groups for _p in _g.get('params', []) if getattr(_p, 'grad', None) is not None)
                            if (global_step < 3) or (global_step % 50 == 0):
                                print(f"[zoo-debug] optimizer params with grads: {_num_opt_grads}")
                        except Exception:
                            pass
                        optimizer.step()
                        # Per-step LoRA debug: report grad norm and max |Δ| for key adapters
                        try:
                            if str(getattr(args, 'tuning', 'prefix')) == 'lora':
                                _base_mod_dbg = getattr(client_model, 'base_model', client_model)
                                if not hasattr(train_split_learning_zoo, '_lora_prev'):
                                    train_split_learning_zoo._lora_prev = {}
                                _prev_map = train_split_learning_zoo._lora_prev
                                _targets = (
                                    'model.decoder.layers.0.self_attn.q_proj.lora_A',
                                    'model.decoder.layers.0.self_attn.q_proj.lora_B',
                                    'model.decoder.layers.0.self_attn.v_proj.lora_A',
                                    'model.decoder.layers.0.self_attn.v_proj.lora_B',
                                )
                                for _name, _param in _base_mod_dbg.named_parameters():
                                    if _param.requires_grad and _name in _targets:
                                        try:
                                            _g = _param.grad
                                            _gn = float(_g.norm().item()) if (_g is not None) else 0.0
                                        except Exception:
                                            _gn = 0.0
                                        _last = _prev_map.get(_name, _param.detach().clone())
                                        _dmax = float((_param.detach() - _last).abs().max().item())
                                        print(f"[lora-delta] {_name}: grad_norm={_gn:.3e} max|Δ|={_dmax:.3e}")
                                        _prev_map[_name] = _param.detach().clone()
                        except Exception:
                            pass
                        # Parameter restoration vs step check (after optimizer step)
                        try:
                            _ = verify_parameter_restoration(getattr(client_model, 'base_model', client_model), "after_step")
                        except Exception:
                            pass
                        # Track memory after optimizer step
                        try:
                            if _srv_tracker is not None:
                                _srv_tracker.set_phase('train')
                                _srv_tracker.update_memory()
                        except Exception:
                            pass
                        stepped_this_iter = True
                        if not is_plateau_sched:
                            try:
                                scheduler.step()
                                # Diagnostic: Log LR every 25 steps to verify scheduler is working
                                if global_step % 25 == 0:
                                    current_lr = _get_current_lr(optimizer)
                                    print(f"[LR] Step {global_step}: LR = {current_lr:.6g}")
                            except Exception:
                                pass

                        # ================================================================
                        # Phase 2: Trigger SERVER ZOO update with fixed client (sequential)
                        # ================================================================
                        try:
                            if bool(getattr(args, 'use_zeroth_order_server', False)) and bool(getattr(args, 'sequential_zoo', False)):
                                # Gate Phase 2 by warmup/cadence OR force-probe for first N steps
                                warm = int(getattr(args, 'server_sgd_warmup_steps', 0))
                                every = max(1, int(getattr(args, 'server_sgd_every', 1)))
                                force_n = max(0, int(getattr(args, 'force_phase2_probe', 0)))
                                try:
                                    print(f"[phase2-check] gstep={global_step} seq={bool(getattr(args,'sequential_zoo', False))} srvZOO={bool(getattr(args,'use_zeroth_order_server', False))} warm={warm} every={every} force={force_n}", flush=True)
                                except Exception:
                                    pass
                                should_run = ((global_step >= warm) and (global_step % every == 0)) or (global_step < force_n)
                                try:
                                    print(f"[phase2-should-run] gstep={global_step} -> {bool(should_run)}", flush=True)
                                except Exception:
                                    pass
                                if should_run:
                                    phase2_id = int(global_step)
                                    if global_step < force_n:
                                        print(f"\n[phase2-enter] gstep={global_step}, phase2_id={phase2_id} [forced]", flush=True)
                                    else:
                                        print(f"\n[phase2-enter] gstep={global_step}, phase2_id={phase2_id}", flush=True)
                                    print(f"PHASE 2: Updating SERVER parameters (client FIXED)...", flush=True)
                                    was_training = client_model.training
                                    client_model.eval()
                                    with torch.no_grad():
                                        _use_fp16 = False  # send fixed activations in full precision for stability
                                        h_cut_fixed, pkg_fixed = _client_forward_to_cut_payload(
                                            client_model, input_ids, attention_mask, labels, send_fp16=_use_fp16
                                        )
                                        pkg_fixed["tgt_len"] = int(h_cut_fixed.shape[1])
                                    # distinct base seed range for server phase
                                    base_seed_server = int(sync_seed) + int(getattr(args, 'num_pert', 10))
                                    trainer.send_data({
                                        "type": "zoo_server_update",
                                        "mode": "train",
                                        "data": pkg_fixed,
                                        "meta": {
                                            "zoo_phase": "server_update",
                                            "num_perturbations": int(getattr(args, 'num_pert', 10)),
                                            "base_seed": int(base_seed_server),
                                            "mu": float(getattr(args, 'mu', 1e-3)),
                                            "step": int(global_step),
                                            "phase2_id": int(phase2_id)
                                        }
                                    })
                                    server_response = trainer.receive_data()
                                    if not isinstance(server_response, dict):
                                        raise RuntimeError(f"Expected dict reply from server, got: {type(server_response)}")
                                    print(f"[phase2-exit] gstep={global_step}, resp_type={server_response.get('type')}, phase2_id={server_response.get('phase2_id')}")
                                    if server_response.get("type") != "zoo_phase_complete":
                                        raise RuntimeError(f"Expected 'zoo_phase_complete', got: {server_response}")
                                    if int(server_response.get("phase2_id", -1)) != int(phase2_id):
                                        raise RuntimeError(f"Phase2 ACK mismatch (client={phase2_id}, server={server_response.get('phase2_id')})")
                                    server_grad_norm = float(server_response.get("grad_norm", 0.0))
                                    server_loss_diff_avg = float(server_response.get("loss_diff_avg", 0.0))
                                    server_loss_diff_std = float(server_response.get("loss_diff_std", 0.0))
                                    print(f"\nSERVER ZOO Results:")
                                    print(f"  Gradient norm: {server_grad_norm:.6f}")
                                    print(f"  Avg loss diff: {server_loss_diff_avg:.6f}")
                                    print(f"  Loss diff std: {server_loss_diff_std:.6f}")
                                    if was_training:
                                        client_model.train()
                                else:
                                    # Occasional debug to verify cadence gating behavior
                                    if (global_step < 5) or (global_step % 25 == 0):
                                        print(f"Skipping server ZOO (gstep={global_step}, warm={warm}, every={every})")
                        except Exception as _se:
                            print(f"⚠️ Sequential server ZOO update failed or skipped: {_se}")

                    # Optional: trigger a small server-side SGD update with cadence
                    try:
                        if not bool(getattr(args, 'use_zeroth_order_server', False)):
                            warm = int(getattr(args, 'server_sgd_warmup_steps', 0))
                            every = max(1, int(getattr(args, 'server_sgd_every', 1)))
                            if (global_step >= warm) and (global_step % every == 0):
                                # Build fresh pkg and ask server to do a local step
                                client_model.eval()
                                with torch.no_grad():
                                    _use_fp16 = (False if (bool(getattr(args,'use_zeroth_order', False)) or bool(getattr(args,'use_zeroth_order_server', False))) else True)
                                    h2, pkg2 = _client_forward_to_cut_payload(
                                        client_model, input_ids, attention_mask, labels, send_fp16=_use_fp16
                                    )
                                    pkg2["tgt_len"] = int(h2.shape[1])
                                trainer.send_data({
                                    "type": "forward_cut",
                                    "mode": "train",
                                    "data": pkg2,
                                    "meta": {"zoo_eval": False, "need_g_cut": False, "local_sgd": True}
                                })
                                _ = trainer.receive_data()
                    except Exception:
                        pass

                    # --- monitor: use ZOO objective loss (authoritative for client ZOO) ---
                    try:
                        cur_obj_loss = float(objective_fn().item())
                    except Exception:
                        cur_obj_loss = 0.0
                    losses.append(cur_obj_loss)
                    # Log diagnostics
                    try:
                        if logger_client is not None:
                            logger_client.log_step(global_step, cur_obj_loss, getattr(client_model, 'base_model', client_model), optimizer)
                            if (global_step % 100) == 0 and global_step > 0:
                                logger_client.save()
                                # plotting is optional and may be heavy
                    except Exception:
                        pass
                    
                    
                    # FIX: Compute actual training accuracy instead of hardcoded 0.0
                    # Use split learning communication path to get logits from server
                    try:
                        was_training = client_model.training
                        client_model.eval()
                        with torch.no_grad():
                            # Use the same split learning forward pass mechanism
                            _use_fp16 = (str(getattr(args, 'precision', 'fp32')).lower() == 'fp16')
                            h_cut_acc, pkg_acc = _client_forward_to_cut_payload(
                                client_model,
                                input_ids, attention_mask, labels,
                                send_fp16=_use_fp16
                            )
                            pkg_acc["tgt_len"] = int(h_cut_acc.shape[1])
                            # Request server to compute logits (for accuracy computation)
                            trainer.send_data({
                                "type": "forward_cut",
                                "mode": "train",
                                "data": pkg_acc,
                                    "meta": {"zoo_eval": True, "need_g_cut": False, "return_logits": True, "step": int(global_step)}
                            })
                            resp_acc = trainer.receive_data()
                            
                            # Server should return logits if available
                            if isinstance(resp_acc, dict) and "logits" in resp_acc:
                                logits_tensor = torch.as_tensor(resp_acc["logits"], dtype=torch.float32, device=device)
                                # Ensure labels match logits sequence length
                                # Logits shape: [B, seq_len, vocab_size]
                                # Labels shape: [B, original_seq_len]
                                logits_seq_len = logits_tensor.shape[1]
                                labels_seq_len = labels.shape[1]
                                
                                # Align labels to match logits sequence length
                                if labels_seq_len > logits_seq_len:
                                    # Truncate labels to match logits
                                    labels_aligned = labels[:, :logits_seq_len]
                                elif labels_seq_len < logits_seq_len:
                                    # Pad labels to match logits (this shouldn't happen, but handle it)
                                    padding = torch.full((labels.shape[0], logits_seq_len - labels_seq_len), 
                                                        -100, device=device, dtype=labels.dtype)
                                    labels_aligned = torch.cat([labels, padding], dim=1)
                                else:
                                    labels_aligned = labels
                                
                                # Shift for language modeling
                                shift_logits = logits_tensor[..., :-1, :].contiguous()
                                shift_labels = labels_aligned[..., 1:].contiguous()
                                
                                # Compute accuracy
                                preds = torch.argmax(shift_logits, dim=-1)
                                non_padding = (shift_labels != -100)
                                if non_padding.sum() > 0:
                                    correct = (preds == shift_labels) & non_padding
                                    batch_acc = correct.sum().float() / non_padding.sum().float()
                                    accs.append(batch_acc.item())
                                    # Debug logging (first step only)
                                    if global_step == 1:
                                        print(f"✅ Training accuracy computation working: batch_acc={batch_acc.item():.4f}, logits_shape={logits_tensor.shape}, labels_shape={labels.shape}, aligned_labels_shape={labels_aligned.shape}")
                                else:
                                    accs.append(0.0)
                            else:
                                # Fallback: if server doesn't return logits, use a simple estimate
                                # For SST-2, we can check if loss is decreasing as a proxy
                                # But for now, just append 0.0 and log a warning
                                if global_step % 100 == 0:  # Only log occasionally
                                    print(f"⚠️ Server didn't return logits for accuracy computation at step {global_step}")
                                    print(f"   Response keys: {list(resp_acc.keys()) if isinstance(resp_acc, dict) else 'not a dict'}")
                                accs.append(0.0)
                        
                        # Restore training mode
                        if was_training:
                            client_model.train()
                    except Exception as acc_error:
                        # Only print error occasionally to avoid spam
                        if global_step % 100 == 0:
                            print(f"⚠️ Training accuracy computation failed at step {global_step}: {acc_error}")
                        accs.append(0.0)
                        # Ensure training mode is restored
                        client_model.train()

                    global_step += 1
                    if global_step == 10 and _client_initial_params:
                        try:
                            _base_mod = getattr(client_model, 'base_model', client_model)
                            print("Parameter max changes after 10 steps (client):")
                            for _name, _param in _base_mod.named_parameters():
                                if _param.requires_grad and _name in _client_initial_params:
                                    _delta = (_param - _client_initial_params[_name]).abs().max().item()
                                    print(f"{_name}: max param change = {_delta:.6e}")
                        except Exception:
                            pass
                    cur_loss = sum(losses[-10:]) / min(len(losses), 10)
                    cur_acc  = sum(accs[-10:]) / min(len(accs), 10)
                    pbar.set_description(f"Training (ZOO) | Loss: {cur_loss:.4f} | Acc: {cur_acc:.6f}")
                    pbar.update(1)

                    # Periodically step LR scheduler on a smoothed ZOO objective (Plateau only)
                    if is_plateau_sched and global_step > 0 and (global_step % 200 == 0):
                        try:
                            smooth_loss = sum(losses[-50:]) / min(len(losses), 50)
                            scheduler.step(float(smooth_loss))
                            print(f"[LR] Plateau step on train MA (ZOO) @ step {global_step}: metric={smooth_loss:.4f}, lr={_get_current_lr(optimizer):.6g}")
                        except Exception:
                            pass

                    # Optional diagnostic: effect of -alpha * grad step on objective
                    if (global_step % 100 == 0):
                        try:
                            base_loss = float(objective_fn().item())
                            alpha = min(1e-2, float(getattr(args, 'zoo_lr', 1e-4)))
                            with torch.no_grad():
                                for p in client_params:
                                    if p.grad is not None:
                                        p.add_(p.grad, alpha=-alpha)
                            new_loss = float(objective_fn().item())
                            with torch.no_grad():
                                for p in client_params:
                                    if p.grad is not None:
                                        p.add_(p.grad, alpha=+alpha)
                            print(f"[diag] Δloss from -α·g step: {base_loss - new_loss:.6f}")
                        except Exception:
                            pass

                    # periodic eval (uses the safe FullLLMModel)
                    if global_step % args.eval_steps == 0:
                        # Switch tracker to eval phase
                        try:
                            if _srv_tracker is not None:
                                _srv_tracker.set_phase('eval')
                        except Exception:
                            pass
                        print(f"\nStep {global_step}: Running ZOO evaluation...")
                        _moved = False
                        _eval_device = device
                        try:
                            if bool(getattr(args, 'eval_on_cpu', False)) and str(getattr(args, 'tuning', 'prefix')) == 'lora':
                                full_model.to(torch.device('cpu'))
                                _eval_device = torch.device('cpu')
                                _moved = True
                            eval_loss, eval_acc, eval_f1, eval_em = evaluate_model(
                                full_model, eval_loader, _eval_device, tokenizer, args,
                                client_model=client_model, trainer=trainer, tracker=_srv_tracker
                            )
                        finally:
                            if _moved:
                                full_model.to(device)
                        # Switch back to train phase
                        try:
                            if _srv_tracker is not None:
                                _srv_tracker.set_phase('train')
                        except Exception:
                            pass
                        print(f".  Step {global_step} ZOO Evaluation:")
                        print(f"   Loss: {eval_loss:.4f}")
                        print(f"   Answer Accuracy: {eval_acc:.6f}")
                        print(f"   F1 Score: {eval_f1:.6f}")
                        print(f"   Exact Match: {eval_em:.6f}")
                        # Step LR scheduler on eval metric when available (Plateau only)
                        if is_plateau_sched:
                            try:
                                if bool(getattr(args, 'sched_step_on_eval', False)):
                                    scheduler.step(float(eval_loss))
                                    print(f"[LR] Plateau step on eval (ZOO) @ step {global_step}: metric={eval_loss:.4f}, lr={_get_current_lr(optimizer):.6g}")
                            except Exception:
                                pass
                        client_model.train()

                except StopIteration:
                    stop_training = True
                    break
                except Exception as e:
                    print(f"\n✖ Error in ZOO step {global_step}: {e}")
                    traceback.print_exc()
                    continue

            if stop_training:
                break

        pbar.close()
        # Final scheduler step behavior
        if is_plateau_sched:
            try:
                scheduler.step(eval_loss)
                print(f"[LR] Final plateau step (ZOO) on eval: metric={eval_loss:.4f}, lr={_get_current_lr(optimizer):.6g}")
            except Exception:
                try:
                    last_loss = sum(losses[-100:]) / min(len(losses), 100) if losses else 0.0
                    scheduler.step(last_loss)
                    print(f"[LR] Final plateau step (ZOO) on train MA: metric={last_loss:.4f}, lr={_get_current_lr(optimizer):.6g}")
                except Exception:
                    pass

        avg_loss = sum(losses) / len(losses) if losses else 0.0
        avg_acc  = sum(accs) / len(accs) if accs else 0.0
        print(f"\n✅ ZOO Training Complete - Final Train Loss: {avg_loss:.4f}, Final Train Acc: {avg_acc:.6f}")
        # Final evaluation and memory update
        try:
            # Mark eval phase and update memory one last time
            if _srv_tracker is not None:
                _srv_tracker.set_phase('eval')
                _srv_tracker.update_memory()
        except Exception:
            pass
        try:
            _moved = False
            _eval_device = device
            try:
                if bool(getattr(args, 'eval_on_cpu', False)) and str(getattr(args, 'tuning', 'prefix')) == 'lora':
                    full_model.to(torch.device('cpu'))
                    _eval_device = torch.device('cpu')
                    _moved = True
                final_eval_loss, final_eval_acc, final_eval_f1, final_eval_em = evaluate_model(
                    full_model, eval_loader, _eval_device, tokenizer, args,
                    client_model=client_model, trainer=trainer, tracker=_srv_tracker
                )
            finally:
                if _moved:
                    full_model.to(device)
            print(f"🔍 Final Evaluation - Loss: {final_eval_loss:.4f}, Acc: {final_eval_acc:.6f}, F1: {final_eval_f1:.6f}, EM: {final_eval_em:.6f}")
        except Exception:
            pass
        return avg_loss, avg_acc

    except Exception as e:
        print(f"✖ ZOO training failed: {e}")
        traceback.print_exc()
        return 0.0, 0.0


def train_split_learning_sgd(client_model, full_model, train_loader, eval_loader, optimizer, 
                           scheduler, trainer, device, tokenizer, args):
    """Train using split learning with SGD - FIXED VERSION"""
    print("   Starting split learning training (SGD)...")
    print(f"   Training configuration:")
    print(f"   Max steps: {args.max_steps}")
    print(f"   Batch size: {args.train_batch_size}")
    print(f"   Learning rate: {args.lr}")
    print(f"   Eval every: {args.eval_steps} steps")
    
    client_model.train()
    # Ensure tracker is in train phase and reset phase peak at start
    try:
        if _srv_tracker is not None:
            _srv_tracker.set_phase('train')
    except Exception:
        pass
    # Diagnostic: snapshot initial trainable parameters to measure changes (SGD)
    _client_initial_params_sgd = {}
    try:
        _base_mod_sgd = getattr(client_model, 'base_model', client_model)
        for _name, _param in _base_mod_sgd.named_parameters():
            if _param.requires_grad:
                _client_initial_params_sgd[_name] = _param.detach().clone()
    except Exception:
        _client_initial_params_sgd = None
    losses = []
    accs = []
    global_step = 0
    
    try:
        is_plateau_sched = (str(getattr(args, 'scheduler', 'plateau')).lower() == 'plateau')
        accum_steps = max(1, int(getattr(args, 'sgd_accum_steps', 1)))
        pbar = tqdm(total=args.max_steps, desc="Training (SGD)", 
                   bar_format='{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}] Loss: {postfix}')
        
        epoch = 0
        while global_step < args.max_steps:
            epoch += 1
            print(f"\nEpoch {epoch} (Steps {global_step}/{args.max_steps})")

            for i, batch in enumerate(train_loader):
                if global_step >= args.max_steps:
                    break

                try:
                    input_ids = batch['input_ids'].to(device)
                    attention_mask = batch['attention_mask'].to(device)
                    # labels = batch['labels'].to(device)
                    
                    # Note: zeroing is handled before accumulate or on step boundaries
                    # Prefer dataset-provided labels (answer-only), fallback to pad-masked copy
                    labels = batch.get('labels', None)
                    if isinstance(labels, torch.Tensor):
                        labels = labels.to(device)
                    else:
                        labels = input_ids.clone()
                        labels[attention_mask == 0] = -100
                        pad_id = getattr(tokenizer, 'pad_token_id', None)
                        if pad_id is not None:
                            labels[input_ids == pad_id] = -100
                        labels = labels.long()

                    # === True-split: compute and send h_cut ===
                    # Use unified precision flag
                    _use_fp16 = (str(getattr(args, 'precision', 'fp32')).lower() == 'fp16')

                    h_cut_live, pkg = _client_forward_to_cut_payload(
                        client_model,                 # clientKVOnly instance
                        input_ids, attention_mask, labels,
                        send_fp16=_use_fp16
                    )
                    pkg["tgt_len"] = int(h_cut_live.shape[1])

                    # Track train-phase memory after client forward to cut
                    try:
                        if _srv_tracker is not None:
                            _srv_tracker.set_phase('train')
                            _srv_tracker.update_memory()
                    except Exception:
                        pass

                    trainer.send_data({"type": "forward_cut", "mode": "train", "data": pkg, "meta": {"zoo_eval": False}})

                    # === Receive loss + g_cut; backprop on client ===
                    resp = trainer.receive_data()  # {'loss': float, 'g_cut': tensor}
                    loss_val = float(resp["loss"])
                    print(f"Loss: {loss_val:.4f}")

                    g_cut = torch.as_tensor(resp["g_cut"])
                    if g_cut.shape != h_cut_live.shape:
                        raise RuntimeError(f"g_cut shape {tuple(g_cut.shape)} != h_cut {tuple(h_cut_live.shape)}")
                    g_cut = g_cut.to(device=h_cut_live.device, dtype=h_cut_live.dtype)

                    # Accumulation-aware backward and step
                    if (global_step % accum_steps) == 0:
                        optimizer.zero_grad(set_to_none=True)
                    h_cut_live.backward(g_cut)
                    # Track memory after backward
                    try:
                        if _srv_tracker is not None:
                            _srv_tracker.set_phase('train')
                            _srv_tracker.update_memory()
                    except Exception:
                        pass
                    do_step = ((global_step + 1) % accum_steps) == 0
                    if do_step:
                        try:
                            if getattr(args, 'clip_grad_norm', 0.0) > 0.0:
                                params_to_clip = []
                                for group in optimizer.param_groups:
                                    for p in group.get('params', []):
                                        if p.grad is not None:
                                            params_to_clip.append(p)
                                if params_to_clip:
                                    torch.nn.utils.clip_grad_norm_(params_to_clip, max_norm=args.clip_grad_norm)
                        except Exception:
                            pass
                        optimizer.step()
                        # Track memory after optimizer step
                        try:
                            if _srv_tracker is not None:
                                _srv_tracker.set_phase('train')
                                _srv_tracker.update_memory()
                        except Exception:
                            pass
                        if not is_plateau_sched:
                            try:
                                scheduler.step()
                            except Exception:
                                pass

                    # Calculate accuracy for monitoring using full model
                    with torch.no_grad():
                        outputs = full_model(input_ids, attention_mask, labels)
                        loss_check, accuracy, f1, _ = calculate_metrics(outputs, labels, batch, tokenizer, full_model, device)
                        losses.append(loss_val)
                        accs.append(accuracy)
                    
                    # Update progress bar with current loss, accuracy, and F1
                    global_step += 1
                    if global_step == 10 and _client_initial_params_sgd:
                        try:
                            _base_mod_sgd = getattr(client_model, 'base_model', client_model)
                            print("Parameter max changes after 10 steps (client, SGD):")
                            for _name, _param in _base_mod_sgd.named_parameters():
                                if _param.requires_grad and _name in _client_initial_params_sgd:
                                    _delta = (_param - _client_initial_params_sgd[_name]).abs().max().item()
                                    print(f"{_name}: max param change = {_delta:.6e}")
                        except Exception:
                            pass
                    current_loss = sum(losses[-10:]) / min(len(losses), 10)  # Moving average of last 10
                    current_acc = sum(accs[-10:]) / min(len(accs), 10)
                    
                    pbar.set_description(f"Training (SGD) | Loss: {current_loss:.4f} | Acc: {current_acc:.3f}")
                    pbar.update(1)
                    
                    # Periodically step LR scheduler on a smoothed training loss (Plateau only)
                    if is_plateau_sched and global_step > 0 and (global_step % 200 == 0):
                        try:
                            smooth_loss = sum(losses[-50:]) / min(len(losses), 50)
                            scheduler.step(float(smooth_loss))
                            print(f"[LR] Plateau step on train MA (SGD) @ step {global_step}: metric={smooth_loss:.4f}, lr={_get_current_lr(optimizer):.6g}")
                        except Exception:
                            pass

                    # Print detailed stats every 20 batches
                    if (i + 1) % 20 == 0:
                        avg_loss = sum(losses) / len(losses)
                        avg_acc = sum(accs) / len(accs)
                        print(f"\nStep {i+1}/{len(train_loader)} - Loss: {avg_loss:.4f}, Acc: {avg_acc:.3f}")

                    if global_step % args.eval_steps == 0:
                        # Switch tracker to eval phase
                        try:
                            if _srv_tracker is not None:
                                _srv_tracker.set_phase('eval')
                        except Exception:
                            pass
                        print(f"\nStep {global_step}: Running evaluation...")
                        _moved = False
                        _eval_device = device
                        try:
                            if bool(getattr(args, 'eval_on_cpu', False)) and str(getattr(args, 'tuning', 'prefix')) == 'lora':
                                full_model.to(torch.device('cpu'))
                                _eval_device = torch.device('cpu')
                                _moved = True
                            eval_loss, eval_acc, eval_f1, eval_em = evaluate_model(
                                full_model, eval_loader, _eval_device, tokenizer, args,
                                client_model=client_model, trainer=trainer, tracker=_srv_tracker
                            )
                        finally:
                            if _moved:
                                full_model.to(device)

                        # Switch back to train phase
                        try:
                            if _srv_tracker is not None:
                                _srv_tracker.set_phase('train')
                        except Exception:
                            pass
                        print(f"   Step {global_step} Evaluation:")
                        print(f"   Loss: {eval_loss:.4f}")
                        print(f"   Answer Accuracy: {eval_acc:.6f}")
                        print(f"   F1 Score: {eval_f1:.6f}")
                        print(f"   Exact Match: {eval_em:.6f}")
                        
                        # Return to training mode
                        client_model.train()

                        # Step LR scheduler on evaluation loss immediately (Plateau only)
                        if is_plateau_sched:
                            try:
                                scheduler.step(float(eval_loss))
                                print(f"[LR] Plateau step on eval (SGD) @ step {global_step}: metric={eval_loss:.4f}, lr={_get_current_lr(optimizer):.6g}")
                            except Exception:
                                pass

                    # Print progress every 100 steps
                    if global_step % 100 == 0:
                        avg_loss = sum(losses[-100:]) / min(len(losses), 100)
                        avg_acc = sum(accs[-100:]) / min(len(accs), 100)
                        print(f"\nStep {global_step}/{args.max_steps} - Loss: {avg_loss:.4f}, Acc: {avg_acc:.6f}")
                    
                except Exception as e:
                    traceback.print_exc()
                    print(f"Batch {global_step} Error: {e}")
                    continue
        
        if is_plateau_sched:
            try:
                scheduler.step(eval_loss)
                print(f"[LR] Final plateau step (SGD) on eval: metric={eval_loss:.4f}, lr={_get_current_lr(optimizer):.6g}")
            except Exception:
                try:
                    last_loss = sum(losses[-100:]) / min(len(losses), 100) if losses else 0.0
                    scheduler.step(last_loss)
                    print(f"[LR] Final plateau step (SGD) on train MA: metric={last_loss:.4f}, lr={_get_current_lr(optimizer):.6g}")
                except Exception:
                    pass
        
        avg_loss = sum(losses) / len(losses) if losses else 0.0
        avg_acc = sum(accs) / len(accs) if accs else 0.0
        
        print(f"\nSGD Training Complete - Final Loss: {avg_loss:.4f}, Final Acc: {avg_acc:.3f}")
        return avg_loss, avg_acc
        
    except Exception as e:
        print(f"âŒ Training failed: {e}")
        return 0.0, 0.0


def parse_args():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(description='Enhanced Split Learning LLM client')
    
    # Model parameters
    parser.add_argument('--model_name', type=str, default='facebook/opt-125m', help='Model name')
    parser.add_argument('--num_prefix', type=int, default=20, help='Number of prefix tokens')
    parser.add_argument('--cut_layer', type=int, default=1, help='Split index: 0..L-1 goes to client; cut..L-1 to server')
    parser.add_argument('--max_length', type=int, default=512, help='Max sequence length')
    
    # Training parameters
    parser.add_argument('--seed', type=int, default=42, help='Random seed')
    parser.add_argument('--zoo_lr', type=float, default=1e-5, help='ZOO learning rate')
    parser.add_argument('--lr', type=float, default=1e-3, help='Learning rate') 
    parser.add_argument('--momentum', type=float, default=0.9, help='SGD momentum')
    parser.add_argument('--weight_decay', type=float, default=0.001, help='Weight decay for optimizer')
    parser.add_argument('--epochs', type=int, default=3, help='Number of epochs')
    parser.add_argument('--train_batch_size', type=int, default=16, help='Training batch size')
    parser.add_argument('--test_batch_size', type=int, default=16, help='Test batch size')
    parser.add_argument('--clip_grad_norm', type=float, default=0.0, help='Max gradient norm for clipping (0 disables clipping).')

    # Dataset sizes - NEW ARGUMENTS
    parser.add_argument('--train_examples', type=int, default=None, help='Number of training examples (None => full dataset)')
    parser.add_argument('--dev_examples', type=int, default=None, help='Number of dev examples (None => full dataset)')
    parser.add_argument('--eval_examples', type=int, default=None, help='Number of eval examples (None => full dataset)')
    
    # Training steps - NEW ARGUMENTS
    parser.add_argument('--max_steps', type=int, default=2000, help='Maximum training steps')
    parser.add_argument('--eval_steps', type=int, default=50, help='Evaluate every N steps')
    parser.add_argument('--scheduler', type=str, choices=['plateau','linear','cosine','constant','constant_with_warmup'], default='constant_with_warmup', help='LR scheduler type')
    parser.add_argument('--warmup_steps', type=int, default=200, help='Number of warmup steps (overrides ratio if > 0)')
    parser.add_argument('--warmup_ratio', type=float, default=0.0, help='Warmup steps as a fraction of max_steps (used if warmup_steps==0)')
    parser.add_argument('--sgd_accum_steps', type=int, default=1, help='Gradient accumulation steps for SGD client path')
    parser.add_argument('--sched_factor', type=float, default=0.5, help='LR scheduler reduce factor')
    parser.add_argument('--sched_patience', type=int, default=6, help='LR scheduler patience')
    parser.add_argument('--sched_threshold', type=float, default=1e-2, help='LR scheduler threshold')
    parser.add_argument('--sched_threshold_mode', type=str, default='rel', choices=['rel','abs'], help='LR scheduler threshold mode')
    parser.add_argument('--sched_cooldown', type=int, default=1, help='LR scheduler cooldown')
    parser.add_argument('--sched_min_lr', type=float, default=1e-5, help='LR scheduler minimum learning rate')
    
    # ZOO parameters
    parser.add_argument('--mu', type=float, default=1e-3, help='ZOO perturbation scale (base, will be RMS-scaled)')
    parser.add_argument('--num_pert', type=int, default=1, help='ZOO perturbations (antithetic pairs internally)')
    parser.add_argument('--estimator', type=str, choices=['central','forward'], default='central', help='Finite-diff estimator type for ZOO/g_cut')
    parser.add_argument('--zoo_perturbation_type', type=str, choices=['rademacher','bernoulli','gaussian'], default='gaussian', help='Perturbation distribution for ZOO (default gaussian=N(0,1))')
    parser.add_argument('--zoo_mu_scaling', type=str, choices=['fixed','rms'], default='fixed', help='Mu scaling method for ZOO (fixed=use mu directly like MeZO, rms=scale by parameter RMS)')
    parser.add_argument('--use_zeroth_order', action='store_true', help='Use ZOO for client')
    parser.add_argument('--use_zeroth_order_server', action='store_true', help='Use ZOO for server')
    
    
    
    parser.add_argument('--zoo_max_grad_value', type=float, default=0.0, help='Maximum gradient norm threshold for adaptive clipping in ZOO estimator (0.0 disables clipping)')
    # Enable sequential two-phase ZOO updates (client then server)
    parser.add_argument('--sequential_zoo', action='store_true',
                        help='Use sequential ZOO updates (Phase 1: client update with server fixed; Phase 2: server update with client fixed)')
    # Make g_cut variance-reduced objective DISABLED by default
    parser.add_argument('--force_phase2_probe', type=int, default=0, help='Force Phase 2 to run for first N steps (diagnostic)')
    # Using parameter perturbations (like standard MeZO) is more stable than activation perturbations
    # Activation perturbations cause exponential amplification through layers (see GRADIENT_EXPLOSION_ANALYSIS.md)
    # Set zoo_use_gcut=True only if you need variance reduction and understand the trade-offs
    parser.add_argument('--zoo_use_gcut', action='store_true', help='Enable variance-reduced ZOO g_cut objective (uses activation perturbations, may cause instability)')
    parser.set_defaults(zoo_use_gcut=False)  # Changed default: use parameter perturbations only
    parser.add_argument('--server_sgd_warmup_steps', type=int, default=800, help='Delay server local SGD for N steps in client-ZOO mode')
    parser.add_argument('--server_sgd_every', type=int, default=1, help='Perform server local SGD every K steps after warmup')
    
    # Evaluation
    parser.add_argument('--evaluate_every', type=int, default=1, help='Evaluate every N epochs')
    parser.add_argument('--f1_method', type=str, default='micro', 
                       choices=['micro', 'macro', 'sequence'],
                       help='F1 score calculation method')

    parser.add_argument(
        '--task',
        choices=[
            'squad','xsum','drop','sst2',
            'boolq','copa','multirc','cb','wic','wsc','record','rte'
        ],
        default='squad',
        help='Task/dataset to load'
    )
    parser.add_argument('--tuning', choices=['prefix','lora','full','none'], default='prefix')
    parser.add_argument('--lora_r', type=int, default=8)
    parser.add_argument('--lora_alpha', type=int, default=16)
    parser.add_argument('--lora_dropout', type=float, default=0.0)
    parser.add_argument('--lora_targets', type=str, default='q_proj,v_proj')
    
    # Network configuration
    parser.add_argument('--host', type=str, default='localhost', help='client host to bind')
    parser.add_argument('--port', type=int, default=12345, help='client port to bind')
    
    # Unified precision flag governing compute dtype and wire payloads
    parser.add_argument('--precision', choices=['fp16','fp32'], default='fp32',
                        help='Global precision for compute and wire payloads (fp16 or fp32)')
    # Control whether to step LR scheduler on eval loss for ZOO (default off)
    parser.add_argument('--sched_step_on_eval', action='store_true', help='Also step LR scheduler on eval loss (ZOO)')
    # Evaluation device control
    parser.add_argument('--eval_on_cpu', action='store_true', help='Run evaluation on CPU to reduce GPU memory usage')
    
    args = parser.parse_args()
    return args

if __name__ == "__main__":
    try:
        print("STARTING ENHANCED SPLIT LEARNING client")
        print("=" * 60)

        # Attempt Hugging Face login early to avoid rate limits when loading hub assets
        try_hf_login()

        args = parse_args()
        torch.manual_seed(args.seed)
        np.random.seed(args.seed)
        
        print(f"  Configuration:")
        print(f"   Model: {args.model_name}")
        print(f"   Batch size: {args.train_batch_size}")
        print(f"   Max length: {args.max_length}")
        print(f"   ZOO client: {args.use_zeroth_order}")
        print(f"   ZOO server: {args.use_zeroth_order_server}")
        print(f"   Sequential ZOO: {getattr(args, 'sequential_zoo', False)}")
        try:
            print(f"   Force Phase2 probe: {int(getattr(args, 'force_phase2_probe', 0))}")
        except Exception:
            pass
        print(f"   F1 method: {args.f1_method}")
        if args.use_zeroth_order:
            print(f"   ZOO mu: {args.mu}")
            print(f"   ZOO perturbations: {args.num_pert}")
            print(f"   ZOO use g_cut objective: {getattr(args, 'zoo_use_gcut', False)}")
        print(f"   Precision: {getattr(args, 'precision', 'fp32')}")
        
        # Device configuration
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"  Using device: {device}")
        
        # Load tokenizer
        print(f"  Loading tokenizer: {args.model_name}")
        tokenizer = safe_get_hf_tokenizer(args.model_name)
        
        print("  Tokenizer loaded successfully")

        # Create models
        print("  Creating models...")
        if args.tuning == 'lora':
            client_model = LoRAclientModel(
                args.model_name, args.cut_layer,
                r=args.lora_r, alpha=args.lora_alpha, dropout=args.lora_dropout,
                torch_dtype=(torch.float16 if str(getattr(args, 'precision', 'fp32')).lower() == 'fp16' else torch.float32),
                targets=tuple(args.lora_targets.split(','))
            ).to(device)
            trainable_params = list(client_model.trainable_parameters())
            print(f"client owns layers [0, {args.cut_layer-1}] with LoRA r={args.lora_r}, alpha={args.lora_alpha}")
            _assert_only_expected_trainables(client_model, args.tuning, side="client")

        elif args.tuning == 'full':
            client_model = clientFullModel(args.model_name, cut_layer=args.cut_layer).to(device)
            # Restrict requires_grad to embeddings + layers [0..cut-1]
            # Ensure flags are correct
            for n, p in client_model.base_model.named_parameters():
                req = False
                if n.startswith('model.decoder.embed_'):
                    req = True
                elif n.startswith('model.decoder.layernorm_embedding'):
                    req = True
                elif '.model.decoder.layers.' in ('.' + n):
                    try:
                        after = n.split('model.decoder.layers.')[1]
                        idx = int(after.split('.')[0])
                        req = (idx < args.cut_layer)
                    except Exception:
                        req = False
                p.requires_grad = req
            trainable_params = [p for p in client_model.base_model.parameters() if p.requires_grad]
            print(f"client full-FT layers [0, {args.cut_layer-1}] enabled; embeddings trainable: True")
            _assert_only_expected_trainables(client_model, args.tuning, side="client")

        else:
            # existing PrefixKV client path (keep yours)
            client_model = clientKVOnly(args.model_name, cut_layer=args.cut_layer, num_prefix=args.num_prefix,
                                        torch_dtype=(torch.float16 if str(getattr(args, 'precision', 'fp32')).lower() == 'fp16' else torch.float32)).to(device)
            trainable_params = list(client_model.kv.parameters())  # unchanged
            _assert_only_expected_trainables(client_model, args.tuning, side="client")


        if args.tuning == 'lora':
            assert all((not p.requires_grad) for n,p in client_model.named_parameters()
                    if ("lora_A" not in n and "lora_B" not in n)), "Only LoRA params must be trainable in LoRA mode!"
        
        full_model = FullLLMModel(args.model_name, cut_layer=args.cut_layer, num_prefix=args.num_prefix,
                                  torch_dtype=(torch.float16 if str(getattr(args, 'precision', 'fp32')).lower() == 'fp16' else torch.float32)).to(device)
        # Only attach client KV in prefix mode; in LoRA there is no live client KV to use
        if args.tuning != 'lora':
            full_model.attach_live_client_kv(client_model.kv)

        # Synchronize models
        print("  Models created and synchronized")
        # Create data loaders
        print(" Creating dataloaders...")
        if getattr(args, "task", "squad") == "squad":
            train_loader, eval_loader = get_squad_dataloaders(args, tokenizer)
        else:
            print(f" Creating dataloaders for task: {args.task}")
            train_loader, eval_loader = get_task_dataloaders(
                args.task,
                tokenizer,
                train_batch_size=args.train_batch_size,
                test_batch_size=args.test_batch_size,
                max_length=args.max_length,
                num_train_examples=args.train_examples,
                num_eval_examples=args.eval_examples,
            )
        
        # Check dataset class balance for classification tasks (especially SST-2)
        if args.task == 'sst2':
            print("  Checking SST-2 dataset class balance...")
            try:
                # Sample a few batches to check class distribution
                class_counts = {'great': 0, 'terrible': 0, 'unknown': 0}
                total_samples = 0
                for i, batch in enumerate(train_loader):
                    if i >= 10:  # Check first 10 batches
                        break
                    if 'formatted_text' in batch:
                        for text in batch['formatted_text']:
                            total_samples += 1
                            if 'Answer: great' in text or 'Answer:great' in text:
                                class_counts['great'] += 1
                            elif 'Answer: terrible' in text or 'Answer:terrible' in text:
                                class_counts['terrible'] += 1
                            else:
                                class_counts['unknown'] += 1
                
                print(f"  Class distribution in first {total_samples} samples:")
                print(f"    'great' (positive): {class_counts['great']} ({class_counts['great']/max(total_samples,1)*100:.1f}%)")
                print(f"    'terrible' (negative): {class_counts['terrible']} ({class_counts['terrible']/max(total_samples,1)*100:.1f}%)")
                if class_counts['unknown'] > 0:
                    print(f"    'unknown': {class_counts['unknown']} ({class_counts['unknown']/max(total_samples,1)*100:.1f}%)")
                
                # Check for severe imbalance
                if total_samples > 0:
                    ratio = max(class_counts['great'], class_counts['terrible']) / max(min(class_counts['great'], class_counts['terrible']), 1)
                    if ratio > 2.0:
                        print(f"  ⚠️ WARNING: Class imbalance detected (ratio={ratio:.2f})")
                        print(f"    This may cause the model to bias towards the majority class.")
                    else:
                        print(f"  ✅ Dataset appears balanced (ratio={ratio:.2f})")
            except Exception as balance_error:
                print(f"  ⚠️ Could not check class balance: {balance_error}")
        
        print("  Dataloaders created successfully")
        # Setup optimizer
        print("  Setting up optimizer...")
        if args.use_zeroth_order:
            # ZOO: use vanilla SGD (no momentum)
            optimizer = optim.SGD(trainable_params, lr=args.zoo_lr, momentum=0.0, weight_decay=args.weight_decay)
        else:
            # Regular optimizer path: SGD only
            optimizer = optim.SGD(
                trainable_params,
                lr=args.lr,
                momentum=args.momentum,
                weight_decay=args.weight_decay,
            )

        # Scheduler selection
        if getattr(args, 'scheduler', 'plateau') == 'plateau':
            try:
                scheduler = optim.lr_scheduler.ReduceLROnPlateau(
                    optimizer,
                    mode='min',
                    factor=args.sched_factor,
                    patience=args.sched_patience,
                    threshold=args.sched_threshold,
                    threshold_mode=args.sched_threshold_mode,
                    cooldown=args.sched_cooldown,
                    min_lr=args.sched_min_lr,
                    verbose=True,
                )
            except TypeError:
                scheduler = optim.lr_scheduler.ReduceLROnPlateau(
                    optimizer,
                    mode='min',
                    factor=args.sched_factor,
                    patience=args.sched_patience,
                    threshold=args.sched_threshold,
                )
            print("✅ Optimizer ready (Plateau scheduler)")
        else:
            total_steps = max(1, int(args.max_steps))
            warmup_steps = int(args.warmup_steps) if int(args.warmup_steps) > 0 else int(float(args.warmup_ratio) * float(total_steps))
            warmup_steps = max(0, warmup_steps)

            if args.scheduler == 'linear':
                def lr_lambda(current_step: int):
                    if warmup_steps > 0 and current_step < warmup_steps:
                        return float(current_step) / float(max(1, warmup_steps))
                    progress = float(current_step - warmup_steps) / float(max(1, total_steps - warmup_steps))
                    return max(0.0, 1.0 - progress)
            elif args.scheduler == 'cosine':
                import math
                def lr_lambda(current_step: int):
                    if warmup_steps > 0 and current_step < warmup_steps:
                        return float(current_step) / float(max(1, warmup_steps))
                    progress = float(current_step - warmup_steps) / float(max(1, total_steps - warmup_steps))
                    return 0.5 * (1.0 + math.cos(math.pi * min(1.0, max(0.0, progress))))
            else:  # constant with warmup
                def lr_lambda(current_step: int):
                    if warmup_steps > 0 and current_step < warmup_steps:
                        return float(current_step) / float(max(1, warmup_steps))
                    return 1.0

            scheduler = optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lr_lambda)
            print(f"✅ Optimizer ready ({args.scheduler} scheduler, warmup_steps={warmup_steps})")
        try:
            print(f"  Initial LR: {_get_current_lr(optimizer):.6g}")
        except Exception:
            pass
        
        # Setup network
        print("Setting up network...")
        client_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        client_socket.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        client_socket.bind((args.host, args.port))
        client_socket.listen(1)
        
        # Debug: Print model dimensions
        print(f"  Debug - client model info:")
        
        print("=" * 60)
        print("client READY - WAITING FOR server")
        print("=" * 60)
        print(f"client listening on {args.host}:{args.port}")
        print("Start server with same parameters")
        print("=" * 60)
        
        # Accept server connection
        conn, addr = client_socket.accept()
        print(f"✅ server connected from {addr}")
        
        # Initialize compute/comm tracker for the client process
        try:
            _srv_tracker = ComputeTracker(device=device, role="client")
        except Exception:
            _srv_tracker = None
        trainer = Trainer(conn, tracker=_srv_tracker)
        server_config = trainer.receive_data()
        print(f"Received server config: {server_config}")
        # Handshake fix: respect server-declared optimization mode
        try:
            args.use_zeroth_order_server = (not bool(server_config.get('server_sgd', True)))
            print(f"   ZOO server (from handshake): {args.use_zeroth_order_server}")
        except Exception:
            pass
        
        print("Starting training...")
        
        # Training loop
        for epoch in range(args.epochs):
            print(f"\nEpoch {epoch+1}/{args.epochs}")
            
            # Choose training method based on client configuration only
            if args.use_zeroth_order:
                train_loss, train_acc = train_split_learning_zoo(
                    client_model, full_model, train_loader, eval_loader, optimizer, 
                    scheduler, trainer, device, tokenizer, args
                )
            else:
                train_loss, train_acc = train_split_learning_sgd(
                    client_model, full_model, train_loader, eval_loader, optimizer, 
                    scheduler, trainer, device, tokenizer, args
                )
            
            print(f"✅ Epoch {epoch+1} Training: Loss {train_loss:.4f}, Acc {train_acc:.6f}")
            
            # Evaluation
            if (epoch + 1) % args.evaluate_every == 0:
                print(f"Running evaluation...")
                eval_loss, eval_acc, eval_f1, eval_em = evaluate_model(
                    full_model, eval_loader, device, tokenizer, args,
                    client_model=client_model, trainer=trainer
                )
                print(f"\nEPOCH {epoch+1} RESULTS:")
                print(f"{'='*60}")
                print(f"TRAINING   - Loss: {train_loss:.4f}, Accuracy: {train_acc:.4f}")
                print(f"EVALUATION - Loss: {eval_loss:.4f}, Accuracy: {eval_acc:.4f}, F1: {eval_f1:.4f}")
                print(f"{'='*60}")
                
        # Final evaluation
        # Mark eval phase for final evaluation
        try:
            if _srv_tracker is not None:
                _srv_tracker.set_phase('eval')
        except Exception:
            pass
        print("\nFinal model evaluation...")
        _moved = False
        _eval_device = device
        try:
            if bool(getattr(args, 'eval_on_cpu', False)) and str(getattr(args, 'tuning', 'prefix')) == 'lora':
                full_model.to(torch.device('cpu'))
                _eval_device = torch.device('cpu')
                _moved = True
            final_loss, final_acc, final_f1, final_em = evaluate_model(
                full_model, eval_loader, _eval_device, tokenizer, args, client_model=client_model, trainer=trainer
            )
        finally:
            if _moved:
                full_model.to(device)
        # Return to train phase (done)
        try:
            if _srv_tracker is not None:
                _srv_tracker.set_phase('train')
        except Exception:
            pass

        # Print and persist client compute/communication summary
        try:
            if _srv_tracker is not None:
                _ = _srv_tracker.update_memory()
                summary = _srv_tracker.print_summary()
                # Save JSON summary next to logs
                try:
                    _srv_tracker.save_stats('client_stats_squad.json')
                except Exception:
                    pass
        except Exception:
            pass

        # now signal completion (non-fatal if server already closed)
        try:
            trainer.send_data({'type': 'training_complete'})
        except Exception as _e:
            print(f"⚠️ Could not notify server of completion (likely closed): {_e}")

        
        print(f"\nFINAL RESULTS ({getattr(args, 'task', 'squad').upper()}):")
        print(f"{'='*60}")
        print(f"Model: {args.model_name}")
        print(f"Epochs: {args.epochs}")
        print(f"Optimization: {'ZOO' if args.use_zeroth_order else 'SGD'}")
        print(f"Final Loss: {final_loss:.4f}")
        print(f"Final Accuracy: {final_acc:.4f}")
        print(f"Final F1 Score: {final_f1:.4f}")
        print(f"{'='*60}")
        
    except Exception as e:
        print(f"❌ CRITICAL client ERROR: {e}")
        print("  Full traceback:")
        traceback.print_exc()
        sys.exit(1)
        
    finally:
        try:
            if 'conn' in locals():
                conn.close()
            if 'client_socket' in locals():
                client_socket.close()
        except:
            pass
        print("  client shutdown complete")