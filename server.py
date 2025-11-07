import socket
import pickle
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from transformers import (
    AutoModelForCausalLM, 
    AutoTokenizer,
    AutoConfig
)
import numpy as np
import argparse
import traceback
from SGDGradientEst import StochasticGradientApproximator
import types
from prefix_kv import (
    PrefixKV, 
    merge_past_key_values, 
    flatten_grad_state
)
# Global KV model holder (set in main)
kv_model = None

# Diagnostics: track server parameter change deltas
_SERVER_SGD_STEP = 0
_SERVER_SGD_INITIALS = None
_SERVER_ZOO_INITIALS = None

# Optional global CE controls (configured in main)
_CE_CLASS_WEIGHTS = None  # torch.FloatTensor[vocab_size] or None
_LABEL_SMOOTHING = 0.0    # float >= 0.0

from lora import (
    apply_lora_to_opt, 
    iter_lora_parameters, 
    get_lora_state_dict
)

# Compute/memory & communication tracker
from compute_tracker import ComputeTracker

try:
    from transformers.cache_utils import DynamicCache, StaticCache
except Exception:
    DynamicCache = None
    StaticCache = None

class _NoPrefixStub:
    def get_local_past(self, bsz): return {}
    def set_requires_grad(self, flag: bool): pass


def verify_parameter_restoration(model: nn.Module, stage: str = "unknown") -> bool:
    """
    Verify that model parameters are restored after ZOO estimation.

    Should be called BEFORE optimizer.step() to ensure we didn't leave the model
    in a perturbed state, and optionally after step to confirm parameters changed.
    """
    import hashlib

    try:
        param_str = ""
        for param in model.parameters():
            if param.requires_grad:
                param_str += str(param.data.detach().cpu().numpy().tobytes())

        current_hash = hashlib.md5(param_str.encode()).hexdigest()

        if not hasattr(verify_parameter_restoration, 'original_hashes'):
            verify_parameter_restoration.original_hashes = {}
            verify_parameter_restoration.original_hashes[id(model)] = current_hash
            print(f"[{stage}] Stored original parameter hash: {current_hash[:8]}")
            return True

        original_hash = verify_parameter_restoration.original_hashes.get(id(model))
        if original_hash != current_hash:
            print(f"[{stage}] WARNING: Parameters changed unexpectedly!")
            print(f"  Original hash: {original_hash[:8]}")
            print(f"  Current hash: {current_hash[:8]}")
            return False
        return True
    except Exception:
        # Non-fatal; only for diagnostics
        return True


def handle_zoo_server_update(server, kv_model, optimizer, grad_estimator, args, device, pkt: dict):
    """
    Phase 2 of sequential ZOO: update server parameters with fixed h_cut from client.
    Client already updated; server perturbs only its own params and steps optimizer.
    """
    client_data = pkt.get("data", {})
    meta = pkt.get("meta", {})
    try:
        print(f"[server-phase2] received zoo_server_update step={meta.get('step')} phase2_id={meta.get('phase2_id')} num_pert={meta.get('num_perturbations')}")
    except Exception:
        pass

    num_pert = int(meta.get("num_perturbations", getattr(args, 'num_pert', 10)))
    base_seed = int(meta.get("base_seed", getattr(args, 'seed', 0)))
    mu = float(meta.get("mu", getattr(args, 'mu', 1e-3)))
    phase2_id = meta.get("phase2_id", None)

    model_ref = getattr(kv_model, "base_model", kv_model)
    param0 = next(model_ref.parameters())
    target_device = param0.device
    target_dtype = param0.dtype

    h_cut_fixed = client_data["h_cut"].to(device=target_device, dtype=target_dtype)
    attn_mask = client_data.get("attention_mask", None)
    labels = client_data.get("labels", None)

    if attn_mask is not None:
        attn_mask = attn_mask.to(device=target_device)
    if labels is not None:
        labels = labels if torch.is_tensor(labels) else torch.as_tensor(labels)
        labels = labels.to(device=target_device)

    if attn_mask is not None and labels is not None:
        _, attn_mask, labels = right_trim(torch.zeros_like(attn_mask), attn_mask, labels)

    cut = int(client_data.get("cut_layer", getattr(args, 'cut_layer', 1)))

    server_params = list(getattr(kv_model, "trainable_parameters", 
                                 lambda: kv_model.server_kv.parameters())())

    print(f"\n[SERVER] Sequential ZOO Phase 2: updating {len(server_params)} parameter tensors...")
    print(f"[SERVER] Using {num_pert} perturbations with μ={mu}")
    try:
        print(f"[SERVER] base_seed={base_seed}")
    except Exception:
        pass

    # Initialize initial snapshot for ZOO server params if not already
    global _SERVER_ZOO_INITIALS
    if _SERVER_ZOO_INITIALS is None:
        try:
            _SERVER_ZOO_INITIALS = {}
            _base_mod = getattr(kv_model, "base_model", kv_model)
            for _name, _p in _base_mod.named_parameters():
                if _p.requires_grad:
                    _SERVER_ZOO_INITIALS[_name] = _p.detach().clone()
        except Exception:
            _SERVER_ZOO_INITIALS = None

    optimizer.zero_grad(set_to_none=True)
    for p in server_params:
        if p.grad is not None:
            p.grad.zero_()
        else:
            p.grad = torch.zeros_like(p, device=p.device, dtype=p.dtype)

    server_loss_diffs = []

    kv_model.train()

    # Parameter restoration safety check (before ZOO accumulation)
    try:
        _ = verify_parameter_restoration(model_ref, "server_before_ZOO")
    except Exception:
        pass

    for pert_idx in range(num_pert):
        pert_seed = base_seed + pert_idx

        # θ_s -> θ_s + μ z
        with torch.no_grad():
            for idx, p in enumerate(server_params):
                gen = torch.Generator(device=p.device)
                gen.manual_seed(pert_seed * 1000003 + idx)
                z = torch.empty_like(p)
                z.normal_(mean=0.0, std=1.0, generator=gen)
                p.add_(z, alpha=mu)

        # loss at +
        with torch.no_grad():
            out_plus = _server_forward_from_cut(
                kv_model,
                h_cut=h_cut_fixed.detach(),
                attention_mask=attn_mask,
                labels=labels,
                cut=cut,
                compute_g_cut=False,
                return_loss_tensor=False,
            )
        loss_plus = float(out_plus["loss"])

        # θ_s -> θ_s - μ z (from + to -)
        with torch.no_grad():
            for idx, p in enumerate(server_params):
                gen = torch.Generator(device=p.device)
                gen.manual_seed(pert_seed * 1000003 + idx)
                z = torch.empty_like(p)
                z.normal_(mean=0.0, std=1.0, generator=gen)
                p.add_(z, alpha=-2.0 * mu)

        # loss at -
        with torch.no_grad():
            out_minus = _server_forward_from_cut(
                kv_model,
                h_cut=h_cut_fixed.detach(),
                attention_mask=attn_mask,
                labels=labels,
                cut=cut,
                compute_g_cut=False,
                return_loss_tensor=False,
            )
        loss_minus = float(out_minus["loss"])

        # restore to original
        with torch.no_grad():
            for idx, p in enumerate(server_params):
                gen = torch.Generator(device=p.device)
                gen.manual_seed(pert_seed * 1000003 + idx)
                z = torch.empty_like(p)
                z.normal_(mean=0.0, std=1.0, generator=gen)
                p.add_(z, alpha=mu)

        loss_diff = abs(loss_plus - loss_minus)
        server_loss_diffs.append(loss_diff)
        fd = (loss_plus - loss_minus) / (2.0 * mu)

        # accumulate grads: grad += fd * z
        with torch.no_grad():
            for idx, p in enumerate(server_params):
                gen = torch.Generator(device=p.device)
                gen.manual_seed(pert_seed * 1000003 + idx)
                z = torch.empty_like(p)
                z.normal_(mean=0.0, std=1.0, generator=gen)
                if p.grad is None:
                    p.grad = torch.zeros_like(p)
                p.grad.add_(z, alpha=fd)

        if pert_idx % 5 == 0:
            print(f"  [SERVER] pert {pert_idx}/{num_pert}: L+={loss_plus:.4f}, L-={loss_minus:.4f}, diff={loss_diff:.4f}, FD={fd:.2f}")

    # average grads
    with torch.no_grad():
        for p in server_params:
            if p.grad is not None:
                p.grad.div_(num_pert)

    # grad norm
    server_grad_norm = 0.0
    with torch.no_grad():
        for p in server_params:
            if p.grad is not None:
                server_grad_norm += p.grad.norm().item() ** 2
    server_grad_norm = server_grad_norm ** 0.5

    print(f"\n[SERVER] ZOO gradient computed:")
    print(f"  Gradient norm: {server_grad_norm:.6f}")
    try:
        avg_ld = float(np.mean(server_loss_diffs)) if server_loss_diffs else 0.0
        std_ld = float(np.std(server_loss_diffs)) if server_loss_diffs else 0.0
        print(f"  Avg loss diff: {avg_ld:.6f}")
    except Exception:
        avg_ld, std_ld = 0.0, 0.0

    # clip if configured
    try:
        if float(getattr(args, 'clip_grad_norm', 0.0)) > 0.0:
            torch.nn.utils.clip_grad_norm_(server_params, max_norm=float(args.clip_grad_norm))
            # Recompute and report norm after clipping
            try:
                post_clip_sq = 0.0
                for p in server_params:
                    if p.grad is not None:
                        post_clip_sq += float(p.grad.norm().item()) ** 2
                post_clip_norm = post_clip_sq ** 0.5
                print(f"  Gradient norm after clip: {post_clip_norm:.6f} (threshold={float(args.clip_grad_norm):.6g})")
            except Exception:
                pass
    except Exception:
        pass

    # Parameter restoration safety check (after ZOO, before optimizer step)
    try:
        _ = verify_parameter_restoration(model_ref, "server_after_ZOO")
    except Exception:
        pass

    optimizer.step()

    # Parameter restoration vs step check (after optimizer step)
    try:
        _ = verify_parameter_restoration(model_ref, "server_after_step")
    except Exception:
        pass

    print(f"[SERVER] Parameters updated!\n")

    # If client provided a global step, print param deltas at step 10
    try:
        _step_val = int(pkt.get("meta", {}).get("step", -1))
    except Exception:
        _step_val = -1
    if _step_val == 10 and _SERVER_ZOO_INITIALS:
        try:
            _base_mod = getattr(kv_model, "base_model", kv_model)
            print("[SERVER] Parameter max changes after 10 steps (ZOO server):")
            for _name, _p in _base_mod.named_parameters():
                if _p.requires_grad and _name in _SERVER_ZOO_INITIALS:
                    _delta = (_p - _SERVER_ZOO_INITIALS[_name]).abs().max().item()
                    print(f"{_name}: max param change = {_delta:.6e}")
        except Exception:
            pass

    server.send_data({
        "type": "zoo_phase_complete",
        "phase": "server_update",
        "grad_norm": float(server_grad_norm),
        "loss_diff_avg": avg_ld,
        "loss_diff_std": std_ld,
    })

def right_trim(input_ids, attention_mask, labels=None):
    """Remove right padding for efficiency"""
    L = attention_mask.sum(dim=1).max().item()
    input_ids = input_ids[:, :int(L)]
    attention_mask = attention_mask[:, :int(L)]
    if labels is not None: 
        labels = labels[:, :int(L)]
    return input_ids, attention_mask, labels

class LoRAserverModel(nn.Module):
    """
    server-side when tuning=LoRA. Keeps full base_model, injects LoRA only into layers [cut..L-1].
    Presents client_kv_mirror/server_kv stubs so the rest of your code (forward, masks) stays unified.
    """
    def __init__(self, model_name: str, total_layers: int, cut_layer: int,
                 r: int = 8, alpha: int = 16, dropout: float = 0.0,
                 targets=("q_proj","v_proj"), torch_dtype: torch.dtype = torch.float32):
        super().__init__()
        self.base_model = AutoModelForCausalLM.from_pretrained(model_name, torch_dtype=torch_dtype, device_map=None)
        for p in self.base_model.parameters():
            p.requires_grad = False
        self.total_layers = total_layers
        self.cut_layer = cut_layer

        apply_lora_to_opt(self.base_model,
                          targets=tuple(targets),
                          layer_range=(cut_layer, total_layers-1),
                          r=r, lora_alpha=alpha, lora_dropout=dropout)

        # keep interface used elsewhere
        self.client_kv_mirror = _NoPrefixStub()
        self.server_kv       = _NoPrefixStub()

    def trainable_parameters(self):
        return iter_lora_parameters(self.base_model, layer_range=(self.cut_layer, self.total_layers-1))

class FullserverModel(nn.Module):
    """
    server-side when tuning=full. Keeps full base_model; trains layers [cut..L-1] + final_layer_norm + lm_head.
    Presents no-prefix stubs so unified forward-from-cut path works without KV prefixes.
    """
    def __init__(self, model_name: str, total_layers: int, cut_layer: int, torch_dtype: torch.dtype = torch.float32):
        super().__init__()
        self.base_model = AutoModelForCausalLM.from_pretrained(model_name, torch_dtype=torch_dtype, device_map=None)
        for p in self.base_model.parameters():
            p.requires_grad = False
        self.total_layers = total_layers
        self.cut_layer = cut_layer

        # Enable grads for server half: decoder layers [cut..L-1], final_layer_norm, lm_head
        for name, p in self.base_model.named_parameters():
            req = False
            if name.startswith('model.decoder.final_layer_norm'):
                req = True
            elif name.startswith('lm_head'):
                req = True
            elif '.model.decoder.layers.' in ('.' + name):
                try:
                    after = name.split('model.decoder.layers.')[1]
                    idx = int(after.split('.')[0])
                    req = (idx >= cut_layer)
                except Exception:
                    req = False
            p.requires_grad = req

        # Prefix stubs to satisfy forward path
        self.client_kv_mirror = _NoPrefixStub()
        self.server_kv       = _NoPrefixStub()

    def trainable_parameters(self):
        return (p for p in self.base_model.parameters() if p.requires_grad)


@torch.no_grad()
def _estimate_g_cut_fd(kv_model, h_cut, attention_mask, labels, cut, mu=1e-3, num_pert=10, mode: str = 'central'):
    """
    DEPRECATED: This function perturbs activations, causing exponential amplification.
    
    For stable split learning MeZO, use parameter perturbations instead (via StochasticGradientApproximator).
    This function is only kept for optional variance reduction when zoo_use_gcut=True.
    
    See GRADIENT_EXPLOSION_ANALYSIS.md for why activation perturbations are problematic.
    """
    # h_cut on the right device/dtype already
    h0 = h_cut
    g_hat = torch.zeros_like(h0, dtype=h0.dtype)

    for _ in range(num_pert):
        u = torch.randn_like(h0, dtype=h0.dtype)
        with torch.no_grad():
            lp = _server_forward_from_cut(
                kv_model, h_cut=h0 + mu*u,
                attention_mask=attention_mask, labels=labels, cut=cut,
                compute_g_cut=False, return_loss_tensor=False
            )["loss"]
            if str(mode).lower() == 'central':
                ln = _server_forward_from_cut(
                    kv_model, h_cut=h0 - mu*u,
                    attention_mask=attention_mask, labels=labels, cut=cut,
                    compute_g_cut=False, return_loss_tensor=False
                )["loss"]
                scale = (lp - ln) / (2.0*mu)
            else:
                ln = _server_forward_from_cut(
                    kv_model, h_cut=h0,
                    attention_mask=attention_mask, labels=labels, cut=cut,
                    compute_g_cut=False, return_loss_tensor=False
                )["loss"]
                scale = (lp - ln) / (mu)

        g_hat.add_(scale * u)

    g_hat.div_(float(num_pert))
    return g_hat

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


def _neg_inf(dtype: torch.dtype) -> float:
    return torch.finfo(dtype).min

def _build_class_weight_vector(tokenizer, vocab_size: int, spec_tokens: str = None, spec_ids: str = None):
    """
    Build a vocab-level weight vector for CE.
    - spec_tokens: comma-separated token:weight pairs (e.g., "terrible:1.5,great:1.0").
      We will try both raw and leading-space variants for GPT-style tokenizers and only
      accept single-token encodings.
    - spec_ids: comma-separated id:weight pairs (e.g., "123:1.5,456:1.0").
    Returns torch.FloatTensor[vocab_size].
    """
    import re
    w = torch.ones(int(vocab_size), dtype=torch.float32)

    # Parse id:weight pairs first (takes precedence if overlaps)
    if spec_ids:
        for kv in str(spec_ids).split(','):
            kv = kv.strip()
            if not kv:
                continue
            if ':' not in kv:
                continue
            k, v = kv.split(':', 1)
            try:
                tid = int(k.strip())
                wt = float(v.strip())
                if 0 <= tid < int(vocab_size):
                    w[tid] = float(wt)
            except Exception:
                continue

    # Parse token:weight pairs; map tokens to single ids if possible
    if spec_tokens:
        for kv in str(spec_tokens).split(','):
            kv = kv.strip()
            if not kv or ':' not in kv:
                continue
            tok, v = kv.split(':', 1)
            tok = tok.strip()
            try:
                wt = float(v.strip())
            except Exception:
                continue

            cand_forms = [tok]
            # For GPT-like tokenizers, answers often include a leading space
            if not tok.startswith(' '):
                cand_forms.append(' ' + tok)

            tid = None
            for form in cand_forms:
                try:
                    ids = tokenizer.encode(form, add_special_tokens=False)
                except Exception:
                    ids = []
                if isinstance(ids, (list, tuple)) and len(ids) == 1:
                    tid = int(ids[0])
                    break
            if tid is not None and 0 <= tid < int(vocab_size):
                w[tid] = float(wt)

    return w

def _build_self_attn_mask(attention_mask: torch.Tensor,
                          tgt_len: int,
                          prefix_len: int,
                          dtype: torch.dtype,
                          device: torch.device) -> torch.Tensor:
    bsz = attention_mask.size(0)
    causal = torch.triu(
        torch.full((tgt_len, tgt_len), _neg_inf(dtype), device=device),
        diagonal=1
    )
    if prefix_len > 0:
        prefix_block = torch.zeros((tgt_len, prefix_len), dtype=dtype, device=device)
        base = torch.cat([prefix_block, causal], dim=-1)
    else:
        base = causal
    attn = base.unsqueeze(0).unsqueeze(1).expand(bsz, 1, tgt_len, prefix_len + tgt_len)
    pad = (1.0 - attention_mask.to(dtype))
    if prefix_len > 0:
        src_pad = torch.cat([torch.zeros((bsz, prefix_len), dtype=dtype, device=device), pad], dim=-1)
    else:
        src_pad = pad
    attn = attn + src_pad.view(bsz, 1, 1, prefix_len + tgt_len) * _neg_inf(dtype)
    return attn

def _server_forward_from_cut(
    kv_model,                       # your serverKVModel instance
    h_cut: torch.Tensor,
    attention_mask: torch.Tensor,
    labels: torch.Tensor,
    cut: int,
    compute_g_cut: bool = True,
    return_loss_tensor: bool = False,
    return_logits: bool = False,  # NEW: return logits for accuracy computation
    ):
    base_model = kv_model.base_model
    decoder    = base_model.model.decoder
    device     = next(base_model.parameters()).device
    dtype      = next(base_model.parameters()).dtype

    # on-device, correct dtype
    h = h_cut.to(device=device, dtype=dtype)
    if compute_g_cut or return_loss_tensor:  # Need grad if we're computing g_cut OR doing local update
        h.requires_grad_(True)

    attention_mask = attention_mask.to(device=device)
    labels = labels.to(device=device) if labels is not None else None

    # Figure out server prefix length P (for layers >= cut)
    try:
        prefix_len = int(kv_model.server_kv.k.shape[-2])  # [L, H, P, D]
    except Exception:
        prefix_len = 0

    tgt_len = h.shape[1]
    attn_mask_4d = _build_self_attn_mask(
        attention_mask=attention_mask,
        tgt_len=tgt_len,
        prefix_len=prefix_len,
        dtype=h.dtype,
        device=h.device,
    )

    # Build server past for layers >= cut
    bsz = h.size(0)
    # Build server past only if server_kv exists (prefix mode)
    server_past = {}
    if hasattr(kv_model, "server_kv") and hasattr(kv_model.server_kv, "get_local_past"):
        server_past = kv_model.server_kv.get_local_past(bsz)


    # Run layers cut..end with per-layer server prefixes
    for li in range(cut, len(decoder.layers)):
        layer = decoder.layers[li]
        pkv = server_past.get(li, None)
        if pkv is not None:
            pkv = _PrefixConcatCache(pkv[0], pkv[1])

        layer_out = layer(
            h,
            attention_mask=attn_mask_4d,
            layer_head_mask=None,
            past_key_value=pkv,        # _PrefixConcatCache or None
            output_attentions=False,
            use_cache=False,
        )
        h = layer_out[0] if isinstance(layer_out, (tuple, list)) else layer_out

    # Final norm + LM head
    if decoder.final_layer_norm is not None:
        h = decoder.final_layer_norm(h)
    logits = base_model.lm_head(h)

    if labels is not None and attention_mask is not None:
        labels = torch.where(attention_mask == 0, torch.tensor(-100, device=labels.device), labels)
        shift_logits = logits[:, :-1, :].contiguous()
        shift_labels = labels[:, 1:].contiguous()
        # Optional class weighting and label smoothing
        ce_weight = None
        try:
            if _CE_CLASS_WEIGHTS is not None:
                ce_weight = _CE_CLASS_WEIGHTS.to(device=shift_logits.device, dtype=shift_logits.dtype)
        except Exception:
            ce_weight = None

        flat_logits = shift_logits.view(-1, shift_logits.size(-1))
        flat_labels = shift_labels.view(-1)
        try:
            loss = F.cross_entropy(
                flat_logits,
                flat_labels,
                ignore_index=-100,
                reduction="mean",
                weight=ce_weight,
                label_smoothing=float(_LABEL_SMOOTHING) if float(_LABEL_SMOOTHING) > 0.0 else 0.0,
            )
        except TypeError:
            # Fallback for PyTorch without label_smoothing support
            loss = F.cross_entropy(
                flat_logits,
                flat_labels,
                ignore_index=-100,
                reduction="mean",
                weight=ce_weight,
            )
    else:
        loss = torch.zeros((), device=device, dtype=dtype)

    out = {"loss": float(loss.item())}
    if compute_g_cut:
        g_cut, = torch.autograd.grad(loss, h, retain_graph=return_loss_tensor)
        out["g_cut"] = g_cut.detach().to(h.dtype).cpu()
    
    if return_loss_tensor:
        out["loss_tensor"] = loss  # Return the torch tensor for local update
    
    # Also return logits if requested (for accuracy computation)
    # This allows client to compute training accuracy without extra forward passes
    if return_logits:
        out["logits"] = logits.detach().cpu().numpy()  # Return logits as numpy for serialization
    
    return out

def safe_get_hf_tokenizer(model_name):
    try:
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
        return tokenizer
    except Exception as e:
        print(f"⌠Failed to load tokenizer for {model_name}: {e}")
        raise

class serverKVModel(nn.Module):
    """
    server-side model that supports per-layer KV-prefix (only prefixes are trainable).
    It holds two PrefixKV modules:
      - client_kv_mirror: non-owned copy used to compute gradients for client (when client uses SGD)
      - server_kv: trainable prefixes for the server's layers
    We still run the full model on the server, but inject KV prefixes into all layers.
    """
    def __init__(self, model_name, total_layers, cut_layer, num_prefix=10, torch_dtype: torch.dtype = torch.float32):
        super().__init__()
        self.base_model = AutoModelForCausalLM.from_pretrained(model_name, torch_dtype=torch_dtype, device_map=None)
        for p in self.base_model.parameters():
            p.requires_grad = False

        self.total_layers = total_layers
        self.cut_layer = cut_layer
        self.hidden_size = self.base_model.config.hidden_size

        # client prefixes live on layers [0..cut-1]; server prefixes live on [cut..L-1]
        self.client_kv_mirror = PrefixKV(self.base_model.config, list(range(0, cut_layer)), num_prefix=num_prefix, device=self.base_model.device)
        self.server_kv = PrefixKV(self.base_model.config, list(range(cut_layer, total_layers)), num_prefix=num_prefix, device=self.base_model.device)
        # Only server prefixes are trainable by default
        for p in self.client_kv_mirror.parameters():
            p.requires_grad = False
        for p in self.server_kv.parameters():
            p.requires_grad = True


    def load_client_state(self, state_dict: dict):
        # state_dict expected to contain keys "k" and "v" tensors matching client_kv_mirror
        with torch.no_grad():
            if "k" in state_dict:
                self.client_kv_mirror.k.copy_(state_dict["k"].to(self.client_kv_mirror.k.device, dtype=self.client_kv_mirror.k.dtype))
            if "v" in state_dict:
                self.client_kv_mirror.v.copy_(state_dict["v"].to(self.client_kv_mirror.v.device, dtype=self.client_kv_mirror.v.dtype))

    def forward_full(self, input_ids, attention_mask, labels=None, require_client_grad=False):
        bsz = input_ids.size(0)

        # ---- 1) trim right padding (you already had this, keep it) ----
        if attention_mask is not None:
            valid_len = int(attention_mask.sum(dim=1).max().item())
            valid_len = max(valid_len, 1)
            if valid_len < input_ids.size(1):
                input_ids = input_ids[:, :valid_len]
                attention_mask = attention_mask[:, :valid_len]
                if labels is not None:
                    labels = labels[:, :valid_len]
        seq_len = input_ids.size(1)

        # ---- 2) build KV cache from prefixes (client + server) ----
        client_past = self.client_kv_mirror.get_local_past(bsz)
        server_past = self.server_kv.get_local_past(bsz)
        past_kv = merge_past_key_values(self.total_layers, client_past, server_past)
        self.client_kv_mirror.set_requires_grad(require_client_grad)

        # wrap to HF Cache if available (StaticCache/DynamicCache), else legacy tuple
        legacy = tuple(past_kv)
        cache = legacy
        try:
            if StaticCache is not None and hasattr(StaticCache, "from_legacy_cache"):
                cache = StaticCache.from_legacy_cache(legacy)
            elif DynamicCache is not None and hasattr(DynamicCache, "from_legacy_cache"):
                cache = DynamicCache.from_legacy_cache(legacy)
        except Exception:
            cache = legacy

        # ---- 3) compute past length (prefix length) for positions ----
        try:
            past_len = cache.get_seq_length()  # new HF Cache API
        except Exception:
            # derive from first layer K: [B, H, past_len, D]
            first_k = legacy[0][0] if isinstance(legacy, (list, tuple)) else None
            past_len = int(first_k.shape[2]) if isinstance(first_k, torch.Tensor) else 0

        # ---- 4) explicit position_ids with correct offset; no attention_mask ----
        position_ids = torch.arange(
            past_len, past_len + seq_len, device=input_ids.device, dtype=torch.long
        ).unsqueeze(0).expand(bsz, -1)

        outputs = self.base_model(
            input_ids=input_ids,
            attention_mask=None,         # we trimmed pads; avoid HF's mask-based pos path
            position_ids=position_ids,   # explicit, matches seq_len with past offset
            labels=labels,
            past_key_values=cache,
            use_cache=False,
        )
        return outputs

    def zero_prefix_grads(self):
        for mod in [self.client_kv_mirror, self.server_kv]:
            if hasattr(mod.k, "grad") and mod.k.grad is not None:
                mod.k.grad.zero_()
            if hasattr(mod.v, "grad") and mod.v.grad is not None:
                mod.v.grad.zero_()

    def trainable_parameters(self):
        # For prefix mode, the only trainables should be server_kv
        return self.server_kv.parameters()


class server:
    def __init__(self, host, port):
        self.socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self.socket.connect((host, port))
        self.tracker = None
        
    def send_data(self, data):
        try:
            if getattr(self, 'tracker', None) is not None:
                try:
                    if isinstance(data, dict) and ('mode' in data):
                        self.tracker.set_phase('eval' if str(data.get('mode')).lower() == 'eval' else 'train')
                    self.tracker.track_send(data)
                except Exception:
                    pass
            serialized = pickle.dumps(data)
            self.socket.sendall(len(serialized).to_bytes(4, 'big'))
            self.socket.sendall(serialized)
        except Exception as e:
            print(f"⌠Failed to send data: {e}")
            raise
    
    def receive_data(self):
        try:
            length = int.from_bytes(self.socket.recv(4), 'big')
            data = b''
            while len(data) < length:
                data += self.socket.recv(length - len(data))
            obj = pickle.loads(data)
            if getattr(self, 'tracker', None) is not None:
                try:
                    # If the message carries a mode, set phase accordingly before counting
                    if isinstance(obj, dict) and ('mode' in obj):
                        self.tracker.set_phase('eval' if str(obj.get('mode')).lower() == 'eval' else 'train')
                    self.tracker.track_receive(obj)
                except Exception:
                    pass
            return obj
        except Exception as e:
            print(f"⌠Failed to receive data: {e}")
            raise
    
    def close(self):
        self.socket.close()

def handle_forward_cut_unified(server, kv_model, optimizer, grad_estimator, args, device,
                               pkt: dict, meta: dict, batch_idx: int):
    """
    Unified handler for 'forward_cut' covering all 4 combos:
      (client_opt, server_opt) in {(SGD,SGD), (SGD,ZOO), (ZOO,SGD), (ZOO,ZOO)}.

    Protocol:
      - client sends (h_cut, attention_mask, labels, cut) and meta.need_g_cut.
      - server ALWAYS computes the loss on its half.
      - If server_opt == sgd: loss.backward(); optimizer.step().
      - If server_opt == zoo: do a ZOO step using a loss-only objective.
      - If meta.need_g_cut: return ∂L/∂h_cut (never server param grads).

    True split guarantees:
      - No parameter gradients cross the boundary.
      - Only activations go down; only loss (and optional g_cut) go up.

    Args:
      server: your network wrapper (has send_data/receive_data)
      kv_model: server model wrapper that owns the trainable server prefixes
      optimizer: torch optimizer for server prefixes
      grad_estimator: your ZOO estimator (has estimate_gradients(objective_function=...))
      args: parsed args (must have args.server_opt in {"sgd","zoo"})
      device: torch.device
      pkt: payload from client with keys ['h_cut','attention_mask','labels','cut']
      meta: meta dict, expects meta.get('need_g_cut', bool)
    """
    # Prefer explicit flag; default to legacy behavior (no zoo_eval => need g_cut)
    need_g_cut = bool(meta.get("need_g_cut", not bool(meta.get("zoo_eval", False)))) 
    server_sgd = (not args.use_zeroth_order_server)  # True => SGD, False => ZOO

    # SOLUTION 1: Extract synchronized seed from client metadata for coordinated perturbations
    # This ensures client and server use the same random seed, eliminating cross-terms
    # that cause gradient explosion in split learning ZOO-ZOO
    # Fallback to old seed format for backward compatibility
    sync_seed = meta.get("sync_seed", None)
    if sync_seed is None:
        # Fallback to old independent seed format (for backward compatibility)
        sync_seed = batch_idx * 2029 + getattr(args, "seed", 0)
    # Optional: log sync_seed for early steps
    try:
        _step_val = int(pkt.get("meta", {}).get("step", -1))
    except Exception:
        _step_val = -1
    if _step_val in (0, 10):
        print(f"[SERVER] sync_seed @step {_step_val}: {sync_seed}")

    model_ref = getattr(kv_model, "base_model", kv_model)
    param0 = next(model_ref.parameters())
    target_device = param0.device
    target_dtype  = param0.dtype

    client_data = pkt["data"]
    mode        = pkt.get("mode", "train")   # <-- new, default 'train'
    task_type   = pkt.get("meta", {}).get("task_type", "qa")
    max_new     = pkt.get("meta", {}).get("max_new_tokens", 20)
    tgt_len    = int(client_data.get("tgt_len", 0))

    # --- Pure evaluation mode OR strict ZOO probe: compute MEAN loss only, no updates or gradients ---
    if mode == "eval" or bool(meta.get("zoo_eval", False)):
        # Ensure tracker reflects eval phase for accurate memory/round attribution
        try:
            if server.tracker is not None:
                server.tracker.set_phase('eval')
        except Exception:
            pass
        model_ref = getattr(kv_model, "base_model", kv_model)
        param0 = next(model_ref.parameters())
        target_device = param0.device
        # Match model parameter dtype for eval to keep parity with training
        target_dtype  = param0.dtype

        attn_mask = client_data.get("attention_mask", None)
        labels = client_data.get("labels", None)
        if attn_mask is not None:
            attn_mask = attn_mask.to(device=target_device)
        if labels is not None:
            labels = labels if torch.is_tensor(labels) else torch.as_tensor(labels)
            labels = labels.to(device=target_device)

        cut = int(client_data.get("cut_layer", pkt.get("meta", {}).get("cut_layer", args.cut_layer)))

        from torch.cuda.amp import autocast
        kv_model.eval()  # disable dropout for stability
        with torch.no_grad():
            with autocast(enabled=False):
                h_cut = client_data["h_cut"].to(device=target_device, dtype=target_dtype)
                # Check if logits are requested
                need_logits = bool(meta.get("return_logits", False))
                out = _server_forward_from_cut(
                    kv_model,
                    h_cut=h_cut.detach(),
                    attention_mask=attn_mask,
                    labels=labels,
                    cut=cut,
                    compute_g_cut=False,
                    return_loss_tensor=False,
                    return_logits=need_logits,  # Compute logits if requested
                )

        # Track eval-phase memory after forward
        try:
            if server.tracker is not None:
                server.tracker.set_phase('eval')
                server.tracker.update_memory()
        except Exception:
            pass

        if bool(meta.get("zoo_eval", False)):
            # Strict ZOO probe: return scalar loss only (or logits if requested)
            resp = {
                "type": "loss_report",
                "pert_id": pkt.get("meta", {}).get("pert_id", -1),
                "sign": pkt.get("meta", {}).get("sign", 0),
                "loss": float(out.get("loss", 0.0)),
            }
            # If return_logits is requested, include logits (already computed in out)
            if bool(meta.get("return_logits", False)):
                resp["logits"] = out.get("logits", None)
            server.send_data(resp)
            return
        else:
            reply = {
                "type": "eval_stats",
                "loss": float(out.get("loss", 0.0)),
                "answer_acc": 0.0,
                "f1": 0.0,
                "em": 0.0,
            }
            server.send_data(reply)
            return

    # ---- Unpack incoming tensors ----
    # Expect tensors already on CPU -> move to device (or adapt to your transport)
    h_cut = client_data["h_cut"].to(device=target_device, dtype=target_dtype)
    B, T, H = h_cut.shape
    attn_mask = client_data.get("attention_mask", None)
    labels = client_data.get("labels", None)

    T = h_cut.shape[1]
    if tgt_len and tgt_len != T:
        # Sanity check: trust the client's view if present
        print(f"⚠️ server: adjusting local T={T} -> client T={tgt_len} (will pad gradients if needed)")
        T = tgt_len

    if attn_mask is not None:
        attn_mask = attn_mask.to(device=target_device)
    # labels = pkt.get("labels", None)
    if labels is not None:
        labels = labels if torch.is_tensor(labels) else torch.as_tensor(labels)
        labels = labels.to(device=target_device)
    if attn_mask is not None and labels is not None:
        # Note: h_cut should match the trimmed length from client
        _, attn_mask, labels = right_trim(torch.zeros_like(attn_mask), attn_mask, labels)
    
    if labels is not None and labels.dim() == 2 and labels.shape[1] != T:
        if labels.shape[1] > T:
            labels = labels[:, :T]
        else:
            pad = torch.full((labels.shape[0], T - labels.shape[1]), -100,
                            dtype=labels.dtype, device=labels.device)
            labels = torch.cat([labels, pad], dim=1)
    cut = client_data.get("cut_layer", None)
    if cut is None:
        raise KeyError("Could not determine cut layer: expected pkt['cut'] or pkt['meta']['cut_layer'] or args.cut_layer")
    cut = int(cut)
    
    if server_sgd:
        h_cut = h_cut.detach().requires_grad_(True)
        # allow autograd (for server prefixes + optional boundary grad)
        out = _server_forward_from_cut(
            kv_model, h_cut=h_cut, attention_mask=attn_mask, labels=labels,
            cut=cut, compute_g_cut=False, return_loss_tensor=True
        )
        # Sample memory after forward
        try:
            if server.tracker is not None:
                server.tracker.set_phase('train')
                server.tracker.update_memory()
        except Exception:
            pass
    else:
        # STRICT ZOO: no graph, no boundary-grad inside this forward
        with torch.no_grad():
            h_cut = h_cut.detach()
            out = _server_forward_from_cut(
                kv_model, h_cut=h_cut, attention_mask=attn_mask, labels=labels,
                cut=cut, compute_g_cut=False, return_loss_tensor=True
            )

    loss_tensor = out["loss_tensor"]                 # scalar tensor
    loss_value = float(out["loss"])                  # python float for wire

    # Prepare optional g_cut to send
    g_cut_to_send = None

    # ---- Local server update ----
    if server_sgd:
        # True first-order update of server prefixes
        optimizer.zero_grad(set_to_none=True)
        loss_tensor.backward()  # grads flow into server prefixes (and h_cut.grad if required)
        # Sample memory after backward
        try:
            if server.tracker is not None:
                server.tracker.set_phase('train')
                server.tracker.update_memory()
        except Exception:
            pass
        try:
            if float(getattr(args, 'clip_grad_norm', 0.0)) > 0.0:
                torch.nn.utils.clip_grad_norm_(list(getattr(kv_model, "trainable_parameters", lambda: kv_model.server_kv.parameters())()), max_norm=float(args.clip_grad_norm))
        except Exception:
            pass
        if need_g_cut:
            if h_cut.grad is None:
                raise RuntimeError("need_g_cut=True but h_cut.grad is None. Ensure h_cut.requires_grad_(True).")
            g = h_cut.grad.detach()
            gcut_choice = str(getattr(args, "precision", "fp32")).lower()
            gcut_dtype = (torch.float16 if gcut_choice == "fp16" else torch.float32)
            g_cut_to_send = g.to(dtype=gcut_dtype, device="cpu").contiguous().numpy()
        optimizer.step()
        # Diagnostics: track param delta after 10 SGD server steps
        try:
            global _SERVER_SGD_STEP, _SERVER_SGD_INITIALS
        except Exception:
            pass
        try:
            if _SERVER_SGD_INITIALS is None:
                _SERVER_SGD_INITIALS = {}
                _base_mod = getattr(kv_model, "base_model", kv_model)
                for _name, _p in _base_mod.named_parameters():
                    if _p.requires_grad:
                        _SERVER_SGD_INITIALS[_name] = _p.detach().clone()
        except Exception:
            _SERVER_SGD_INITIALS = None
        try:
            _SERVER_SGD_STEP += 1
        except Exception:
            _SERVER_SGD_STEP = 1
        if _SERVER_SGD_STEP == 10 and _SERVER_SGD_INITIALS:
            try:
                _base_mod = getattr(kv_model, "base_model", kv_model)
                print("[SERVER] Parameter max changes after 10 steps (SGD server):")
                for _name, _p in _base_mod.named_parameters():
                    if _p.requires_grad and _name in _SERVER_SGD_INITIALS:
                        _delta = (_p - _SERVER_SGD_INITIALS[_name]).abs().max().item()
                        print(f"{_name}: max param change = {_delta:.6e}")
            except Exception:
                pass
        # Sample memory after optimizer step
        try:
            if server.tracker is not None:
                server.tracker.set_phase('train')
                server.tracker.update_memory()
        except Exception:
            pass

        # clean leaf grad
        if h_cut.grad is not None:
            h_cut.grad = None

    else:
        # Zeroth order update of server prefixes (loss-only objective)
        # If client needs g_cut, compute it via finite-difference on h_cut
        if need_g_cut:
            # WARNING: This uses activation perturbations which cause exponential amplification
            # For stable training, use parameter perturbations instead (default: zoo_use_gcut=False)
            print("WARNING: Using activation perturbation for g_cut computation. This may cause gradient explosion.")
            print("  Consider disabling zoo_use_gcut for more stable training with parameter perturbations only.")
            # CRITICAL: Keep model in training mode during finite-difference estimation
            # This ensures consistent behavior with the server ZOO objective evaluation.
            # Even though we're estimating boundary gradients (not optimizing server params),
            # maintaining consistent model state (dropout, BatchNorm) is important for
            # proper gradient flow and training dynamics.
            # Note: The theoretical mismatch between ZO gradients (server) and exact
            # boundary gradients (client) is inherent to split learning with ZOO, but
            # mode consistency helps mitigate practical inconsistencies.
            g = _estimate_g_cut_fd(
                kv_model,
                h_cut.to(dtype=target_dtype, device=target_device),
                attn_mask, labels, cut,
                mu=getattr(args, "mu_gcut", getattr(args, "mu", 1e-3)),
                num_pert=getattr(args, "num_pert_gcut", getattr(args, "num_pert", 10)),
                mode=str(getattr(args, 'estimator', 'central')),
            )
            gcut_choice = str(getattr(args, "precision", "fp32")).lower()
            gcut_dtype = (torch.float16 if gcut_choice == "fp16" else torch.float32)
            g_cut_to_send = g.to(dtype=gcut_dtype, device="cpu").contiguous().numpy()

        # Update train-phase memory after optional g_cut probe
        try:
            if server.tracker is not None:
                server.tracker.set_phase('train')
                server.tracker.update_memory()
        except Exception:
            pass

    # else:
    #     # Zeroth order update of server prefixes (loss-only objective)
    #     # If client needs g_cut, compute it via autograd.grad wrt h_cut (no server param grads cross boundary)
    #     if need_g_cut:
    #         g = _estimate_g_cut_fd(
    #             kv_model, h_cut, attn_mask, labels, cut,
    #             mu=getattr(args, "mu_gcut", getattr(args, "mu", 1e-3)),
    #             num_pert=getattr(args, "num_pert_gcut", getattr(args, "num_pert", 8)),
    #         )
    #         g_cut_to_send = g.to(dtype=torch.float32, device="cpu").contiguous().numpy()

        # ========================================================================
        # THEORETICAL CONSIDERATIONS: Split Learning with Zero-Order Optimization
        # ========================================================================
        # Split learning with ZOO creates a unique challenge not present in standard MeZO:
        #
        # 1. GRADIENT BOUNDARY COMPUTATION:
        #    - Server uses ZO gradients (estimated via finite differences)
        #    - Client may need exact boundary gradients (via autograd or FD)
        #    - This mixing of ZO and exact gradients at the split boundary creates
        #      a theoretical mismatch: ZO gradients are stochastic approximations,
        #      while exact gradients are deterministic (modulo numerical precision)
        #
        # 2. ASYMMETRIC OPTIMIZATION:
        #    - Client and server can use different optimizers (e.g., ZOO-ZOO, ZOO-SGD)
        #    - Each side optimizes wrt different parameters with different gradient
        #      qualities, potentially leading to optimization dynamics that differ
        #      from standard end-to-end training
        #
        # 3. MODE CONSISTENCY:
        #    - CRITICAL: Model must remain in training mode during perturbation
        #      evaluation to ensure dropout, BatchNorm, etc. behave consistently
        #    - Using eval() mode would cause inconsistent behavior between
        #      perturbation evaluations and actual training steps
        #
        # 4. COMMUNICATION OVERHEAD:
        #    - Passing activations/gradients between split nodes introduces
        #      communication costs and potential synchronization issues
        #
        # Despite these theoretical challenges, maintaining mode consistency and
        # proper gradient estimation helps ensure practical convergence behavior.
        # ========================================================================

        # Define a pure loss-only objective for ZOO (no gradients, just returns float loss)
        # def server_objective(_x, _y):
        #     with torch.no_grad():
        #         o = _server_forward_from_cut(
        #             kv_model,
        #             h_cut=client_data["h_cut"].to(device).detach(),
        #             attention_mask=attn_mask,
        #             labels=labels, cut=cut,
        #             compute_g_cut=False,
        #             return_loss_tensor=False
        #         )
        #         return torch.tensor(float(o["loss"]), device="cpu")

        def server_objective(_x, _y):
            # CRITICAL: Keep model in training mode during perturbation evaluation
            # This ensures consistent behavior of dropout, BatchNorm, and other training-time
            # behaviors across all forward passes. Using eval() mode would cause:
            # - Dropout to be disabled (inconsistent with training)
            # - BatchNorm to use running stats instead of batch stats (inconsistent)
            # - Potential loss computation differences
            # Even though we use torch.no_grad() to avoid building computational graph,
            # the model's internal state (dropout masks, BN statistics) must match training mode.
            # Note: This creates a theoretical mismatch when mixing ZO gradients (server)
            # with exact boundary gradients (client), but maintaining mode consistency
            # is crucial for proper optimization dynamics.
            with torch.no_grad():
                o = _server_forward_from_cut(
                    kv_model,
                    h_cut=client_data["h_cut"].to(device=device, dtype=target_dtype),
                    attention_mask=attn_mask,
                    labels=labels, cut=cut,
                    compute_g_cut=False,
                    return_loss_tensor=False
                )
                return torch.tensor(float(o["loss"]), device="cpu")

        _dummy_x = torch.zeros(1)
        _dummy_y = torch.zeros(1)

        # Choose mu scaling method for server ZOO
        base_mu = float(getattr(args, 'mu', 1e-3))
        mu_scaling = str(getattr(args, 'zoo_mu_scaling', 'fixed')).lower()

        if mu_scaling == 'fixed':
            # Fixed mu (like MeZO) - use base_mu directly
            scaled_mu = base_mu
        else:  # mu_scaling == 'rms'
            # RMS scale μ per step (current implementation)
            with torch.no_grad():
                squares, count = 0.0, 0
                for p in list(getattr(kv_model, "trainable_parameters", lambda: kv_model.server_kv.parameters())()):
                    squares += (p.data.float()**2).sum().item()
                    count   += int(p.numel())
                rms = (squares / max(1, count))**0.5
            scaled_mu = max(1e-5, base_mu) * (rms if rms > 0 else 1.0)

        old_mu = getattr(grad_estimator, 'perturbation_scale', scaled_mu)
        grad_estimator.perturbation_scale = scaled_mu

        # Run your estimator (MeZO-style), then step
        optimizer.zero_grad(set_to_none=True)
        grad_estimator.model_params = list(getattr(kv_model, "trainable_parameters", lambda: kv_model.server_kv.parameters())())
        try:
            # SOLUTION 1: Use synchronized seed from client for server perturbations
            # This ensures client and server perturbations are correlated, eliminating cross-terms
            grad_estimator.estimate_gradients(_dummy_x, _dummy_y, server_objective,
                                                random_seed=sync_seed)
        except TypeError:
            # Fallback if your class has the old signature
            grad_estimator.estimate_gradients(torch.zeros(1), torch.zeros(1), server_objective,
                                                random_seed=sync_seed)
        finally:
            grad_estimator.perturbation_scale = old_mu
        # Track memory after gradient estimation (train phase)
        try:
            if server.tracker is not None:
                server.tracker.set_phase('train')
                server.tracker.update_memory()
        except Exception:
            pass
        # Compute server grad L2 norm before step for diagnostics
        try:
            _g2 = 0.0
            for _p in server_params:
                if _p.grad is not None:
                    _gn = _p.grad.detach().data.float().norm().item()
                    _g2 += (_gn * _gn)
            server_grad_norm = float(_g2 ** 0.5)
        except Exception:
            server_grad_norm = 0.0

        optimizer.step()
        try:
            torch.nn.utils.clip_grad_norm_(list(getattr(kv_model, "trainable_parameters", lambda: kv_model.server_kv.parameters())()), max_norm=1.0)
        except Exception:
            pass
        # optimizer.step()
        # Track memory after optimizer step (train phase)
        try:
            if server.tracker is not None:
                server.tracker.set_phase('train')
                server.tracker.update_memory()
        except Exception:
            pass

    # ---- Reply to client ----
    reply = {
        "type": "zoo_phase_complete",
        "grad_norm": float(server_grad_norm if 'server_grad_norm' in locals() else 0.0),
        "loss_diff_avg": 0.0,
        "loss_diff_std": 0.0,
        "phase2_id": phase2_id,
        "loss": float(loss_tensor.item()),
    }
    if g_cut_to_send is not None:
        # sanity: shape must match h_cut
        # (can't use tensors in dict over pickle reliably -> send numpy)
        # Choose outbound dtype according to --precision
        _is_fp16 = (str(getattr(args, 'precision', 'fp32')).lower() == 'fp16')
        if isinstance(g_cut_to_send, torch.Tensor):
            gc = g_cut_to_send.detach().to(torch.float16 if _is_fp16 else torch.float32, device="cpu").contiguous().numpy()
        else:
            gc = np.asarray(g_cut_to_send, dtype=(np.float16 if _is_fp16 else np.float32))
        reply["g_cut"] = gc
    try:
        print(f"[server-phase2] complete step={meta.get('step')} phase2_id={phase2_id} grad_norm={reply.get('grad_norm'):.4f}")
    except Exception:
        pass
    server.send_data(reply)

def parse_args():
    parser = argparse.ArgumentParser(description='Split Learning LLM server')
    parser.add_argument('--model_name', type=str, default='facebook/opt-125m', help='Pretrained model name')
    parser.add_argument('--num_prefix', type=int, default=20, help='Number of prefix tokens')
    parser.add_argument('--cut_layer', type=int, default=1, help='Split index: 0..cut-1 client; cut..L-1 server')
    parser.add_argument('--max_length', type=int, default=512, help='Maximum sequence length')
    parser.add_argument('--seed', type=int, default=42, help='Random seed')
    parser.add_argument('--lr', type=float, default=1e-3, help='Learning rate')
    parser.add_argument('--zoo_lr', type=float, default=1e-5, help='ZOO learning rate')
    parser.add_argument('--momentum', type=float, default=0.9, help='SGD momentum')
    parser.add_argument('--weight_decay', type=float, default=0.0, help='Weight decay for optimizer')
    parser.add_argument('--train_batch_size', type=int, default=4, help='Training batch size')
    parser.add_argument('--test_batch_size', type=int, default=4, help='Test batch size')
    parser.add_argument('--max_steps', type=int, default=4000, help='Maximum training steps')
    parser.add_argument('--epochs', type=int, default=10, help='Number of epochs (fallback)')
    # ZOO parameters
    parser.add_argument('--mu', type=float, default=1e-3, help='ZOO perturbation scale')
    parser.add_argument('--num_pert', type=int, default=1, help='Number of ZOO perturbations')
    parser.add_argument('--estimator', type=str, choices=['central','forward'], default='central', help='Finite-diff estimator type for ZOO/g_cut')
    parser.add_argument('--zoo_perturbation_type', type=str, choices=['rademacher','bernoulli','gaussian'], default='gaussian', help='Perturbation distribution for ZOO (default gaussian=N(0,1))')
    parser.add_argument('--zoo_mu_scaling', type=str, choices=['fixed','rms'], default='fixed', help='Mu scaling method for ZOO (fixed=use mu directly like MeZO, rms=scale by parameter RMS)')
    parser.add_argument('--zoo_max_grad_value', type=float, default=0.0, help='Maximum gradient norm threshold for adaptive clipping in ZOO estimator (0.0 disables clipping)')
    parser.add_argument('--clip_grad_norm', type=float, default=0.0, help='Max global norm for server gradients (0 disables)')
    parser.add_argument('--use_zeroth_order_server', action='store_true', help='Use ZOO for server')
    # Enable sequential two-phase ZOO updates (client then server)
    parser.add_argument('--sequential_zoo', action='store_true',
                        help='Use sequential ZOO updates (Phase 1: client update with server fixed; Phase 2: server update with client fixed)')
    # Classification calibration/control (SST-2): optional CE label smoothing and class weights
    parser.add_argument('--label_smoothing', type=float, default=0.0, help='Label smoothing for CE on server (0 disables)')
    parser.add_argument('--class_weight_tokens', type=str, default=None, help='Comma-separated token:weight pairs for CE class weighting on answer tokens (e.g., "terrible:1.5,great:1.0")')
    parser.add_argument('--class_weight_ids', type=str, default=None, help='Comma-separated id:weight pairs for CE class weighting (e.g., "1234:1.5,5678:1.0")')
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
    parser.add_argument('--host', type=str, default='127.0.0.1', help='client host to connect')
    parser.add_argument('--port', type=int, default=12345, help='client port to connect')
    
    # unified precision flag
    parser.add_argument('--precision', choices=['fp16','fp32'], default='fp32', help='Global precision for compute and wire payloads')

    args = parser.parse_args()
    return args


def _assert_only_expected_trainables(module: nn.Module, mode: str, layer_range=None, side: str = None):
    for n, p in module.named_parameters():
        if mode == "prefix":
            if side == "server":
                is_allowed = n.startswith("server_kv.")
            elif side == "client":
                is_allowed = n.startswith("kv.")
            else:
                is_allowed = (n.startswith("server_kv.") or n.startswith("kv.") or ("prefix" in n))

            ok = ("lora_A" not in n and "lora_B" not in n) and ((is_allowed) == p.requires_grad)

        elif mode == "lora":
            is_lora = ("lora_A" in n) or ("lora_B" in n)
            ok = (is_lora == p.requires_grad) and ("kv." not in n) and ("server_kv." not in n)

        elif mode == "full":
            # server side full-FT should only train decoder layers [cut..L-1], final_layer_norm and lm_head
            cut = int(getattr(module, "cut_layer", 0))
            def _server_allowed(name: str) -> bool:
                core = name.split("base_model.", 1)[1] if "base_model." in name else name
                if core.startswith("model.decoder.final_layer_norm"):
                    return True
                if core.startswith("lm_head"):
                    return True
                if ".model.decoder.layers." in ("." + core):
                    try:
                        idx = int(core.split("model.decoder.layers.")[1].split(".")[0])
                        return idx >= cut
                    except Exception:
                        return False
                return False
            def _forbidden(name: str) -> bool:
                return (
                    name.startswith("kv.") or name.startswith("server_kv.") or ("prefix" in name)
                    or ("lora_A" in name) or ("lora_B" in name)
                )
            if side == "server":
                if p.requires_grad:
                    ok = (not _forbidden(n)) and _server_allowed(n)
                else:
                    ok = True
            else:
                ok = not (_forbidden(n) and p.requires_grad)

        else:  # none
            ok = (p.requires_grad is False)

        assert ok, f"Unexpected trainable param in {mode} mode{f' ({side})' if side else ''}: {n} requires_grad={p.requires_grad}"


if __name__ == "__main__":
    args = parse_args()
    
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    
    print("=" * 60)
    print("SPLIT LEARNING server WITH PREFIX TUNING")
    print("=" * 60)
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    print(f"Loading tokenizer: {args.model_name}")
    tokenizer = safe_get_hf_tokenizer(args.model_name)
    print("Tokenizer loaded successfully")
    
    print("Creating server model...")
    # Resolve compute dtype from unified precision
    _compute_dtype = (torch.float16 if str(getattr(args, 'precision', 'fp32')).lower() == 'fp16' else torch.float32)
    # KV prefix model (new): always create so handlers can access
    tmp_cfg = AutoModelForCausalLM.from_pretrained(args.model_name, torch_dtype=_compute_dtype, device_map=None).config
    # Configure optional CE controls
    try:
        _LABEL_SMOOTHING = max(0.0, float(getattr(args, 'label_smoothing', 0.0)))
        wt_tokens = getattr(args, 'class_weight_tokens', None)
        wt_ids = getattr(args, 'class_weight_ids', None)
        if (wt_tokens and wt_tokens.strip()) or (wt_ids and wt_ids.strip()):
            vocab_size = int(getattr(tmp_cfg, 'vocab_size', 0) or getattr(tokenizer, 'vocab_size', 0))
            if vocab_size and vocab_size > 0:
                _CE_CLASS_WEIGHTS = _build_class_weight_vector(tokenizer, vocab_size, wt_tokens, wt_ids)
                try:
                    num_mod = int((_CE_CLASS_WEIGHTS != 1.0).sum().item()) if torch.is_tensor(_CE_CLASS_WEIGHTS) else 0
                except Exception:
                    num_mod = 0
                print(f"CE weighting enabled: {num_mod} vocab ids adjusted; label_smoothing={_LABEL_SMOOTHING}")
            else:
                print("⚠️ Could not determine vocab size; CE class weighting disabled")
                _CE_CLASS_WEIGHTS = None
        else:
            _CE_CLASS_WEIGHTS = None
    except Exception as _ce_e:
        print(f"⚠️ CE controls setup failed: {_ce_e}")
        _CE_CLASS_WEIGHTS = None
    total_layers = tmp_cfg.num_hidden_layers
    if args.tuning == 'lora':
        kv_model = LoRAserverModel(
            args.model_name, total_layers, args.cut_layer,
            r=args.lora_r, alpha=args.lora_alpha, dropout=args.lora_dropout,
            targets=tuple(args.lora_targets.split(',')),
            torch_dtype=_compute_dtype
        ).to(device)
        trainable_params = list(kv_model.trainable_parameters())
        _assert_only_expected_trainables(kv_model, args.tuning, side="server")

        print(f"server owns layers [{args.cut_layer}, {total_layers-1}] with LoRA r={args.lora_r}, alpha={args.lora_alpha}")
    elif args.tuning == 'full':
        kv_model = FullserverModel(args.model_name, total_layers, args.cut_layer, torch_dtype=_compute_dtype).to(device)
        trainable_params = list(kv_model.trainable_parameters())
        _assert_only_expected_trainables(kv_model, args.tuning, side="server")
        print(f"server full-FT layers [{args.cut_layer}, {total_layers-1}] enabled; final_layer_norm/lm_head trainable")
    else:
        kv_model = serverKVModel(args.model_name, total_layers, args.cut_layer, num_prefix=args.num_prefix, torch_dtype=_compute_dtype).to(device)
        for p in getattr(kv_model, "base_model", kv_model).parameters():
            p.requires_grad = False
        for p in kv_model.server_kv.parameters():
            p.requires_grad = True
        for p in kv_model.client_kv_mirror.parameters():
            p.requires_grad = False

        trainable_params = list(kv_model.server_kv.parameters())
        _assert_only_expected_trainables(kv_model, args.tuning, side="server")

        print(f"server owns layers [{args.cut_layer}, {total_layers-1}] with {args.num_prefix} prefix tokens each")

    
    print(f"Model loaded: {args.model_name}")
    print(f"server owns layers [{args.cut_layer}, {total_layers-1}] with {args.num_prefix} prefix tokens each")
    
    if args.use_zeroth_order_server:
        # server ZOO: SGD as a simple applicator when estimator populates grads
        optimizer = optim.SGD(
            list(getattr(kv_model, "trainable_parameters", lambda: kv_model.server_kv.parameters())()),
            lr=args.zoo_lr,
            momentum=0.0,
            weight_decay=args.weight_decay,
        )
        print(f"server using ZOO optimizer with lr={args.zoo_lr}, weight_decay={args.weight_decay}")
    else:
        trainable = list(getattr(kv_model, "trainable_parameters", lambda: kv_model.server_kv.parameters())())
        optimizer = optim.SGD(
            trainable,
            lr=args.lr,
            momentum=args.momentum,
            weight_decay=args.weight_decay,
        )
        print(f"server using SGD optimizer with lr={args.lr}, momentum={args.momentum}, wd={args.weight_decay}")
    
    # Setup ZOO gradient estimator if needed
    grad_estimator = None
    if args.use_zeroth_order_server:
        print("Setting up ZOO gradient estimator...")
        grad_estimator = StochasticGradientApproximator(
            model_params=trainable_params,
            perturbation_scale=args.mu,
            sample_count=args.num_pert,
            compute_device=device,
            data_type=(torch.float16 if str(getattr(args, 'precision', 'fp32')).lower() == 'fp16' else torch.float32),
            perturbation_type=str(getattr(args, 'zoo_perturbation_type', 'gaussian')),
            max_grad_value=float(getattr(args, 'zoo_max_grad_value', 500.0))
        )
        print(f"ZOO gradient estimator created with mu={args.mu}, num_pert={args.num_pert}")
    
    try:
        print("=" * 60)
        print("server STARTING - ATTEMPTING TO CONNECT TO client")
        print("=" * 60)
        print(f"Trying to connect to client at {args.host}:{args.port}...")
        
        server = server(args.host, args.port)
        # Attach compute tracker for server process
        try:
            _cli_tracker = ComputeTracker(device=device, role="server")
        except Exception:
            _cli_tracker = None
        server.tracker = _cli_tracker
        
        print("=" * 60)
        print("server SUCCESSFULLY CONNECTED TO client!")
        print("=" * 60)

        server_config = {
            'model_name': args.model_name,
            'num_prefix': args.num_prefix,
            'lr': args.lr,
            # True => server uses SGD; False => server uses ZOO
            'server_sgd': (not args.use_zeroth_order_server),
            'zoo_lr': getattr(args, "zoo_lr", None),
            'mu': getattr(args, "mu", None),
            'num_pert': getattr(args, "num_pert", None),
            'server_tuning': args.tuning,
            'lora': {
                'r': args.lora_r,
                'alpha': args.lora_alpha,
                'dropout': args.lora_dropout,
                'targets': args.lora_targets,
            }
        }
        server.send_data(server_config)
        print(f"Sent server configuration")

        print("=" * 60)
        if args.use_zeroth_order_server:
            print("STARTING ZOO TRAINING REQUEST HANDLER...")
            print(f"  server prefixes WILL be trained using ZOO")
        else:
            print("STARTING SGD TRAINING REQUEST HANDLER...")
            print(f"  server prefixes WILL be trained using SGD")
        print("=" * 60)

        batch_idx = 0
        while True:
            msg = server.receive_data()
            if msg is None:
                print("No message received (client closed?). Exiting loop.")
                break

            mtype = msg.get("type", "")
            if mtype == "get_server_kv_state":
                try:
                    if _cli_tracker is not None:
                        _cli_tracker.set_phase('eval')  # state fetch occurs during eval context
                except Exception:
                    pass
                # In LoRA mode there are no server prefixes; return empty state
                if args.tuning == 'lora' or not hasattr(kv_model, 'server_kv'):
                    server.send_data({"type": "server_kv_state", "state": {"k": None, "v": None}})
                else:
                    state = {
                        "k": kv_model.server_kv.k.detach().cpu(),
                        "v": kv_model.server_kv.v.detach().cpu(),
                    }
                    server.send_data({"type": "server_kv_state", "state": state})
                continue

            elif mtype == "get_server_prefix_snapshot":
                try:
                    if _cli_tracker is not None:
                        _cli_tracker.set_phase('eval')
                except Exception:
                    pass
                with torch.no_grad():
                    if hasattr(kv_model, "server_kv") and getattr(kv_model.server_kv, "k", None) is not None:
                        kc = kv_model.server_kv.k.detach().cpu()
                        vc = kv_model.server_kv.v.detach().cpu()
                        server.send_data({"type": "server_prefix_snapshot", "ok": True, "k": kc, "v": vc})
                    else:
                        server.send_data({"type": "server_prefix_snapshot", "ok": True, "k": None, "v": None})
                continue

            elif mtype == "training_complete":
                print("TRAINING COMPLETED - server SHUTTING DOWN")
                break

            elif mtype == "get_server_lora_state":
                try:
                    if _cli_tracker is not None:
                        _cli_tracker.set_phase('eval')
                except Exception:
                    pass
                try:
                    # Return LoRA A/B only for server half [cut..L-1]
                    if args.tuning == 'lora':
                        state = get_lora_state_dict(
                            getattr(kv_model, "base_model", kv_model),
                            layer_range=(args.cut_layer, total_layers-1)
                        )
                        server.send_data({"type": "server_lora_state", "ok": True, "state": state})
                    else:
                        server.send_data({"type": "server_lora_state", "ok": True, "state": {}})
                except Exception as e:
                    server.send_data({"type": "server_lora_state", "ok": False, "error": str(e)})
                continue

            elif mtype == "get_server_full_state":
                try:
                    if _cli_tracker is not None:
                        _cli_tracker.set_phase('eval')
                except Exception:
                    pass
                try:
                    if args.tuning == 'full':
                        base = getattr(kv_model, "base_model", kv_model)
                        sd = base.state_dict()
                        # Filter to server half: decoder layers [cut..L-1], final_layer_norm, lm_head
                        filt = {}
                        for k, v in sd.items():
                            take = False
                            if k.startswith('model.decoder.final_layer_norm'):
                                take = True
                            elif k.startswith('lm_head'):
                                take = True
                            elif '.model.decoder.layers.' in ('.' + k):
                                try:
                                    after = k.split('model.decoder.layers.')[1]
                                    idx = int(after.split('.')[0])
                                    if idx >= int(args.cut_layer):
                                        take = True
                                except Exception:
                                    pass
                            if take:
                                filt[k] = v.detach().cpu()
                        server.send_data({"type": "server_full_state", "ok": True, "state": filt})
                    else:
                        server.send_data({"type": "server_full_state", "ok": True, "state": {}})
                except Exception as e:
                    server.send_data({"type": "server_full_state", "ok": False, "error": str(e)})
                continue

            if mtype == "zoo_server_update":
                try:
                    _meta = msg.get("meta", {})
                    print(f"[server-phase2] dispatch zoo_server_update step={_meta.get('step')} phase2_id={_meta.get('phase2_id')}")
                except Exception:
                    pass
                handle_zoo_server_update(
                    server=server,
                    kv_model=kv_model,
                    optimizer=optimizer,
                    grad_estimator=grad_estimator,
                    args=args,
                    device=device,
                    pkt=msg,
                )
                batch_idx += 1
                continue

            if mtype == "forward_cut":
                # Training path: mark train phase
                try:
                    if _cli_tracker is not None:
                        _cli_tracker.set_phase('train')
                except Exception:
                    pass
                data = msg.get("data")
                if data is None or not isinstance(data, dict):
                    print("⌠server: malformed forward_cut (missing 'data'); ignoring this packet.")
                    # Optionally: server.send_data({"type":"error","reason":"missing_data_in_forward_cut"})
                    continue
                handle_forward_cut_unified(
                    server=server,
                    kv_model=kv_model,
                    optimizer=optimizer,
                    grad_estimator=grad_estimator,
                    args=args,
                    device=device,
                    pkt=msg,
                    meta=msg.get("meta", {}),
                    batch_idx=batch_idx
                )
                batch_idx += 1

            elif mtype in ("training_complete", "shutdown"):
                print(f"Received '{mtype}' from client. Exiting.")
                break

            else:
                # You can log/ignore other control messages here
                print(f"Unknown message type: {mtype}")
        
    except ConnectionRefusedError:
        print("=" * 60)
        print("⌠CONNECTION FAILED!")
        print("=" * 60)
        print(f"Could not connect to client at {args.host}:{args.port}")
        print("Make sure the client is running first:")
        print(f"   python client.py --model_name {args.model_name}")
        print("=" * 60)
    except Exception as e:
        print(f"⌠server ERROR: {e}")
        traceback.print_exc()
    finally:
        try:
            server.close()
        except:
            pass
        # Print and persist server compute/communication summary
        try:
            if '_cli_tracker' in locals() and _cli_tracker is not None:
                _ = _cli_tracker.update_memory()
                _ = _cli_tracker.print_summary()
                try:
                    _cli_tracker.save_stats('server_stats_squad.json')
                except Exception:
                    pass
        except Exception:
            pass
        print("server SHUTDOWN COMPLETE")
        