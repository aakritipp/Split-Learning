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
from coordinated_zoo import (
    CoordinatedZOOTrainer,
    coordinated_zoo_step_server,
    CLIENT_OFFSET_DEFAULT,
)
import types
from prefix_kv import (
    PrefixKV, 
    merge_past_key_values, 
    flatten_grad_state
)
# Global KV model holder (set in main)
kv_model = None
coordinated_trainer = None

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
def _estimate_g_cut_fd(kv_model, h_cut, attention_mask, labels, cut, mu=1e-3, num_pert=8, mode: str = 'central'):
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
    return_loss_tensor: bool = False,):
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
        loss = F.cross_entropy(
            shift_logits.view(-1, shift_logits.size(-1)),
            shift_labels.view(-1),
            ignore_index=-100,
            reduction="mean",)
    else:
        loss = torch.zeros((), device=device, dtype=dtype)

    out = {"loss": float(loss.item())}
    if compute_g_cut:
        g_cut, = torch.autograd.grad(loss, h, retain_graph=return_loss_tensor)
        out["g_cut"] = g_cut.detach().to(h.dtype).cpu()
    
    if return_loss_tensor:
        out["loss_tensor"] = loss  # Return the torch tensor for local update
    
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
        global coordinated_trainer
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
        coord_enabled = bool(meta.get("coordinated_zoo", False)) and coordinated_trainer is not None
        coord_seed = int(meta.get("coord_seed", 0)) if coord_enabled else 0
        coord_sign = int(meta.get("coord_sign", 0)) if coord_enabled else 0
        coord_mu = None
        coord_estimator = None
        if coord_enabled:
            coord_mu = float(meta.get("coord_mu", getattr(coordinated_trainer, "mu", getattr(args, "mu", 1e-3))))
            coord_estimator = meta.get("coord_estimator", getattr(args, "estimator", "central"))
            if coord_sign != 0:
                coordinated_trainer.apply_perturbation(seed=coord_seed, sign=coord_sign, mu=coord_mu)

        out = {"loss": 0.0}
        try:
            with torch.no_grad():
                with autocast(enabled=False):
                    h_cut = client_data["h_cut"].to(device=target_device, dtype=target_dtype)
                    out = _server_forward_from_cut(
                        kv_model,
                        h_cut=h_cut.detach(),
                        attention_mask=attn_mask,
                        labels=labels,
                        cut=cut,
                        compute_g_cut=False,
                        return_loss_tensor=False,
                    )
        finally:
            if coord_enabled and coord_sign != 0:
                coordinated_trainer.apply_perturbation(seed=coord_seed, sign=-coord_sign, mu=coord_mu)

        if coord_enabled:
            try:
                coordinated_trainer.record_result(
                    seed=coord_seed,
                    sign=coord_sign,
                    loss=float(out.get("loss", 0.0)),
                    mu=coord_mu,
                    estimator=coord_estimator,
                )
            except Exception as _ce:
                print(f"⚠️ Coordinated ZOO recorder failure: {_ce}")

        # Track eval-phase memory after forward
        try:
            if server.tracker is not None:
                server.tracker.set_phase('eval')
                server.tracker.update_memory()
        except Exception:
            pass

        if bool(meta.get("zoo_eval", False)):
            # Strict ZOO probe: return scalar loss only
            server.send_data({
                "type": "loss_report",
                "pert_id": pkt.get("meta", {}).get("pert_id", -1),
                "sign": pkt.get("meta", {}).get("sign", 0),
                "loss": float(out.get("loss", 0.0)),
            })
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
            torch.nn.utils.clip_grad_norm_(list(getattr(kv_model, "trainable_parameters", lambda: kv_model.server_kv.parameters())()), max_norm=float(args.clip_grad_norm))
        except Exception:
            pass
        if need_g_cut:
            if h_cut.grad is None:
                raise RuntimeError("need_g_cut=True but h_cut.grad is None. Ensure h_cut.requires_grad_(True).")
            g = h_cut.grad.detach()
            precision_choice = str(getattr(args, "precision", "fp32")).lower()
            gcut_dtype = (torch.float16 if precision_choice == "fp16" else torch.float32)
            g_cut_to_send = g.to(dtype=gcut_dtype, device="cpu").contiguous().numpy()
        optimizer.step()
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
            was_training = kv_model.training
            kv_model.eval()
            try:
                g = _estimate_g_cut_fd(
                    kv_model,
                    h_cut.to(dtype=target_dtype, device=target_device),
                    attn_mask, labels, cut,
                    mu=getattr(args, "mu_gcut", getattr(args, "mu", 1e-2)),
                    num_pert=getattr(args, "num_pert_gcut", getattr(args, "num_pert", 16)),
                    mode=str(getattr(args, 'estimator', 'central')),
                )
            finally:
                if was_training:
                    kv_model.train()
            
            precision_choice = str(getattr(args, "precision", "fp32")).lower()
            gcut_dtype = (torch.float16 if precision_choice == "fp16" else torch.float32)
            g_cut_to_send = g.to(dtype=gcut_dtype, device="cpu").contiguous().numpy()

        # Update train-phase memory after optional g_cut probe
        try:
            if server.tracker is not None:
                server.tracker.set_phase('train')
                server.tracker.update_memory()
        except Exception:
            pass
        # ============ SEQUENTIAL ZOO-ZOO FIX (server) ============
        server_turn = bool(pkt.get("meta", {}).get("server_turn", True))
        if not server_turn:
            print(f"[SEQ-ZOO] Batch {batch_idx}: Server frozen (client's turn)")
            # Reply with loss (and g_cut if requested), but DO NOT update server params
            reply = {"loss": float(loss_tensor.item())}
            if g_cut_to_send is not None:
                reply["g_cut"] = g_cut_to_send
            server.send_data(reply)
            return
        else:
            print(f"[SEQ-ZOO] Batch {batch_idx}: Server updating")
        # ============ SEQUENTIAL ZOO-ZOO FIX (server) ============

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
            was_training = kv_model.training
            kv_model.eval()
            try:
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
            finally:
                if was_training:
                    kv_model.train()

        _dummy_x = torch.zeros(1)
        _dummy_y = torch.zeros(1)
        # Run your estimator (MeZO-style), then step
        optimizer.zero_grad(set_to_none=True)
        if grad_estimator is not None:
            grad_estimator.model_params = list(getattr(kv_model, "trainable_parameters", lambda: kv_model.server_kv.parameters())())
            try:
                grad_estimator.estimate_gradients(_dummy_x, _dummy_y, server_objective,
                                                    random_seed=batch_idx * 2029 + getattr(args, "seed", 0))
            except TypeError:
                # Fallback if your class has the old signature
                grad_estimator.estimate_gradients(torch.zeros(1), torch.zeros(1), server_objective,
                                                    random_seed=batch_idx * 2029 + getattr(args, "seed", 0))
        # Track memory after gradient estimation (train phase)
        try:
            if server.tracker is not None:
                server.tracker.set_phase('train')
                server.tracker.update_memory()
        except Exception:
            pass
        optimizer.step()
        try:
            if float(getattr(args, "clip_grad_norm", 0.0)) > 0.0:
                torch.nn.utils.clip_grad_norm_(list(getattr(kv_model, "trainable_parameters", lambda: kv_model.server_kv.parameters())()), max_norm=float(args.clip_grad_norm))
        except Exception:
            pass
        # NOTE [Phase 0]: Duplicate optimizer.step() intentionally commented to avoid double stepping.
        # optimizer.step()  # FIXED: Commented out duplicate step
        # Track memory after optimizer step (train phase)
        try:
            if server.tracker is not None:
                server.tracker.set_phase('train')
                server.tracker.update_memory()
        except Exception:
            pass

    # ---- Reply to client ----
    reply = {"loss": float(loss_tensor.item())}
    if g_cut_to_send is not None:
        # sanity: shape must match h_cut
        # (can't use tensors in dict over pickle reliably -> send numpy)
        reply["g_cut"] = g_cut_to_send
    server.send_data(reply)

def parse_args():
    parser = argparse.ArgumentParser(description='Split Learning LLM server')
    parser.add_argument('--model_name', type=str, default='facebook/opt-125m', help='Pretrained model name')
    parser.add_argument('--num_prefix', type=int, default=20, help='Number of prefix tokens')
    parser.add_argument('--cut_layer', type=int, default=1, help='Split index: 0..cut-1 client; cut..L-1 server')
    parser.add_argument('--max_length', type=int, default=512, help='Maximum sequence length')
    parser.add_argument('--seed', type=int, default=42, help='Random seed')
    parser.add_argument('--lr', type=float, default=1e-3, help='Learning rate')
    parser.add_argument('--zoo_lr', type=float, default=1e-4, help='ZOO learning rate')
    parser.add_argument('--momentum', type=float, default=0.0, help='SGD momentum (0 disables)')
    parser.add_argument('--weight_decay', type=float, default=0.0, help='Weight decay for optimizer')
    parser.add_argument('--train_batch_size', type=int, default=4, help='Training batch size')
    parser.add_argument('--test_batch_size', type=int, default=4, help='Test batch size')
    parser.add_argument('--max_steps', type=int, default=4000, help='Maximum training steps')
    parser.add_argument('--epochs', type=int, default=10, help='Number of epochs (fallback)')
    # ZOO parameters
    parser.add_argument('--mu', type=float, default=1e-3, help='ZOO perturbation scale')
    parser.add_argument('--num_pert', type=int, default=64, help='Number of ZOO perturbations')
    parser.add_argument('--estimator', type=str, choices=['central','forward'], default='central', help='Finite-diff estimator type for ZOO/g_cut')
    parser.add_argument('--use_zeroth_order_server', action='store_true', help='Use ZOO for server')
    parser.add_argument('--coordinated_zoo', action='store_true', help='Enable coordinated client/server ZOO with shared seeds')
    # ZOO perturbation distribution for server
    parser.add_argument('--perturbation_dist', type=str, choices=['rademacher','gaussian','bernoulli'], default='rademacher', help='Distribution for ZOO perturbations on server')
    parser.add_argument('--bernoulli_p', type=float, default=0.5, help='Bernoulli(p) parameter (used when perturbation_dist=bernoulli)')
    parser.add_argument('--no_center_bernoulli', dest='center_bernoulli', action='store_false', help='Disable centering/scaling Bernoulli to zero mean and unit variance')
    parser.set_defaults(center_bernoulli=True)
    parser.add_argument('--clip_grad_norm', type=float, default=0.0, help='Max gradient norm for clipping server params (0 disables)')
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
    
    # Global precision (replaces compute_dtype, wire_fp16, gcut-dtype)
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
    # Resolve compute dtype
    _precision_choice = str(getattr(args, 'precision', getattr(args, 'compute_dtype', 'fp32'))).lower()
    _compute_dtype = (torch.float16 if _precision_choice == 'fp16' else torch.float32)
    # KV prefix model (new): always create so handlers can access
    tmp_cfg = AutoModelForCausalLM.from_pretrained(args.model_name, torch_dtype=_compute_dtype, device_map=None).config
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
    coordinated_trainer = None
    if args.use_zeroth_order_server and bool(getattr(args, 'coordinated_zoo', False)):
        coordinated_trainer = CoordinatedZOOTrainer(
            params=trainable_params,
            num_pert=args.num_pert,
            mu=args.mu,
            seed=args.seed,
            estimator=str(getattr(args, 'estimator', 'central')),
            perturbation_dist=str(getattr(args, 'perturbation_dist', 'rademacher')),
            bernoulli_p=float(getattr(args, 'bernoulli_p', 0.5)),
            center_bernoulli=bool(getattr(args, 'center_bernoulli', True)),
            device=device,
        )
        print("✅ Coordinated ZOO trainer initialized (server).")
        coordinated_trainer.register_params(trainable_params)

    if args.use_zeroth_order_server and not bool(getattr(args, 'coordinated_zoo', False)):
        print("Setting up ZOO gradient estimator...")
        grad_estimator = StochasticGradientApproximator(
            model_params=trainable_params,
            perturbation_scale=args.mu,
            sample_count=args.num_pert,
            compute_device=device,
            data_type=torch.float32,
            perturbation_distribution=str(getattr(args, 'perturbation_dist', 'rademacher')).lower(),
            bernoulli_p=float(getattr(args, 'bernoulli_p', 0.5)),
            center_bernoulli=bool(getattr(args, 'center_bernoulli', True)),
        )
        print(f"ZOO gradient estimator created with mu={args.mu}, num_pert={args.num_pert}, dist={getattr(args, 'perturbation_dist', 'rademacher')}")
    
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

            elif mtype == "coordinated_init":
                if not (args.use_zeroth_order_server and bool(getattr(args, "coordinated_zoo", False))):
                    server.send_data({"type": "error", "reason": "coordinated_zoo_disabled"})
                    continue
                if coordinated_trainer is None:
                    server.send_data({"type": "error", "reason": "coordinated trainer not initialized"})
                    continue

                def _coord_loss_fn(h_cut, attention_mask, labels, cut_layer: int):
                    out = _server_forward_from_cut(
                        kv_model,
                        h_cut=h_cut,
                        attention_mask=attention_mask,
                        labels=labels,
                        cut=cut_layer,
                        compute_g_cut=False,
                        return_loss_tensor=False,
                    )
                    return float(out.get("loss", 0.0))

                try:
                    stats = coordinated_zoo_step_server(
                        server_model=kv_model,
                        optimizer=optimizer,
                        server_comm=server,
                        batch_data=msg,
                        trainer=coordinated_trainer,
                        device=device,
                        args=args,
                        loss_fn=_coord_loss_fn,
                        client_offset=CLIENT_OFFSET_DEFAULT,
                    )
                    avg_loss = stats.get("avg_loss", 0.0)
                    grad_norm = stats.get("grad_norm", 0.0)
                    print(f"[coordinated-zoo] step complete: avg_loss={avg_loss:.4f}, grad_norm={grad_norm:.4f}")
                except Exception as _e:
                    print(f"⚠️ Coordinated ZOO server step failed: {_e}")
                    traceback.print_exc()
                    try:
                        server.send_data(
                            {
                                "type": "losses",
                                "losses_plus": [],
                                "losses_minus": [],
                                "error": str(_e),
                            }
                        )
                    except Exception:
                        pass
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
        