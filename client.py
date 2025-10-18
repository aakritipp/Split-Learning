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
from metrics import calculate_client_answer_accuracy
from SGDGradientEst import StochasticGradientApproximator
import types
from prefix_kv import (
    PrefixKV, 
    merge_past_key_values, 
    flatten_grad_state
)
# Global KV model holder (set in main)
kv_model = None

from lora import (
    apply_lora_to_opt, 
    iter_lora_parameters, 
    get_lora_state_dict
)

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

class LoRAClientModel(nn.Module):
    """
    Client-side when tuning=LoRA. Keeps full base_model, injects LoRA only into layers [cut..L-1].
    Presents server_kv_mirror/client_kv stubs so the rest of your code (forward, masks) stays unified.
    """
    def __init__(self, model_name: str, total_layers: int, cut_layer: int,
                 r: int = 8, alpha: int = 16, dropout: float = 0.0,
                 targets=("q_proj","v_proj")):
        super().__init__()
        self.base_model = AutoModelForCausalLM.from_pretrained(model_name, torch_dtype=torch.float32, device_map=None)
        for p in self.base_model.parameters():
            p.requires_grad = False
        self.total_layers = total_layers
        self.cut_layer = cut_layer

        apply_lora_to_opt(self.base_model,
                          targets=tuple(targets),
                          layer_range=(cut_layer, total_layers-1),
                          r=r, lora_alpha=alpha, lora_dropout=dropout)

        # keep interface used elsewhere
        self.server_kv_mirror = _NoPrefixStub()
        self.client_kv       = _NoPrefixStub()

    def trainable_parameters(self):
        return iter_lora_parameters(self.base_model, layer_range=(self.cut_layer, self.total_layers-1))

class FullClientModel(nn.Module):
    """
    Client-side when tuning=full. Keeps full base_model; trains layers [cut..L-1] + final_layer_norm + lm_head.
    Presents no-prefix stubs so unified forward-from-cut path works without KV prefixes.
    """
    def __init__(self, model_name: str, total_layers: int, cut_layer: int):
        super().__init__()
        self.base_model = AutoModelForCausalLM.from_pretrained(model_name, torch_dtype=torch.float32, device_map=None)
        for p in self.base_model.parameters():
            p.requires_grad = False
        self.total_layers = total_layers
        self.cut_layer = cut_layer

        # Enable grads for client half: decoder layers [cut..L-1], final_layer_norm, lm_head
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
        self.server_kv_mirror = _NoPrefixStub()
        self.client_kv       = _NoPrefixStub()

    def trainable_parameters(self):
        return (p for p in self.base_model.parameters() if p.requires_grad)


@torch.no_grad()
def _estimate_g_cut_fd(kv_model, h_cut, attention_mask, labels, cut, mu=1e-3, num_pert=8, mode: str = 'central'):
    # h_cut on the right device/dtype already
    h0 = h_cut
    g_hat = torch.zeros_like(h0, dtype=torch.float32)

    for _ in range(num_pert):
        u = torch.randn_like(h0, dtype=torch.float32)
        with torch.no_grad():
            lp = _client_forward_from_cut(
                kv_model, h_cut=h0 + mu*u,
                attention_mask=attention_mask, labels=labels, cut=cut,
                compute_g_cut=False, return_loss_tensor=False
            )["loss"]
            if str(mode).lower() == 'central':
                ln = _client_forward_from_cut(
                    kv_model, h_cut=h0 - mu*u,
                    attention_mask=attention_mask, labels=labels, cut=cut,
                    compute_g_cut=False, return_loss_tensor=False
                )["loss"]
                scale = (lp - ln) / (2.0*mu)
            else:
                ln = _client_forward_from_cut(
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

def _client_forward_from_cut(
    kv_model,                       # your ClientKVModel instance
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

    # Figure out client prefix length P (for layers >= cut)
    try:
        prefix_len = int(kv_model.client_kv.k.shape[-2])  # [L, H, P, D]
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

    # Build client past for layers >= cut
    bsz = h.size(0)
    # Build client past only if client_kv exists (prefix mode)
    client_past = {}
    if hasattr(kv_model, "client_kv") and hasattr(kv_model.client_kv, "get_local_past"):
        client_past = kv_model.client_kv.get_local_past(bsz)


    # Run layers cut..end with per-layer client prefixes
    for li in range(cut, len(decoder.layers)):
        layer = decoder.layers[li]
        pkv = client_past.get(li, None)
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
        out["g_cut"] = g_cut.detach().to(torch.float32).cpu()
    
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

class ClientKVModel(nn.Module):
    """
    Client-side model that supports per-layer KV-prefix (only prefixes are trainable).
    It holds two PrefixKV modules:
      - server_kv_mirror: non-owned copy used to compute gradients for server (when server uses SGD)
      - client_kv: trainable prefixes for the client's layers
    We still run the full model on the client, but inject KV prefixes into all layers.
    """
    def __init__(self, model_name, total_layers, cut_layer, num_prefix=10):
        super().__init__()
        self.base_model = AutoModelForCausalLM.from_pretrained(model_name, torch_dtype=torch.float32, device_map=None)
        for p in self.base_model.parameters():
            p.requires_grad = False

        self.total_layers = total_layers
        self.cut_layer = cut_layer
        self.hidden_size = self.base_model.config.hidden_size

        # server prefixes live on layers [0..cut-1]; client prefixes live on [cut..L-1]
        self.server_kv_mirror = PrefixKV(self.base_model.config, list(range(0, cut_layer)), num_prefix=num_prefix, device=self.base_model.device)
        self.client_kv = PrefixKV(self.base_model.config, list(range(cut_layer, total_layers)), num_prefix=num_prefix, device=self.base_model.device)
        # Only client prefixes are trainable by default
        for p in self.server_kv_mirror.parameters():
            p.requires_grad = False
        for p in self.client_kv.parameters():
            p.requires_grad = True


    def load_server_state(self, state_dict: dict):
        # state_dict expected to contain keys "k" and "v" tensors matching server_kv_mirror
        with torch.no_grad():
            if "k" in state_dict:
                self.server_kv_mirror.k.copy_(state_dict["k"].to(self.server_kv_mirror.k.device, dtype=self.server_kv_mirror.k.dtype))
            if "v" in state_dict:
                self.server_kv_mirror.v.copy_(state_dict["v"].to(self.server_kv_mirror.v.device, dtype=self.server_kv_mirror.v.dtype))

    def forward_full(self, input_ids, attention_mask, labels=None, require_server_grad=False):
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

        # ---- 2) build KV cache from prefixes (server + client) ----
        server_past = self.server_kv_mirror.get_local_past(bsz)
        client_past = self.client_kv.get_local_past(bsz)
        past_kv = merge_past_key_values(self.total_layers, server_past, client_past)
        self.server_kv_mirror.set_requires_grad(require_server_grad)

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
        for mod in [self.server_kv_mirror, self.client_kv]:
            if hasattr(mod.k, "grad") and mod.k.grad is not None:
                mod.k.grad.zero_()
            if hasattr(mod.v, "grad") and mod.v.grad is not None:
                mod.v.grad.zero_()

    def trainable_parameters(self):
        # For prefix mode, the only trainables should be client_kv
        return self.client_kv.parameters()


class Client:
    def __init__(self, host, port):
        self.socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self.socket.connect((host, port))
        
    def send_data(self, data):
        try:
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
            return pickle.loads(data)
        except Exception as e:
            print(f"⌠Failed to receive data: {e}")
            raise
    
    def close(self):
        self.socket.close()

def handle_forward_cut_unified(client, kv_model, optimizer, grad_estimator, args, device,
                               pkt: dict, meta: dict, batch_idx: int):
    """
    Unified handler for 'forward_cut' covering all 4 combos:
      (server_opt, client_opt) in {(SGD,SGD), (SGD,ZOO), (ZOO,SGD), (ZOO,ZOO)}.

    Protocol:
      - Server sends (h_cut, attention_mask, labels, cut) and meta.need_g_cut.
      - Client ALWAYS computes the loss on its half.
      - If client_opt == sgd: loss.backward(); optimizer.step().
      - If client_opt == zoo: do a ZOO step using a loss-only objective.
      - If meta.need_g_cut: return ∂L/∂h_cut (never client param grads).

    True split guarantees:
      - No parameter gradients cross the boundary.
      - Only activations go down; only loss (and optional g_cut) go up.

    Args:
      client: your network wrapper (has send_data/receive_data)
      kv_model: client model wrapper that owns the trainable client prefixes
      optimizer: torch optimizer for client prefixes
      grad_estimator: your ZOO estimator (has estimate_gradients(objective_function=...))
      args: parsed args (must have args.client_opt in {"sgd","zoo"})
      device: torch.device
      pkt: payload from server with keys ['h_cut','attention_mask','labels','cut']
      meta: meta dict, expects meta.get('need_g_cut', bool)
    """
    # Prefer explicit flag; default to legacy behavior (no zoo_eval => need g_cut)
    need_g_cut = bool(meta.get("need_g_cut", not bool(meta.get("zoo_eval", False)))) 
    client_sgd = (not args.use_zeroth_order_client)  # True => SGD, False => ZOO

    model_ref = getattr(kv_model, "base_model", kv_model)
    param0 = next(model_ref.parameters())
    target_device = param0.device
    target_dtype  = param0.dtype

    server_data = pkt["data"]
    mode        = pkt.get("mode", "train")   # <-- new, default 'train'
    task_type   = pkt.get("meta", {}).get("task_type", "qa")
    max_new     = pkt.get("meta", {}).get("max_new_tokens", 20)
    tgt_len    = int(server_data.get("tgt_len", 0))

    # --- Pure evaluation mode OR strict ZOO probe: compute MEAN loss only, no updates or gradients ---
    if mode == "eval" or bool(meta.get("zoo_eval", False)):
        model_ref = getattr(kv_model, "base_model", kv_model)
        param0 = next(model_ref.parameters())
        target_device = param0.device
        # Force fp32 for strict ZOO probes to preserve FD signal
        target_dtype  = torch.float32

        attn_mask = server_data.get("attention_mask", None)
        labels = server_data.get("labels", None)
        if attn_mask is not None:
            attn_mask = attn_mask.to(device=target_device)
        if labels is not None:
            labels = labels if torch.is_tensor(labels) else torch.as_tensor(labels)
            labels = labels.to(device=target_device)

        cut = int(server_data.get("cut_layer", pkt.get("meta", {}).get("cut_layer", args.cut_layer)))

        from torch.cuda.amp import autocast
        kv_model.eval()  # disable dropout for stability
        with torch.no_grad():
            with autocast(enabled=False):
                h_cut = server_data["h_cut"].to(device=target_device, dtype=torch.float32)
                out = _client_forward_from_cut(
                    kv_model,
                    h_cut=h_cut.detach(),
                    attention_mask=attn_mask,
                    labels=labels,
                    cut=cut,
                    compute_g_cut=False,
                    return_loss_tensor=False,
                )

        if bool(meta.get("zoo_eval", False)):
            # Strict ZOO probe: return scalar loss only
            client.send_data({
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
            client.send_data(reply)
            return

    # ---- Unpack incoming tensors ----
    # Expect tensors already on CPU -> move to device (or adapt to your transport)
    h_cut = server_data["h_cut"].to(device=target_device, dtype=target_dtype)
    B, T, H = h_cut.shape
    attn_mask = server_data.get("attention_mask", None)
    labels = server_data.get("labels", None)

    T = h_cut.shape[1]
    if tgt_len and tgt_len != T:
        # Sanity check: trust the server's view if present
        print(f"⚠️ client: adjusting local T={T} -> server T={tgt_len} (will pad gradients if needed)")
        T = tgt_len

    if attn_mask is not None:
        attn_mask = attn_mask.to(device=target_device)
    # labels = pkt.get("labels", None)
    if labels is not None:
        labels = labels if torch.is_tensor(labels) else torch.as_tensor(labels)
        labels = labels.to(device=target_device)
    if attn_mask is not None and labels is not None:
        # Note: h_cut should match the trimmed length from server
        _, attn_mask, labels = right_trim(torch.zeros_like(attn_mask), attn_mask, labels)
    
    if labels is not None and labels.dim() == 2 and labels.shape[1] != T:
        if labels.shape[1] > T:
            labels = labels[:, :T]
        else:
            pad = torch.full((labels.shape[0], T - labels.shape[1]), -100,
                            dtype=labels.dtype, device=labels.device)
            labels = torch.cat([labels, pad], dim=1)
    cut = server_data.get("cut_layer", None)
    if cut is None:
        raise KeyError("Could not determine cut layer: expected pkt['cut'] or pkt['meta']['cut_layer'] or args.cut_layer")
    cut = int(cut)
    
    if client_sgd:
        h_cut = h_cut.detach().requires_grad_(True)
        # allow autograd (for client prefixes + optional boundary grad)
        out = _client_forward_from_cut(
            kv_model, h_cut=h_cut, attention_mask=attn_mask, labels=labels,
            cut=cut, compute_g_cut=False, return_loss_tensor=True
        )
    else:
        # STRICT ZOO: no graph, no boundary-grad inside this forward
        with torch.no_grad():
            h_cut = h_cut.detach()
            out = _client_forward_from_cut(
                kv_model, h_cut=h_cut, attention_mask=attn_mask, labels=labels,
                cut=cut, compute_g_cut=False, return_loss_tensor=True
            )

    loss_tensor = out["loss_tensor"]                 # scalar tensor
    loss_value = float(out["loss"])                  # python float for wire

    # Prepare optional g_cut to send
    g_cut_to_send = None

    # ---- Local client update ----
    if client_sgd:
        # True first-order update of client prefixes
        optimizer.zero_grad(set_to_none=True)
        loss_tensor.backward()  # grads flow into client prefixes (and h_cut.grad if required)
        try:
            torch.nn.utils.clip_grad_norm_(list(getattr(kv_model, "trainable_parameters", lambda: kv_model.client_kv.parameters())()), max_norm=1.0)
        except Exception:
            pass
        if need_g_cut:
            if h_cut.grad is None:
                raise RuntimeError("need_g_cut=True but h_cut.grad is None. Ensure h_cut.requires_grad_(True).")
            g = h_cut.grad.detach()
            g_cut_to_send = g.to(dtype=torch.float32, device="cpu").contiguous().numpy()
        optimizer.step()

        # clean leaf grad
        if h_cut.grad is not None:
            h_cut.grad = None

    else:
        # Zeroth order update of client prefixes (loss-only objective)
        # If server needs g_cut, compute it via finite-difference on h_cut
        if need_g_cut:
            was_training = kv_model.training
            kv_model.eval()
            try:
                g = _estimate_g_cut_fd(
                    kv_model,
                    h_cut.to(dtype=torch.float32, device=target_device),
                    attn_mask, labels, cut,
                    mu=getattr(args, "mu_gcut", getattr(args, "mu", 1e-2)),
                    num_pert=getattr(args, "num_pert_gcut", getattr(args, "num_pert", 16)),
                    mode=str(getattr(args, 'estimator', 'central')),
                )
            finally:
                if was_training:
                    kv_model.train()
            g_cut_to_send = g.to(dtype=torch.float32, device="cpu").contiguous().numpy()

    # else:
    #     # Zeroth order update of client prefixes (loss-only objective)
    #     # If server needs g_cut, compute it via autograd.grad wrt h_cut (no client param grads cross boundary)
    #     if need_g_cut:
    #         g = _estimate_g_cut_fd(
    #             kv_model, h_cut, attn_mask, labels, cut,
    #             mu=getattr(args, "mu_gcut", getattr(args, "mu", 1e-3)),
    #             num_pert=getattr(args, "num_pert_gcut", getattr(args, "num_pert", 8)),
    #         )
    #         g_cut_to_send = g.to(dtype=torch.float32, device="cpu").contiguous().numpy()

        # Define a pure loss-only objective for ZOO (no gradients, just returns float loss)
        # def client_objective(_x, _y):
        #     with torch.no_grad():
        #         o = _client_forward_from_cut(
        #             kv_model,
        #             h_cut=server_data["h_cut"].to(device).detach(),
        #             attention_mask=attn_mask,
        #             labels=labels, cut=cut,
        #             compute_g_cut=False,
        #             return_loss_tensor=False
        #         )
        #         return torch.tensor(float(o["loss"]), device="cpu")

        def client_objective(_x, _y):
            was_training = kv_model.training
            kv_model.eval()
            try:
                with torch.no_grad():
                    o = _client_forward_from_cut(
                        kv_model,
                        h_cut=server_data["h_cut"].to(device=device, dtype=target_dtype),
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
        # grad_estimator.model_params = list(kv_model.client_kv.parameters())  # ensure current params
        grad_estimator.model_params = list(getattr(kv_model, "trainable_parameters", lambda: kv_model.client_kv.parameters())())

        try:
            grad_estimator.estimate_gradients(_dummy_x, _dummy_y, client_objective,
                                              random_seed=batch_idx * 2029 + getattr(args, "seed", 0))
        except TypeError:
            # Fallback if your class has the old signature
            grad_estimator.estimate_gradients(torch.zeros(1), torch.zeros(1), client_objective,
                                              random_seed=batch_idx * 2029 + getattr(args, "seed", 0))
        # torch.nn.utils.clip_grad_norm_(list(getattr(kv_model, "trainable_parameters", lambda: kv_model.client_kv.parameters())()), max_norm=1.0)
        optimizer.step()
        # STRICT ZOO probe: return scalar loss only, no client updates here
        # with torch.no_grad():
        #     pass  # ensure no autograd graph kept
        # Apply FD gradients to client params
        try:
            torch.nn.utils.clip_grad_norm_(list(getattr(kv_model, "trainable_parameters", lambda: kv_model.client_kv.parameters())()), max_norm=1.0)
        except Exception:
            pass
        optimizer.step()

    # ---- Reply to server ----
    reply = {"loss": float(loss_tensor.item())}
    if g_cut_to_send is not None:
        # sanity: shape must match h_cut
        # (can't use tensors in dict over pickle reliably -> send numpy)
        # reply["g_cut"] = g_cut_to_send
        # reply["g_cut"] = g_cut_to_send.detach().to(torch.float32, device="cpu").contiguous().numpy()
        # reply["g_cut"] = g_cut_to_send.detach().to(torch.float32, device="cpu").contiguous().numpy()
        if isinstance(g_cut_to_send, torch.Tensor):
            gc = g_cut_to_send.detach().to(torch.float32, device="cpu").contiguous().numpy()
        else:
            gc=np.asarray(g_cut_to_send, dtype=np.float32)
        reply["g_cut"] = gc
    client.send_data(reply)

def parse_args():
    parser = argparse.ArgumentParser(description='Split Learning LLM Client')
    parser.add_argument('--model_name', type=str, default='facebook/opt-125m', help='Pretrained model name')
    parser.add_argument('--num_prefix', type=int, default=20, help='Number of prefix tokens')
    parser.add_argument('--cut_layer', type=int, default=1, help='Split index: 0..cut-1 server; cut..L-1 client')
    parser.add_argument('--max_length', type=int, default=512, help='Maximum sequence length')
    parser.add_argument('--seed', type=int, default=42, help='Random seed')
    parser.add_argument('--lr', type=float, default=1e-3, help='Learning rate')
    parser.add_argument('--zoo_lr', type=float, default=1e-4, help='ZOO learning rate')
    parser.add_argument('--momentum', type=float, default=0.9, help='SGD momentum')
    parser.add_argument('--weight_decay', type=float, default=0.0, help='Weight decay for optimizer')
    parser.add_argument('--train_batch_size', type=int, default=4, help='Training batch size')
    parser.add_argument('--test_batch_size', type=int, default=4, help='Test batch size')
    parser.add_argument('--max_steps', type=int, default=4000, help='Maximum training steps')
    parser.add_argument('--epochs', type=int, default=10, help='Number of epochs (fallback)')
    # ZOO parameters
    parser.add_argument('--mu', type=float, default=1e-3, help='ZOO perturbation scale')
    parser.add_argument('--num_pert', type=int, default=64, help='Number of ZOO perturbations')
    parser.add_argument('--estimator', type=str, choices=['central','forward'], default='central', help='Finite-diff estimator type for ZOO/g_cut')
    parser.add_argument('--use_zeroth_order_client', action='store_true', help='Use ZOO for client')
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
    parser.add_argument('--host', type=str, default='127.0.0.1', help='Server host to connect')
    parser.add_argument('--port', type=int, default=12345, help='Server port to connect')
    
    # argparse setup
    parser.add_argument("--gcut-dtype", choices=["fp16", "fp32"], default="fp32", help="Wire precision for g_cut payloads.")
    parser.add_argument('--wire_fp16', choices=['auto','on','off'], default='auto', help='Precision for wire activations (client will honor server)')

    args = parser.parse_args()
    return args


def _assert_only_expected_trainables(module: nn.Module, mode: str, layer_range=None, side: str = None):
    for n, p in module.named_parameters():
        if mode == "prefix":
            if side == "client":
                is_allowed = n.startswith("client_kv.")
            elif side == "server":
                is_allowed = n.startswith("kv.")
            else:
                is_allowed = (n.startswith("client_kv.") or n.startswith("kv.") or ("prefix" in n))

            ok = ("lora_A" not in n and "lora_B" not in n) and ((is_allowed) == p.requires_grad)

        elif mode == "lora":
            is_lora = ("lora_A" in n) or ("lora_B" in n)
            ok = (is_lora == p.requires_grad) and ("kv." not in n) and ("client_kv." not in n)

        elif mode == "full":
            # Client side full-FT should only train decoder layers [cut..L-1], final_layer_norm and lm_head
            cut = int(getattr(module, "cut_layer", 0))
            def _client_allowed(name: str) -> bool:
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
                    name.startswith("kv.") or name.startswith("client_kv.") or ("prefix" in name)
                    or ("lora_A" in name) or ("lora_B" in name)
                )
            if side == "client":
                if p.requires_grad:
                    ok = (not _forbidden(n)) and _client_allowed(n)
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
    print("SPLIT LEARNING CLIENT WITH PREFIX TUNING")
    print("=" * 60)
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    print(f"Loading tokenizer: {args.model_name}")
    tokenizer = safe_get_hf_tokenizer(args.model_name)
    print("Tokenizer loaded successfully")
    
    print("Creating client model...")
    # KV prefix model (new): always create so handlers can access
    tmp_cfg = AutoModelForCausalLM.from_pretrained(args.model_name, torch_dtype=torch.float32, device_map=None).config
    total_layers = tmp_cfg.num_hidden_layers
    if args.tuning == 'lora':
        kv_model = LoRAClientModel(
            args.model_name, total_layers, args.cut_layer,
            r=args.lora_r, alpha=args.lora_alpha, dropout=args.lora_dropout,
            targets=tuple(args.lora_targets.split(','))
        ).to(device)
        trainable_params = list(kv_model.trainable_parameters())
        _assert_only_expected_trainables(kv_model, args.tuning, side="client")

        print(f"Client owns layers [{args.cut_layer}, {total_layers-1}] with LoRA r={args.lora_r}, alpha={args.lora_alpha}")
    elif args.tuning == 'full':
        kv_model = FullClientModel(args.model_name, total_layers, args.cut_layer).to(device)
        trainable_params = list(kv_model.trainable_parameters())
        _assert_only_expected_trainables(kv_model, args.tuning, side="client")
        print(f"Client full-FT layers [{args.cut_layer}, {total_layers-1}] enabled; final_layer_norm/lm_head trainable")
    else:
        kv_model = ClientKVModel(args.model_name, total_layers, args.cut_layer, num_prefix=args.num_prefix).to(device)
        for p in getattr(kv_model, "base_model", kv_model).parameters():
            p.requires_grad = False
        for p in kv_model.client_kv.parameters():
            p.requires_grad = True
        for p in kv_model.server_kv_mirror.parameters():
            p.requires_grad = False

        trainable_params = list(kv_model.client_kv.parameters())
        _assert_only_expected_trainables(kv_model, args.tuning, side="client")

        print(f"Client owns layers [{args.cut_layer}, {total_layers-1}] with {args.num_prefix} prefix tokens each")

    
    print(f"Model loaded: {args.model_name}")
    print(f"Client owns layers [{args.cut_layer}, {total_layers-1}] with {args.num_prefix} prefix tokens each")
    
    if args.use_zeroth_order_client:
        # Client ZOO: SGD as a simple applicator when estimator populates grads
        optimizer = optim.SGD(
            list(getattr(kv_model, "trainable_parameters", lambda: kv_model.client_kv.parameters())()),
            lr=args.zoo_lr,
            momentum=0.0,
            weight_decay=args.weight_decay,
        )
        print(f"Client using ZOO optimizer with lr={args.zoo_lr}, weight_decay={args.weight_decay}")
    else:
        trainable = list(getattr(kv_model, "trainable_parameters", lambda: kv_model.client_kv.parameters())())
        optimizer = optim.SGD(
            trainable,
            lr=args.lr,
            momentum=args.momentum,
            weight_decay=args.weight_decay,
        )
        print(f"Client using SGD optimizer with lr={args.lr}, momentum={args.momentum}, wd={args.weight_decay}")
    
    # Setup ZOO gradient estimator if needed
    grad_estimator = None
    if args.use_zeroth_order_client:
        print("Setting up ZOO gradient estimator...")
        grad_estimator = StochasticGradientApproximator(
            model_params=trainable_params,
            perturbation_scale=args.mu,
            sample_count=args.num_pert,
            compute_device=device,
            data_type=torch.float32
        )
        print(f"ZOO gradient estimator created with mu={args.mu}, num_pert={args.num_pert}")
    
    try:
        print("=" * 60)
        print("CLIENT STARTING - ATTEMPTING TO CONNECT TO SERVER")
        print("=" * 60)
        print(f"Trying to connect to server at {args.host}:{args.port}...")
        
        client = Client(args.host, args.port)
        
        print("=" * 60)
        print("CLIENT SUCCESSFULLY CONNECTED TO SERVER!")
        print("=" * 60)

        client_config = {
            'model_name': args.model_name,
            'num_prefix': args.num_prefix,
            'lr': args.lr,
            # True => client uses SGD; False => client uses ZOO
            'client_sgd': (not args.use_zeroth_order_client),
            'zoo_lr': getattr(args, "zoo_lr", None),
            'mu': getattr(args, "mu", None),
            'num_pert': getattr(args, "num_pert", None),
            'client_tuning': args.tuning,
            'lora': {
                'r': args.lora_r,
                'alpha': args.lora_alpha,
                'dropout': args.lora_dropout,
                'targets': args.lora_targets,
            }
        }
        client.send_data(client_config)
        print(f"Sent client configuration")

        print("=" * 60)
        if args.use_zeroth_order_client:
            print("STARTING ZOO TRAINING REQUEST HANDLER...")
            print(f"  Client prefixes WILL be trained using ZOO")
        else:
            print("STARTING SGD TRAINING REQUEST HANDLER...")
            print(f"  Client prefixes WILL be trained using SGD")
        print("=" * 60)

        batch_idx = 0
        while True:
            msg = client.receive_data()
            if msg is None:
                print("No message received (server closed?). Exiting loop.")
                break

            mtype = msg.get("type", "")
            if mtype == "get_client_kv_state":
                # In LoRA mode there are no client prefixes; return empty state
                if args.tuning == 'lora' or not hasattr(kv_model, 'client_kv'):
                    client.send_data({"type": "client_kv_state", "state": {"k": None, "v": None}})
                else:
                    state = {
                        "k": kv_model.client_kv.k.detach().cpu(),
                        "v": kv_model.client_kv.v.detach().cpu(),
                    }
                    client.send_data({"type": "client_kv_state", "state": state})
                continue

            elif mtype == "get_client_prefix_snapshot":
                with torch.no_grad():
                    if hasattr(kv_model, "client_kv") and getattr(kv_model.client_kv, "k", None) is not None:
                        kc = kv_model.client_kv.k.detach().cpu()
                        vc = kv_model.client_kv.v.detach().cpu()
                        client.send_data({"type": "client_prefix_snapshot", "ok": True, "k": kc, "v": vc})
                    else:
                        client.send_data({"type": "client_prefix_snapshot", "ok": True, "k": None, "v": None})
                continue

            elif mtype == "training_complete":
                print("TRAINING COMPLETED - CLIENT SHUTTING DOWN")
                break

            elif mtype == "get_client_lora_state":
                try:
                    # Return LoRA A/B only for client half [cut..L-1]
                    if args.tuning == 'lora':
                        state = get_lora_state_dict(
                            getattr(kv_model, "base_model", kv_model),
                            layer_range=(args.cut_layer, total_layers-1)
                        )
                        client.send_data({"type": "client_lora_state", "ok": True, "state": state})
                    else:
                        client.send_data({"type": "client_lora_state", "ok": True, "state": {}})
                except Exception as e:
                    client.send_data({"type": "client_lora_state", "ok": False, "error": str(e)})
                continue

            elif mtype == "get_client_full_state":
                try:
                    if args.tuning == 'full':
                        base = getattr(kv_model, "base_model", kv_model)
                        sd = base.state_dict()
                        # Filter to client half: decoder layers [cut..L-1], final_layer_norm, lm_head
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
                        client.send_data({"type": "client_full_state", "ok": True, "state": filt})
                    else:
                        client.send_data({"type": "client_full_state", "ok": True, "state": {}})
                except Exception as e:
                    client.send_data({"type": "client_full_state", "ok": False, "error": str(e)})
                continue

            if mtype == "forward_cut":
                data = msg.get("data")
                if data is None or not isinstance(data, dict):
                    print("⌠CLIENT: malformed forward_cut (missing 'data'); ignoring this packet.")
                    # Optionally: client.send_data({"type":"error","reason":"missing_data_in_forward_cut"})
                    continue
                handle_forward_cut_unified(
                    client=client,
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
                print(f"Received '{mtype}' from server. Exiting.")
                break

            else:
                # You can log/ignore other control messages here
                print(f"Unknown message type: {mtype}")
        
    except ConnectionRefusedError:
        print("=" * 60)
        print("⌠CONNECTION FAILED!")
        print("=" * 60)
        print(f"Could not connect to server at {args.host}:{args.port}")
        print("Make sure the server is running first:")
        print(f"   python server.py --model_name {args.model_name}")
        print("=" * 60)
    except Exception as e:
        print(f"⌠CLIENT ERROR: {e}")
        traceback.print_exc()
    finally:
        try:
            client.close()
        except:
            pass
        print("CLIENT SHUTDOWN COMPLETE")
        