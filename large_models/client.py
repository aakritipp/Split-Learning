import socket
import pickle
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from transformers import AutoModelForCausalLM, AutoTokenizer
import numpy as np
import argparse
import traceback
from metrics import calculate_client_answer_accuracy
from SGDGradientEst import StochasticGradientApproximator

# Global KV model holder (set in main)
kv_model = None

try:
    from transformers.cache_utils import DynamicCache, StaticCache
except Exception:
    DynamicCache = None
    StaticCache = None

import torch.nn.functional as F

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

def _resolve_base_lm(model_like):
    def _looks_like_hf_lm(x):
        return hasattr(x, "model") and hasattr(x.model, "decoder") and hasattr(x, "lm_head")
    obj = model_like
    for _ in range(6):
        if _looks_like_hf_lm(obj):
            return obj
        for name in ("base_model","hf_model","model","module","net","inner","wrapped","lm"):
            if hasattr(obj, name):
                obj = getattr(obj, name)
                break
        else:
            break
    if _looks_like_hf_lm(model_like):
        return model_like
    raise AttributeError(f"Could not resolve HF LM from {type(model_like).__name__}")

def _client_forward_from_cut(
    kv_model,                       # your ClientKVModel instance
    h_cut: torch.Tensor,
    attention_mask: torch.Tensor,
    labels: torch.Tensor,
    cut: int,
    compute_g_cut: bool = True,
    return_loss_tensor: bool = False,  # NEW: option to return loss as tensor
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
    client_past = kv_model.client_kv.get_local_past(bsz)  # {layer_idx: (k,v)}

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

    # Causal LM loss (safe)
    if labels is not None and logits.size(1) >= 2:
        shift_logits = logits[:, :-1, :].contiguous()
        shift_labels = labels[:, 1:].contiguous()
        if (shift_labels != -100).any():
            loss = F.cross_entropy(
                shift_logits.view(-1, shift_logits.size(-1)),
                shift_labels.view(-1),
                ignore_index=-100,
                reduction="mean",
            )
        else:
            loss = torch.zeros((), device=device, dtype=dtype)
    else:
        loss = torch.zeros((), device=device, dtype=dtype)

    out = {"loss": float(loss.item())}
    if compute_g_cut:
        # Use retain_graph=True if we also need the loss tensor for local update
        g_cut, = torch.autograd.grad(loss, h, retain_graph=return_loss_tensor)
        out["g_cut"] = g_cut.detach().to(torch.float16).cpu()
    
    if return_loss_tensor:
        out["loss_tensor"] = loss  # Return the torch tensor for local update
    
    return out


def _normalize_batch_from_server(payload, device, tokenizer):
    ids = payload['input_ids'].to(device)
    am  = payload['attention_mask'].to(device)

    if 'labels' in payload and torch.is_tensor(payload['labels']):
        labels = payload['labels'].to(device).long()
    else:
        # Build labels from input_ids; ignore pads with -100 (HF causal LM shifts internally)
        labels = ids.clone()
        labels[am == 0] = -100
        pad_id = getattr(tokenizer, 'pad_token_id', None)
        if pad_id is not None:
            labels[ids == pad_id] = -100
        labels = labels.long()

    kv_state = payload.get('server_kv_state', None)
    return ids, am, labels, kv_state


def safe_get_hf_tokenizer(model_name):
    try:
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
        return tokenizer
    except Exception as e:
        print(f"⌠Failed to load tokenizer for {model_name}: {e}")
        raise


import types
from prefix_kv import PrefixKV, merge_past_key_values, flatten_grad_state

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


def calculate_accuracy(outputs, labels, batch=None, tokenizer=None):
    """Calculate SQUAD-specific accuracy from model outputs"""
    try:
        if outputs.loss is None:
            return 0.0
            
        logits = outputs.logits
        
        # Debug for first few calls
        global client_debug_count
        if 'client_debug_count' not in globals():
            client_debug_count = 0
            
        if client_debug_count < 3:
            print(f"\n  Client SQUAD Debug call {client_debug_count}:")
            print(f"   Logits shape: {logits.shape}")
            print(f"   Labels shape: {labels.shape}")
            if batch is not None and 'formatted_text' in batch:
                print(f"   Sample text: {batch['formatted_text'][0][:100]}...")
        
        # Ensure logits and labels have compatible shapes
        if logits.shape[1] != labels.shape[1]:
            min_len = min(logits.shape[1], labels.shape[1])
            logits = logits[:, :min_len, :]
            labels = labels[:, :min_len]
        
        # For language modeling: predict next token
        if logits.shape[1] > 1:
            shift_logits = logits[:, :-1, :].contiguous()
            shift_labels = labels[:, 1:].contiguous()
        else:
            shift_logits = logits
            shift_labels = labels
        
        # Get predictions
        predictions = torch.argmax(shift_logits, dim=-1)
        
        # Use the imported function for SQUAD-specific accuracy calculation
        answer_accuracy = calculate_client_answer_accuracy(
            predictions, shift_labels, batch, tokenizer
        )
        
        if client_debug_count < 3:
            print(f"   Answer accuracy: {answer_accuracy:.6f}")
        
        client_debug_count += 1
        return answer_accuracy
        
    except Exception as e:
        print(f"⌠Client SQUAD accuracy calculation failed: {e}")
        return 0.0


def handle_training_requests_zoo(client_model, optimizer, grad_estimator, client, device, args):
    """Handle training with ZOO - FIXED with client prefix training"""
    current_labels = None
    current_batch = None
    batch_count = 0
    recent_losses = []
    recent_accuracies = []
    
    print("Starting ZOO training request handler...")
    print(f"  ZOO Configuration:")
    print(f"    Perturbation scale (mu): {args.mu}")
    print(f"    Number of perturbations: {args.num_pert}")
    
    # Get tokenizer
    try:
        tokenizer = AutoTokenizer.from_pretrained(args.model_name)
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
    except:
        tokenizer = None
        print("⚠️ Could not load tokenizer")
    
    while True:
        try:
            server_data = client.receive_data()
            # inside the while True loop, after receiving `server_data`
            if server_data.get('type') == 'get_client_kv_state':
                state = {
                    "k": client_model.client_kv.k.detach().cpu(),
                    "v": client_model.client_kv.v.detach().cpu(),
                }
                client.send_data({"type": "client_kv_state", "state": state})
                continue

            
            if server_data.get('type') == 'training_complete':
                print("TRAINING COMPLETED - CLIENT SHUTTING DOWN")
                break
            
            elif server_data.get('type') == 'forward_cut':
                payload = server_data['data']
                meta    = server_data.get('meta', {})
                zoo_eval = bool(meta.get('zoo_eval', False))

                # tensors come CPU fp16; don't normalize tokenization here
                h_cut = torch.as_tensor(payload['h_cut'])
                attention_mask = torch.as_tensor(payload['attention_mask'])
                labels = None if payload['labels'] is None else torch.as_tensor(payload['labels'])
                cut = int(payload['meta']['cut_layer'])

                # Run from cut onward with return_loss_tensor=True for local update
                res = _client_forward_from_cut(
                    kv_model,
                    h_cut=h_cut,
                    attention_mask=attention_mask,
                    labels=labels,
                    cut=cut,
                    compute_g_cut=(not zoo_eval),
                    return_loss_tensor=True  # NEW: get loss tensor for local update
                )

                # Send response to server first (server might be blocking)
                if zoo_eval:
                    client.send_data({"loss": res["loss"]})
                else:
                    client.send_data({"loss": res["loss"], "g_cut": res["g_cut"]})

                # NOW do local ZOO update on client prefixes using the SAME batch
                if not zoo_eval and labels is not None:  # Only update when we have labels
                    print(f"\rBatch {batch_count}: ZOO client update...", end='', flush=True)
                    
                    # Define objective function for client ZOO
                    def client_objective():
                        out = _client_forward_from_cut(
                            kv_model,
                            h_cut=h_cut,  # reuse the received activation
                            attention_mask=attention_mask,
                            labels=labels,
                            cut=cut,
                            compute_g_cut=False,  # no need for g_cut locally
                            return_loss_tensor=True,
                        )
                        return out["loss_tensor"]
                    
                    # Make sure estimator targets only client prefixes
                    grad_estimator.model_params = list(kv_model.client_kv.parameters())
                    
                    # Estimate gradients and update
                    optimizer.zero_grad(set_to_none=True)
                    
                    # Compatible with both old and new SGDGradientEst API
                    try:
                        grad_estimator.estimate_gradients(
                            objective_fn=client_objective,
                            random_seed=batch_count * 1000 + args.seed,
                        )
                    except TypeError:
                        # Legacy API
                        dummy_input = torch.zeros(1)
                        dummy_labels = torch.zeros(1)
                        grad_estimator.estimate_gradients(
                            dummy_input, dummy_labels, client_objective,
                            random_seed=batch_count * 1000 + args.seed
                        )
                    
                    optimizer.step()
                    
                    # Clear grads
                    for p in kv_model.client_kv.parameters():
                        if p.grad is not None:
                            p.grad = None
                    
                batch_count += 1
                
            elif server_data.get('type') == 'eval_request':
                # Handle evaluation request from server
                payload = server_data['data']
                h_cut = torch.as_tensor(payload['h_cut'])
                attention_mask = torch.as_tensor(payload['attention_mask'])
                labels = None if payload['labels'] is None else torch.as_tensor(payload['labels'])
                cut = int(payload['meta']['cut_layer'])
                
                # Run evaluation forward pass
                res = _client_forward_from_cut(
                    kv_model,
                    h_cut=h_cut,
                    attention_mask=attention_mask,
                    labels=labels,
                    cut=cut,
                    compute_g_cut=False,  # No gradients needed for eval
                    return_loss_tensor=False
                )
                
                # Send back evaluation loss
                client.send_data({"eval_loss": res["loss"]})
                
        except Exception as e:
            print(f"⌠ERROR IN CLIENT ZOO TRAINING: {e}")
            import traceback
            traceback.print_exc()
            break


def handle_training_requests_sgd(client_model, optimizer, client, device, args):
    """Handle training with standard SGD (backpropagation) - FIXED with client prefix training"""
    current_labels = None
    current_batch = None
    batch_count = 0
    recent_losses = []
    recent_accuracies = []
    
    print("Starting SGD training request handler...")
    
    # Get tokenizer for accuracy calculation
    try:
        tokenizer = AutoTokenizer.from_pretrained(args.model_name)
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
    except:
        tokenizer = None
        print("⚠️ Could not load tokenizer for client-side accuracy calculation")
    
    while True:
        try:
            server_data = client.receive_data()

            # inside the while True loop, after receiving `server_data`
            if server_data.get('type') == 'get_client_kv_state':
                state = {
                    "k": client_model.client_kv.k.detach().cpu(),
                    "v": client_model.client_kv.v.detach().cpu(),
                }
                client.send_data({"type": "client_kv_state", "state": state})
                continue

            
            if server_data.get('type') == 'training_complete':
                print("TRAINING COMPLETED - CLIENT SHUTTING DOWN")
                break
                
            elif server_data.get('type') == 'forward_cut':
                payload = server_data['data']
                meta    = server_data.get('meta', {})
                zoo_eval = bool(meta.get('zoo_eval', False))

                # tensors come CPU fp16; don't normalize tokenization here
                h_cut = torch.as_tensor(payload['h_cut'])
                attention_mask = torch.as_tensor(payload['attention_mask'])
                labels = None if payload['labels'] is None else torch.as_tensor(payload['labels'])
                cut = int(payload['meta']['cut_layer'])

                # Run from cut onward with return_loss_tensor=True for local update
                res = _client_forward_from_cut(
                    kv_model,
                    h_cut=h_cut,
                    attention_mask=attention_mask,
                    labels=labels,
                    cut=cut,
                    compute_g_cut=(not zoo_eval),
                    return_loss_tensor=True  # NEW: get loss tensor for local update
                )

                # Send response to server first
                if zoo_eval:
                    client.send_data({"loss": res["loss"]})
                else:
                    client.send_data({"loss": res["loss"], "g_cut": res["g_cut"]})

                # NOW do local SGD update on client prefixes using the SAME batch
                if not zoo_eval and labels is not None and "loss_tensor" in res:
                    print(f"\rBatch {batch_count}: SGD client update, loss={res['loss']:.4f}", end='', flush=True)
                    
                    # Local client-prefix update using SGD
                    optimizer.zero_grad(set_to_none=True)
                    res["loss_tensor"].backward()  # This populates grads on kv_model.client_kv only
                    optimizer.step()
                    
                    # Clear grads
                    for p in kv_model.client_kv.parameters():
                        if p.grad is not None:
                            p.grad = None
                    
                batch_count += 1
                
            elif server_data.get('type') == 'eval_request':
                # Handle evaluation request from server
                payload = server_data['data']
                h_cut = torch.as_tensor(payload['h_cut'])
                attention_mask = torch.as_tensor(payload['attention_mask'])
                labels = None if payload['labels'] is None else torch.as_tensor(payload['labels'])
                cut = int(payload['meta']['cut_layer'])
                
                # Run evaluation forward pass
                res = _client_forward_from_cut(
                    kv_model,
                    h_cut=h_cut,
                    attention_mask=attention_mask,
                    labels=labels,
                    cut=cut,
                    compute_g_cut=False,  # No gradients needed for eval
                    return_loss_tensor=False
                )
                
                # Send back evaluation loss
                client.send_data({"eval_loss": res["loss"]})
                
        except Exception as e:
            print(f"⌠ERROR IN CLIENT SGD TRAINING: {e}")
            traceback.print_exc()
            break


def parse_args():
    parser = argparse.ArgumentParser(description='Split Learning LLM Client')
    parser.add_argument('--model_name', type=str, default='facebook/opt-125m', help='Pretrained model name')
    parser.add_argument('--num_prefix', type=int, default=5, help='Number of prefix tokens')
    parser.add_argument('--cut_layer', type=int, default=6, help='Split index: 0..cut-1 server; cut..L-1 client')
    parser.add_argument('--max_length', type=int, default=512, help='Maximum sequence length')
    parser.add_argument('--seed', type=int, default=42, help='Random seed')
    parser.add_argument('--lr', type=float, default=1e-3, help='Learning rate')
    parser.add_argument('--zoo_lr', type=float, default=1e-3, help='Learning rate')
    parser.add_argument('--momentum', type=float, default=0.9, help='SGD momentum')
    parser.add_argument('--train_batch_size', type=int, default=4, help='Training batch size')
    parser.add_argument('--test_batch_size', type=int, default=4, help='Test batch size')
    parser.add_argument('--max_steps', type=int, default=4000, help='Maximum training steps')
    parser.add_argument('--epochs', type=int, default=10, help='Number of epochs (fallback)')
    # ZOO parameters
    parser.add_argument('--mu', type=float, default=1e-1, help='ZOO perturbation scale')
    parser.add_argument('--num_pert', type=int, default=5, help='Number of ZOO perturbations')
    parser.add_argument('--use_zeroth_order_client', action='store_true', help='Use ZOO for client')
    return parser.parse_args()


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
    
    print("Creating client KV-prefix model...")
    # KV prefix model (new): always create so handlers can access
    from transformers import AutoConfig
    tmp_cfg = AutoModelForCausalLM.from_pretrained(args.model_name, torch_dtype=torch.float32, device_map=None).config
    total_layers = tmp_cfg.num_hidden_layers
    from prefix_kv import PrefixKV
    kv_model = ClientKVModel(args.model_name, total_layers, args.cut_layer, num_prefix=args.num_prefix).to(device)
    trainable_params = list(kv_model.client_kv.parameters())

    
    print(f"Model loaded: {args.model_name}")
    print(f"Client owns layers [{args.cut_layer}, {total_layers-1}] with {args.num_prefix} prefix tokens each")
    
    # Setup optimizer for client's prefix embeddings
    if args.use_zeroth_order_client:
        optimizer = optim.SGD(kv_model.client_kv.parameters(), lr=args.zoo_lr, momentum=0.0)
        print(f"Client using ZOO optimizer with lr={args.zoo_lr}")
    else:
        optimizer = optim.SGD(kv_model.client_kv.parameters(), lr=args.lr, momentum=args.momentum)
        print(f"Client using SGD optimizer with lr={args.lr}, momentum={args.momentum}")
    
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
        print("Trying to connect to server at localhost:12345...")
        
        client = Client('localhost', 12345)
        
        print("=" * 60)
        print("CLIENT SUCCESSFULLY CONNECTED TO SERVER!")
        print("=" * 60)

        client_config = {
            'model_name': args.model_name,
            'num_prefix': args.num_prefix,
            'lr': args.lr,
            'use_zeroth_order_client': args.use_zeroth_order_client
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
        
        # Choose handler based on optimization method
        if args.use_zeroth_order_client:
            handle_training_requests_zoo(
                kv_model, optimizer, grad_estimator, 
                client, device, args
            )
        else:
            handle_training_requests_sgd(
                kv_model, optimizer, client, device, args
            )
        
    except ConnectionRefusedError:
        print("=" * 60)
        print("⌠CONNECTION FAILED!")
        print("=" * 60)
        print("Could not connect to server at localhost:12345")
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