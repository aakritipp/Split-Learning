
import torch
import torch.nn as nn

def _infer_heads_and_dim(config):
    # OPT-style configs usually have hidden_size and num_attention_heads
    n_heads = getattr(config, "num_attention_heads", None)
    hidden = getattr(config, "hidden_size", None)
    if n_heads is None or hidden is None:
        raise ValueError("Config must have num_attention_heads and hidden_size")
    if hidden % n_heads != 0:
        raise ValueError(f"hidden_size {hidden} not divisible by num_attention_heads {n_heads}")
    head_dim = hidden // n_heads
    return n_heads, head_dim, hidden

class PrefixKV(nn.Module):
    """
    Trainable per-layer key/value prefixes (Li & Liang '21 style, simplified):
    We directly parameterize K/V tensors for each local layer:
      K, V shapes: [num_local_layers, num_heads, num_prefix, head_dim]
    At runtime we expand to batch and return a list of (k, v) per local layer.
    """
    def __init__(self, config, layer_indices, num_prefix=10, init_std=0.02, device=None, dtype=torch.float32):
        super().__init__()
        self.layer_indices = list(layer_indices)  # absolute layer indices these params belong to
        self.num_local_layers = len(self.layer_indices)
        self.num_prefix = int(num_prefix)
        n_heads, head_dim, hidden = _infer_heads_and_dim(config)

        self.n_heads = n_heads
        self.head_dim = head_dim
        self.hidden_size = hidden

        # Parameters: [L_local, H, P, D]
        k = torch.zeros(self.num_local_layers, n_heads, self.num_prefix, head_dim, dtype=dtype, device=device)
        v = torch.zeros(self.num_local_layers, n_heads, self.num_prefix, head_dim, dtype=dtype, device=device)
        # Xavier-normal is common here
        nn.init.normal_(k, mean=0.0, std=init_std)
        nn.init.normal_(v, mean=0.0, std=init_std)

        self.k = nn.Parameter(k)
        self.v = nn.Parameter(v)

    @torch.no_grad()
    def set_requires_grad(self, flag: bool):
        self.k.requires_grad_(flag)
        self.v.requires_grad_(flag)

    def num_params(self) -> int:
        return self.k.numel() + self.v.numel()

    def get_local_past(self, batch_size: int):
        """
        Returns a dict: abs_layer_idx -> (k, v) where tensors are shaped [B, H, P, D]
        """
        # Expand first dim to batch
        # current shapes: [L_local, H, P, D] -> [L_local, B, H, P, D]
        k = self.k.unsqueeze(1).expand(-1, batch_size, -1, -1, -1)  # [L, B, H, P, D]
        v = self.v.unsqueeze(1).expand(-1, batch_size, -1, -1, -1)  # [L, B, H, P, D]

        # Rearrange to expected HF past: [B, H, P, D]
        # We'll map by layer index
        out = {}
        for li in range(self.num_local_layers):
            out[self.layer_indices[li]] = (k[li], v[li])
        return out

def merge_past_key_values(total_layers, *layer_prefix_dicts):
    """
    Merge multiple {abs_layer_idx -> (k,v)} dicts into a list length = total_layers
    Each element is (k, v) where k/v are [B, H, P, D] or None if not provided.
    """
    merged = [None] * total_layers
    for d in layer_prefix_dicts:
        if d is None:
            continue
        for li, kv in d.items():
            merged[li] = kv
    # Convert to the tuple list HF expects: each item is (k, v) with shape [B, H, P, D]
    return merged

def flatten_grad_state(prefix_kv: "PrefixKV") -> dict:
    """Return a compact grad state dict that mirrors parameters."""
    state = {}
    if prefix_kv.k.grad is not None:
        state["k"] = prefix_kv.k.grad.detach().cpu()
    if prefix_kv.v.grad is not None:
        state["v"] = prefix_kv.v.grad.detach().cpu()
    return state

def load_grad_state_into(prefix_kv: "PrefixKV", grad_state: dict, device=None):
    """Load grad tensors into .grad fields of parameters (in-place)."""
    device = device or prefix_kv.k.device
    with torch.no_grad():
        if "k" in grad_state:
            gk = grad_state["k"].to(device=device, dtype=prefix_kv.k.dtype)
            if prefix_kv.k.shape != gk.shape:
                raise ValueError(f"k grad shape mismatch: got {tuple(gk.shape)} expected {tuple(prefix_kv.k.shape)}")
            prefix_kv.k.grad = gk.clone()
        else:
            prefix_kv.k.grad = torch.zeros_like(prefix_kv.k, device=device)

        if "v" in grad_state:
            gv = grad_state["v"].to(device=device, dtype=prefix_kv.v.dtype)
            if prefix_kv.v.shape != gv.shape:
                raise ValueError(f"v grad shape mismatch: got {tuple(gv.shape)} expected {tuple(prefix_kv.v.shape)}")
            prefix_kv.v.grad = gv.clone()
        else:
            prefix_kv.v.grad = torch.zeros_like(prefix_kv.v, device=device
