# import logging

# logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
# logger = logging.getLogger(__name__)
# logger.setLevel(logging.INFO)

# import torch
# from torch import nn
# from torch.nn import functional as F
# import math

# def find_module(root_module: nn.Module, key: str):
#     """
#     Find a module with a specific name in a Transformer model
#     From OpenDelta https://github.com/thunlp/OpenDelta
#     """
#     sub_keys = key.split(".")
#     parent_module = root_module
#     for sub_key in sub_keys[:-1]:
#         parent_module = getattr(parent_module, sub_key)
#     module = getattr(parent_module, sub_keys[-1])
#     return parent_module, sub_keys[-1], module


# class LoRALinear(nn.Linear):
#     """
#     LoRA implemented in a dense layer
#     From https://github.com/microsoft/LoRA/blob/main/loralib/layers.py
#     """
#     def __init__(
#         self, 
#         in_features: int, 
#         out_features: int, 
#         r: int = 0, 
#         lora_alpha: int = 1, 
#         lora_dropout: float = 0.,
#         fan_in_fan_out: bool = False, # Set this to True if the layer to replace stores weight like (fan_in, fan_out)
#         merge_weights: bool = False, # Not sure if this will affect saving/loading models so just set it to be False
#         **kwargs
#     ):
#         nn.Linear.__init__(self, in_features, out_features, **kwargs)

#         self.r = r
#         self.lora_alpha = lora_alpha
#         # Optional dropout
#         if lora_dropout > 0.:
#             self.lora_dropout = nn.Dropout(p=lora_dropout)
#         else:
#             self.lora_dropout = lambda x: x
#         # Mark the weight as unmerged
#         self.merged = False
#         self.merge_weights = merge_weights
#         self.fan_in_fan_out = fan_in_fan_out
#         # Actual trainable parameters
#         if r > 0:
#             self.lora_A = nn.Parameter(self.weight.new_zeros((r, in_features)))
#             self.lora_B = nn.Parameter(self.weight.new_zeros((out_features, r)))
#             self.scaling = self.lora_alpha / self.r
#             # Freezing the pre-trained weight matrix
#             self.weight.requires_grad = False
#         self.reset_parameters()
#         if fan_in_fan_out:
#             self.weight.data = self.weight.data.transpose(0, 1)

#     def reset_parameters(self):
#         nn.Linear.reset_parameters(self)
#         if hasattr(self, 'lora_A'):
#             # initialize A the same way as the default for nn.Linear and B to zero
#             nn.init.kaiming_uniform_(self.lora_A, a=math.sqrt(5))
#             nn.init.zeros_(self.lora_B)

#     def train(self, mode: bool = True):
#         def T(w):
#             return w.transpose(0, 1) if self.fan_in_fan_out else w
#         nn.Linear.train(self, mode)
#         if mode:
#             if self.merge_weights and self.merged:
#                 # Make sure that the weights are not merged
#                 if self.r > 0:
#                     self.weight.data -= T(self.lora_B @ self.lora_A) * self.scaling
#                 self.merged = False
#         else:
#             if self.merge_weights and not self.merged:
#                 # Merge the weights and mark it
#                 if self.r > 0:
#                     self.weight.data += T(self.lora_B @ self.lora_A) * self.scaling
#                 self.merged = True       

#     def forward(self, x: torch.Tensor):
#         def T(w):
#             return w.transpose(0, 1) if self.fan_in_fan_out else w
#         if self.r > 0 and not self.merged:
#             result = F.linear(x, T(self.weight), bias=self.bias)
#             if self.r > 0:
#                 result += (self.lora_dropout(x) @ self.lora_A.transpose(0, 1) @ self.lora_B.transpose(0, 1)) * self.scaling
#             return result
#         else:
#             return F.linear(x, T(self.weight), bias=self.bias)


# class LoRA:

#     def __init__(self, model, r, alpha, float16):
#         """
#         Input:
#         r, alpha: LoRA hyperparameters
#         float16: Whether the model parameters are float16 or not
#         """

#         self.model = model
#         self.hidden_dim = model.config.hidden_size
#         self.float16 = float16

#         if model.config.model_type == "opt":
#             attention_name = "attn"
#         elif model.config.model_type == "roberta":
#             attention_name = "attention"
#         else:
#             raise NotImplementedError

#         # Insert LoRA
#         for key, _ in model.named_modules():
#             if key[-len(attention_name):] == attention_name:
#                 logger.info(f"Inject lora to: {key}")
#                 _, _, attn = find_module(model, key)

#                 if model.config.model_type == "opt":
#                     original_q_weight = attn.q_proj.weight.data
#                     original_q_bias = attn.q_proj.bias.data
#                     original_v_weight= attn.v_proj.weight.data
#                     original_v_bias = attn.v_proj.bias.data
#                     attn.q_proj = LoRALinear(model.config.hidden_size, model.config.hidden_size, r=r, lora_alpha=alpha, bias=model.config.enable_bias).to(original_q_weight.device)
#                     attn.v_proj = LoRALinear(model.config.hidden_size, model.config.hidden_size, r=r, lora_alpha=alpha, bias=model.config.enable_bias).to(original_v_weight.device)
#                     if float16:
#                         attn.q_proj.half()
#                         attn.v_proj.half()
#                     attn.q_proj.weight.data = original_q_weight 
#                     attn.q_proj.bias.data = original_q_bias
#                     attn.v_proj.weight.data = original_v_weight
#                     attn.v_proj.bias.data = original_v_bias
#                 else:
#                     raise NotImplementedError
        
#         # Freeze non-LoRA parameters
#         for n, p in model.named_parameters():
#             if "lora" not in n:
#                 p.requires_grad = False

# lora.py
import torch
from torch import nn
from torch.nn import functional as F
from typing import Iterable, List, Optional, Sequence, Tuple
import math 

# class LoRALinear(nn.Module):
#     """
#     y = xW^T + b + (alpha/r) * ( (x @ A^T) @ B^T )
#     where A: [r, in_features], B: [out_features, r]
#     Only A and B are trainable.
#     """
#     def __init__(self, base_linear: nn.Linear, r: int = 8, alpha: int = 16, dropout: float = 0.0):
#         super().__init__()
#         assert isinstance(base_linear, nn.Linear)
#         self.in_features  = base_linear.in_features
#         self.out_features = base_linear.out_features
#         self.r = int(r)
#         self.alpha = float(alpha)
#         self.scaling = self.alpha / max(self.r, 1)
#         self.dropout = nn.Dropout(dropout) if dropout and dropout > 0 else nn.Identity()

#         # Keep the frozen base weight/bias
#         self.weight = nn.Parameter(base_linear.weight.detach().clone(), requires_grad=False)
#         self.bias   = None
#         if base_linear.bias is not None:
#             self.bias = nn.Parameter(base_linear.bias.detach().clone(), requires_grad=False)

#         # LoRA factors
#         if self.r > 0:
#             self.lora_A = nn.Parameter(torch.zeros(self.r, self.in_features))
#             self.lora_B = nn.Parameter(torch.zeros(self.out_features, self.r))
#             # init: A ~ N(0, 1/r), B = 0 => start as identity (no delta)
#             nn.init.normal_(self.lora_A, std=1.0 / self.r)
#             nn.init.zeros_(self.lora_B)
#         else:
#             # rank 0 -> disabled
#             self.register_parameter("lora_A", None)
#             self.register_parameter("lora_B", None)

#     def forward(self, x: torch.Tensor) -> torch.Tensor:
#         out = F.linear(x, self.weight, self.bias)
#         if self.r > 0:
#             xA = F.linear(self.dropout(x), self.lora_A)         # [B,S,r]
#             out = out + self.scaling * F.linear(xA, self.lora_B) # [B,S,H]
#         return out

class LoRALinear(nn.Module):
    def __init__(self,
                 in_features: int,
                 out_features: int,
                 *,
                 r: int = 8,
                 alpha: int = 16,
                 dropout: float = 0.0,
                 bias: bool = True):
        super().__init__()
        self.in_features  = in_features
        self.out_features = out_features
        self.r = int(r)
        self.alpha = float(alpha)
        self.scaling = self.alpha / max(self.r, 1)
        self.dropout = nn.Dropout(dropout) if dropout and dropout > 0 else nn.Identity()

        # frozen base weight/bias
        self.weight = nn.Parameter(base_linear.weight.detach().clone(), requires_grad=False)
        self.bias = nn.Parameter(base_linear.bias.detach().clone(), requires_grad=False) if base_linear.bias is not None else None

        # LoRA factors
        if self.r > 0:
            self.lora_A = nn.Parameter(torch.zeros(self.r, in_features))
            self.lora_B = nn.Parameter(torch.zeros(out_features, self.r))
            nn.init.normal_(self.lora_A, std=1.0 / self.r)
            nn.init.zeros_(self.lora_B)
        else:
            self.register_parameter("lora_A", None)
            self.register_parameter("lora_B", None)

    def forward(self, x):
        out = F.linear(x, self.weight, self.bias)
        if self.r > 0:
            xA = F.linear(self.dropout(x), self.lora_A)   # [*, r] = x @ A^T
            lora = F.linear(xA, self.lora_B)              # [*, out] = (xA) @ B^T
            out = out + (self.alpha / max(self.r, 1)) * lora
        return out

    @property
    def lora_parameters(self) -> Iterable[nn.Parameter]:
        if self.r > 0:
            yield self.lora_A
            yield self.lora_B

def _layer_index_from_qname(qname: str) -> Optional[int]:
    toks = qname.split(".")
    for i, t in enumerate(toks):
        if t == "layers" and i + 1 < len(toks):
            try:
                return int(toks[i + 1])
            except Exception:
                return None
    return None


def apply_lora_to_opt(model: nn.Module,
                      targets: Sequence[str] = ("q_proj","v_proj"),
                      layer_range: Optional[Tuple[int,int]] = None,
                      r: int = 8, lora_alpha: int = 16, lora_dropout: float = 0.0):
    """
    Wrap OPT-like Linear modules whose qualified-name tail matches any in `targets`.
    Only wrap layers within `layer_range=(start,end)` if provided.
    Also freezes all non-LoRA params.
    """
    to_swap = []
    for qname, mod in model.named_modules():
        if isinstance(mod, nn.Linear):
            tail = qname.split(".")[-1]
            if tail in targets:
                li = _layer_index_from_qname(qname)
                if (layer_range is None) or (li is not None and layer_range[0] <= li <= layer_range[1]):
                    to_swap.append((qname, mod))

    # swap in-place
    for qname, lin in to_swap:
        toks = qname.split(".")
        parent = model
        for t in toks[:-1]:
            parent = getattr(parent, t)
        attr = toks[-1]
        # rebuild LoRALinear with same in/out and bias
        # new = LoRALinear(lin.in_features, lin.out_features,
        #                  r=r, lora_alpha=lora_alpha, lora_dropout=lora_dropout,
        #                  bias=(lin.bias is not None)).to(lin.weight.device)
        # Developer sanity: ensure apply_lora_to_opt calls match the constructor shape
        import inspect
        sig = inspect.signature(LoRALinear.__init__)
        if "base_linear" in sig.parameters:
            _LORA_CONSTRUCTOR = "base"
        else:
            _LORA_CONSTRUCTOR = "dims"

        if _LORA_CONSTRUCTOR == "base":
            new = LoRALinear(base_linear=lin, r=r, alpha=lora_alpha, dropout=lora_dropout).to(lin.weight.device)
        else:
            new = LoRALinear(lin.in_features, lin.out_features, r=r, alpha=lora_alpha,
                            dropout=lora_dropout, bias=(lin.bias is not None)).to(lin.weight.device)
            with torch.no_grad():
                new.weight.copy_(lin.weight)
                if lin.bias is not None and new.bias is not None:
                    new.bias.copy_(lin.bias)

        # load frozen base weight/bias
        with torch.no_grad():
            new.weight.copy_(lin.weight)
            if lin.bias is not None and new.bias is not None:
                new.bias.copy_(lin.bias)
        setattr(parent, attr, new)

    # freeze non-LoRA
    for n, p in model.named_parameters():
        if ("lora_A" in n) or ("lora_B" in n):
            p.requires_grad = True
        else:
            p.requires_grad = False

def iter_lora_parameters(model: nn.Module,
                         layer_range: Optional[Tuple[int,int]] = None) -> Iterable[nn.Parameter]:
    """
    Yield only LoRA parameters, optionally filtered by decoder layer index.
    """
    for n, p in model.named_parameters():
        if ("lora_A" in n) or ("lora_B" in n):
            if layer_range is None:
                yield p
            else:
                li = _layer_index_from_qname(n)
                if li is not None and layer_range[0] <= li <= layer_range[1]:
                    yield p
