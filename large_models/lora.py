import torch
from torch import nn
from torch.nn import functional as F
from typing import Iterable, List, Optional, Sequence, Tuple
import math 

class LoRALinear(nn.Module):
    def __init__(self, in_features: int, out_features: int,
                 *, r: int = 8, alpha: int = 16, dropout: float = 0.0, bias: bool = True,
                 base_weight: torch.Tensor = None, base_bias: torch.Tensor = None):
        super().__init__()
        self.in_features  = in_features
        self.out_features = out_features
        self.r = int(r)
        self.alpha = float(alpha)
        self.scaling = self.alpha / max(self.r, 1)
        self.dropout = nn.Dropout(dropout) if dropout and dropout > 0 else nn.Identity()

        # frozen base weight/bias
        assert base_weight is not None, "base_weight required"
        self.weight = nn.Parameter(base_weight.detach().clone(), requires_grad=False)
        self.bias = None
        if bias:
            assert base_bias is not None, "base_bias required for bias=True"
            self.bias = nn.Parameter(base_bias.detach().clone(), requires_grad=False)

        # LoRA factors
        if self.r > 0:
            self.lora_A = nn.Parameter(torch.zeros(self.r, in_features))
            self.lora_B = nn.Parameter(torch.zeros(out_features, self.r))
            nn.init.kaiming_uniform_(self.lora_A, a=5 ** 0.5)
            nn.init.zeros_(self.lora_B)
        else:
            self.register_parameter("lora_A", None)
            self.register_parameter("lora_B", None)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # base
        y = x @ self.weight.T
        if self.bias is not None:
            y = y + self.bias
        # lora
        if self.r > 0:
            y = y + self.scaling * (self.dropout(x) @ self.lora_A.T @ self.lora_B.T)
        return y

    @property
    def lora_parameters(self) -> Iterable[nn.Parameter]:
        if self.r > 0:
            yield self.lora_A
            yield self.lora_B

# _LAYER_RE = re.compile(r".*decoder\.layers\.(\d+)\.")

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
        tokens = qname.split(".")
        parent = model
        for t in tokens[:-1]:
            parent = getattr(parent, t)
        attr = tokens[-1]

        new = LoRALinear(
            lin.in_features, lin.out_features,
            r=r, alpha=lora_alpha, dropout=lora_dropout, bias=(lin.bias is not None),
            base_weight=lin.weight, base_bias=lin.bias
        ).to(lin.weight.device)
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
