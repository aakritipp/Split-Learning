import socket
import pickle
import torch
from torch import nn
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from transformers import AutoTokenizer, AutoModelForCausalLM
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
from prefix_kv import PrefixKV, load_grad_state_into
from dataset import build_task_datasets
from metrics import compute_metrics_for_task

# Ensure merge_past_key_values is available for prefix-aware eval
try:
    from prefix_kv import merge_past_key_values
except Exception:
    merge_past_key_values = None

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

def _neg_inf(dtype: torch.dtype) -> float:
    # Use the representable minimum as the additive mask value
    return torch.finfo(dtype).min
def _refresh_eval_prefixes(full_model, server_model, trainer):
    """
    Pull the latest client prefix snapshot for eval,
    attach live server prefixes, and enable prefix-aware eval.
    Safe fallback to legacy eval if anything fails.
    """
    full_model.attach_live_server_kv(server_model.kv)
    try:
        trainer.send_data({"type": "get_client_kv_state"})
        resp = trainer.receive_data()
        if isinstance(resp, dict) and resp.get("type") == "client_kv_state":
            full_model.load_client_kv_state(resp["state"])
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


def _resolve_base_lm(model_like):
    def _looks_like_hf_lm(x):
        return hasattr(x, "model") and hasattr(x.model, "decoder") and hasattr(x, "lm_head")
    obj = model_like
    for _ in range(6):
        if _looks_like_hf_lm(obj): return obj
        for name in ("base_model","hf_model","model","module","net","inner","wrapped","lm"):
            if hasattr(obj, name):
                obj = getattr(obj, name); break
        else: break
    if _looks_like_hf_lm(model_like): return model_like
    raise AttributeError(f"Could not resolve HF LM from {type(model_like).__name__}")

import torch
import torch.nn.functional as F

def _right_trim(input_ids, attention_mask, labels):
    with torch.no_grad():
        seq_lens = attention_mask.sum(dim=1)
        max_len = int(seq_lens.max().item())
    return (
        input_ids[:, :max_len],
        attention_mask[:, :max_len],
        labels[:, :max_len] if labels is not None else None,
    )

def _right_trim_inputs(input_ids, attention_mask):
    # Keep only non-pad prefix per row (helps generation behave)
    lengths = attention_mask.sum(dim=1)
    max_len = int(lengths.max().item())
    return input_ids[:, :max_len], attention_mask[:, :max_len]
    
def _server_forward_to_cut_payload(
    server_wrap,                    # ServerKVOnly instance
    input_ids: torch.Tensor,
    attention_mask: torch.Tensor,
    labels: torch.Tensor,
    send_fp16: bool = True,):
    """
    Run embeddings + layers [0..cut-1] with server prefixes as per-layer past_key_value to produce h_cut.
    Returns (h_cut_live, payload_to_client).
    """
    base_model = server_wrap.base_model
    decoder    = base_model.model.decoder
    cut        = server_wrap.cut_layer
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
                p = float(getattr(getattr(server_wrap.base_model, "config", object()), "dropout", 0.0))
        if p and p > 0.0:
            x = F.dropout(
                x,
                p=p,
                training=decoder.training if hasattr(decoder, "training") else server_wrap.base_model.training,
            )
        # else: no-op if p == 0.0


    # 4D attn mask
    # attn_mask_4d = decoder._prepare_decoder_attention_mask(
    #     attention_mask, (x.shape[0], x.shape[1]), x, 0
    # )
    # Determine prefix length P from your server PrefixKV
    try:
        prefix_len = int(server_wrap.kv.k.shape[-2])  # [L, H, P, D] -> P
    except Exception:
        prefix_len = 0

    tgt_len = x.shape[1]
    attn_mask_4d = _build_self_attn_mask(
        attention_mask=attention_mask,
        tgt_len=tgt_len,
        prefix_len=prefix_len,
        dtype=x.dtype,
        device=x.device,
    )

    # Build per-layer past_kv from server prefixes
    bsz = input_ids.size(0)
    server_past = server_wrap.kv.get_local_past(bsz)  # {layer_idx: (k,v)} with [B,H,P,D]

    # Run server side layers [0..cut-1]
    for li in range(cut):
        layer = decoder.layers[li]
        pkv   = server_past.get(li, None)
        # pkv = server_past.get(li, None)
        if pkv is not None:
            # pkv is (k,v) shaped [B,H,P,D] each — wrap to cache-like object
            pkv = _PrefixConcatCache(pkv[0], pkv[1])

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


class ServerKVOnly(nn.Module):
    """
    Minimal server-side holder for KV prefixes on the first `cut_layer` layers.
    (No forward compute here to keep changes minimal; client runs the full model and uses these prefixes.)
    """
    def __init__(self, model_name, cut_layer, num_prefix=10):
        super().__init__()
        # load config to size params correctly
        tmp = AutoModelForCausalLM.from_pretrained(model_name, torch_dtype=torch.float32, device_map=None)
        self.total_layers = tmp.config.num_hidden_layers
        self.cut_layer = cut_layer
        self.kv = PrefixKV(tmp.config, list(range(0, cut_layer)), num_prefix=num_prefix, device=tmp.device)
        self.attach_partial_model(model_name)
        # we do not keep the full model in memory here to save RAM
        del tmp
        torch.cuda.empty_cache()
    
    def attach_partial_model(self, model_name: str):
        """
        Load an OPT-style LM and keep only embeddings + first `cut_layer` decoder blocks.
        Enough to produce h_cut on the server.
        """
        from transformers import AutoModelForCausalLM
        import torch.nn as nn

        base = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype=torch.float32,
            device_map=None
        )
        dec = base.model.decoder
        # keep only [0..cut_layer-1]
        dec.layers = nn.ModuleList(list(dec.layers[: self.cut_layer]))
        self.base_model = base.eval()

    def state_dict_kv(self):
        # minimal state dict to send to client
        return {"k": self.kv.k.detach().cpu(), "v": self.kv.v.detach().cpu()}

from metrics import (
    calculate_squad_metrics,
    calculate_generation_f1_em, 
    test_generation_simple,
    normalize_answer,
    squad_f1_score,
    squad_exact_match
)


def squad_collate_fn(batch):
    """Custom collate function for SQUAD dataset with mixed data types"""
    try:
        # Separate tensor and non-tensor data
        input_ids = []
        attention_masks = []
        labels = []
        formatted_texts = []
        original_examples = []
        
        for item in batch:
            input_ids.append(item['input_ids'])
            attention_masks.append(item['attention_mask'])
            labels.append(item['labels'])
            
            # Handle optional fields
            if 'formatted_text' in item:
                formatted_texts.append(item['formatted_text'])
            else:
                formatted_texts.append("")  # Default empty string
                
            if 'original_example' in item:
                original_examples.append(item['original_example'])
            else:
                original_examples.append({})  # Default empty dict
        
        # Stack tensors
        batch_dict = {
            'input_ids': torch.stack(input_ids),
            'attention_mask': torch.stack(attention_masks),
            'labels': torch.stack(labels),
        }
        
        # Add non-tensor data only if we have valid data
        if any(text for text in formatted_texts):
            batch_dict['formatted_text'] = formatted_texts
            
        if any(ex for ex in original_examples):
            batch_dict['original_example'] = original_examples
        
        return batch_dict
        
    except Exception as e:
        print(f"Collate function error: {e}")
        print(f"Batch info: {len(batch)} items")
        for i, item in enumerate(batch):
            print(f"  Item {i}: keys={list(item.keys())}, shapes={[item[k].shape if torch.is_tensor(item[k]) else type(item[k]) for k in item.keys()]}")
        raise


def safe_get_hf_tokenizer(model_name):
    """Safe tokenizer loading with error handling"""
    try:
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
            tokenizer.padding_side = "left"
        return tokenizer
    except Exception as e:
        print(f"Failed to load tokenizer for {model_name}: {e}")
        raise


class TextDataset(Dataset):
    """Simple text dataset for demonstration purposes"""
    def __init__(self, texts, tokenizer, max_length=512):
        self.texts = texts
        self.tokenizer = tokenizer
        self.max_length = max_length
    
    def __len__(self):
        return len(self.texts)
    
    def __getitem__(self, idx):
        text = self.texts[idx]
        
        try:
            encoding = self.tokenizer(
                text,
                truncation=True,
                padding='max_length',
                max_length=self.max_length,
                return_tensors='pt'
            )
            
            input_ids = encoding['input_ids'].squeeze()
            attention_mask = encoding['attention_mask'].squeeze()
            
            return {
                'input_ids': input_ids,
                        'server_kv_state': server_model.state_dict_kv(),
                        'cut_layer': args.cut_layer,
                        'server_kv_state': server_model.state_dict_kv(),
                        'cut_layer': args.cut_layer,
                'attention_mask': attention_mask,
                'labels': input_ids.clone()
            }
        except Exception as e:
            print(f"Error processing text at index {idx}: {e}")
            # Return a dummy sample
            dummy_ids = torch.zeros(self.max_length, dtype=torch.long)
            return {
                'input_ids': dummy_ids,
                'attention_mask': torch.ones_like(dummy_ids),
                'labels': dummy_ids.clone()
            }


def get_squad_dataloaders(args, tokenizer):
    """Create SQUAD dataloaders with custom collate function"""
    print(f"Creating SQUAD dataset with MeZO hyperparameters...")
    print(f"Train examples: {args.train_examples}")
    print(f"Dev examples: {args.dev_examples}")
    print(f"Eval examples: {args.eval_examples}")
    print(f"Batch size: {args.train_batch_size}")
    
    try:
        from datasets import load_dataset
        
        # Load SQUAD dataset
        dataset = load_dataset('squad')
        
        # Use the specified sizes
        train_size = min(args.train_examples, len(dataset['train']))
        dev_size = min(args.dev_examples, len(dataset['validation']))
        eval_size = min(args.eval_examples, len(dataset['validation']))
        
        # Create datasets with specified sizes
        train_dataset = dataset['train'].shuffle(seed=args.seed).select(range(train_size))
        val_dataset = dataset['validation'].shuffle(seed=args.seed)
        dev_dataset = val_dataset.select(range(dev_size))
        eval_dataset = val_dataset.select(range(dev_size, min(dev_size + eval_size, len(val_dataset))))
        
        print(f"Dataset sizes: Train={len(train_dataset)}, Dev={len(dev_dataset)}, Eval={len(eval_dataset)}")
        
        # Format datasets
        def format_squad_example(example):
            context = example['context']
            question = example['question']
            answer = example['answers']['text'][0] if len(example['answers']['text']) > 0 else ""
            return f"Context: {context}\nQuestion: {question}\nAnswer: {answer}"
        
        # Create formatted texts
        train_texts = [format_squad_example(ex) for ex in train_dataset]
        eval_examples = list(eval_dataset)  # Keep original for evaluation
        eval_texts = [format_squad_example(ex) for ex in eval_examples]
        
        # Create datasets
        train_squad_dataset = SQuADDataset(train_texts, tokenizer, args.max_length)
        eval_squad_dataset = SQuADDataset(eval_texts, tokenizer, args.max_length, eval_examples)
        
        # Create dataloaders with custom collate function
        train_loader = DataLoader(
            train_squad_dataset, 
            batch_size=args.train_batch_size, 
            shuffle=True,
            collate_fn=squad_collate_fn  # Use custom collate function
        )
        eval_loader = DataLoader(
            eval_squad_dataset, 
            batch_size=args.test_batch_size, 
            shuffle=False,
            collate_fn=squad_collate_fn  # Use custom collate function
        )
        
        print(f"SQUAD dataloaders created successfully with custom collate function")
        print(f"   Train batches: {len(train_loader)}")
        print(f"   Eval batches: {len(eval_loader)}")
        
        # Test the dataloader
        print("  Testing dataloader...")
        try:
            test_batch = next(iter(train_loader))
            print(f"Dataloader test passed")
            print(f"   Batch keys: {list(test_batch.keys())}")
            print(f"   Batch shapes: {[f'{k}: {v.shape if torch.is_tensor(v) else len(v)}' for k, v in test_batch.items()]}")
        except Exception as test_error:
            print(f"❌ Dataloader test failed: {test_error}")
            raise
        
        return train_loader, eval_loader
        
    except Exception as e:
        print(f"❌ SQUAD dataset creation failed: {e}")
        import traceback
        traceback.print_exc()
        raise


class SQuADDataset(Dataset):
    """SQUAD dataset with consistent output structure"""
    def __init__(self, texts, tokenizer, max_length=512, original_examples=None):
        self.texts = texts
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.original_examples = original_examples or []
    
    def __len__(self):
        return len(self.texts)
    
    def __getitem__(self, idx):
        text = self.texts[idx]
        
        try:
            encoding = self.tokenizer(
                text,
                truncation=True,
                padding='max_length',
                max_length=self.max_length,
                return_tensors='pt'
            )
            
            input_ids = encoding['input_ids'].squeeze()
            attention_mask = encoding['attention_mask'].squeeze()
            
            # Ensure consistent tensor shapes
            if input_ids.dim() == 0:
                input_ids = input_ids.unsqueeze(0)
            if attention_mask.dim() == 0:
                attention_mask = attention_mask.unsqueeze(0)
            
            # For training, labels = input_ids (next token prediction)
            labels = input_ids.clone()
            
            result = {
                'input_ids': input_ids,
                'server_kv_state': server_model.state_dict_kv(),
                'cut_layer': args.cut_layer,
                'attention_mask': attention_mask,
                'labels': labels
            }
            
            # Add optional fields consistently
            result['formatted_text'] = text  # Always include text
            
            # Add original example if available
            if idx < len(self.original_examples):
                result['original_example'] = self.original_examples[idx]
            else:
                result['original_example'] = {}  # Empty dict as placeholder
            
            return result
            
        except Exception as e:
            print(f"❌ Error processing item {idx}: {e}")
            # Return a dummy item with consistent structure
            dummy_ids = torch.zeros(self.max_length, dtype=torch.long)
            return {
                'input_ids': dummy_ids,
                'attention_mask': torch.ones_like(dummy_ids),
                'labels': dummy_ids.clone(),
                'formatted_text': "",
                'original_example': {}
            }


class PrefixEncoder(nn.Module):
    """Prefix encoder that creates trainable prefix embeddings"""
    def __init__(self, config, num_prefix=5):
        super(PrefixEncoder, self).__init__()
        self.num_prefix = num_prefix
        self.hidden_size = config.hidden_size
        
        # FIXED: Better initialization - use same as model embeddings
        self.prefix_embeddings = nn.Parameter(
            torch.randn(num_prefix, self.hidden_size) * (self.hidden_size ** -0.5)
        )
        
        # Initialize to match existing embedding statistics
        with torch.no_grad():
            # Initialize with normal distribution similar to model embeddings
            nn.init.normal_(self.prefix_embeddings, mean=0.0, std=0.02)
        
        print(f"  PrefixEncoder: {num_prefix} tokens x {self.hidden_size} dims = {num_prefix * self.hidden_size} parameters")
        print(f"  Prefix embedding std: {self.prefix_embeddings.std().item():.6f}")
    
    def forward(self, batch_size):
        """Expand prefix embeddings for the given batch size"""
        return self.prefix_embeddings.unsqueeze(0).expand(batch_size, -1, -1)


class ServerLLMModel(nn.Module):
    """Server-side model that only trains prefix embeddings"""
    def __init__(self, model_name, num_prefix=5):
        super(ServerLLMModel, self).__init__()
        print(f"Loading server model: {model_name}")
        
        try:
            self.base_model = AutoModelForCausalLM.from_pretrained(
                model_name, 
                torch_dtype=torch.float32,
                device_map=None
            )
            print(f"✅ Base model loaded successfully")
        except Exception as e:
            print(f"❌ Failed to load base model: {e}")
            raise
        
        # Freeze base model parameters - server only trains prefix
        for param in self.base_model.parameters():
            param.requires_grad = False
        
        # Create trainable prefix encoder
        try:
            self.num_prefix = num_prefix
            print(f"Prefix encoder created successfully")
        except Exception as e:
            print(f"❌ Failed to create prefix encoder: {e}")
            raise
        
        print(f"Base parameters: {sum(p.numel() for p in self.base_model.parameters()):,}")
        
    def forward(self, input_ids, attention_mask):
        """Forward pass that combines prefix with input embeddings"""
        try:
            batch_size, seq_len = input_ids.shape
            
            # Get input embeddings from base model
            inputs_embeds = self.base_model.get_input_embeddings()(input_ids)
            
            # Get prefix embeddings for this batch
            
            # Concatenate prefix embeddings with input embeddings
            # Shape: [batch_size, num_prefix + seq_len, hidden_size]
            inputs_embeds = torch.cat([prefix_embeds, inputs_embeds], dim=1)
            
            # Extend attention mask to include prefix tokens
            prefix_mask = torch.ones(batch_size, self.num_prefix, device=attention_mask.device, dtype=attention_mask.dtype)
            attention_mask = torch.cat([prefix_mask, attention_mask], dim=1)
            
            return {
                'inputs_embeds': inputs_embeds,
                'attention_mask': attention_mask
            }
        except Exception as e:
            print(f"❌ Server model forward pass failed: {e}")
            raise

class FullLLMModel(nn.Module):
    """Frozen full model used only for monitoring/evaluation."""
    def __init__(self, model_name, cut_layer, num_prefix=5):
        super(FullLLMModel, self).__init__()
        self.base_model = AutoModelForCausalLM.from_pretrained(
            model_name, torch_dtype=torch.float32, device_map=None
        )
        self.total_layers = self.base_model.config.num_hidden_layers
        self.cut_layer    = int(cut_layer)
        self.num_prefix = num_prefix
        for p in self.base_model.parameters():
            p.requires_grad = False
        self._use_prefix_eval = False
        self._server_kv_live  = None            # live reference (server side)
        self._client_kv_eval  = None            # local PrefixKV snapshot for client side

        print("Full model ready; prefix-aware eval OFF by default (legacy behavior).")

    def attach_live_server_kv(self, server_kv_module: PrefixKV):
        """Point eval model to the live server PrefixKV (no copy)."""
        self._server_kv_live = server_kv_module

    @torch.no_grad()
    def load_client_kv_state(self, state: dict):
        """
        Load a snapshot of client prefixes into a local PrefixKV for eval.
        Expects {'k': tensor[Lc,H,P,D], 'v': tensor[Lc,H,P,D]} for layers [cut..L-1].
        """
        device = next(self.base_model.parameters()).device
        dtype  = next(self.base_model.parameters()).dtype

        if self._client_kv_eval is None:
            # Build eval-side PrefixKV container for client half
            self._client_kv_eval = PrefixKV(
                self.base_model.config,
                list(range(self.cut_layer, self.total_layers)),
                num_prefix=state["k"].shape[-2],
                device=device,
                dtype=dtype,
            )
        # Copy weights
        self._client_kv_eval.k.copy_(state["k"].to(device=device, dtype=self._client_kv_eval.k.dtype))
        self._client_kv_eval.v.copy_(state["v"].to(device=device, dtype=self._client_kv_eval.v.dtype))

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
        if self._use_prefix_eval and self._server_kv_live is not None and self._client_kv_eval is not None:
            bsz, seq_len = input_ids.size(0), input_ids.size(1)

            # Build per-layer K/V caches from both halves (legacy list of tuples)
            server_past = self._server_kv_live.get_local_past(bsz)   # {layer_idx: (k,v)}
            client_past = self._client_kv_eval.get_local_past(bsz)   # {layer_idx: (k,v)}
            legacy_cache = merge_past_key_values(self.total_layers, server_past, client_past)

            # Infer prefix length P from any present layer
            past_len = 0
            for kv in legacy_cache:
                if kv is not None:
                    past_len = kv[0].shape[-2]  # [B,H,P,D]
                    break

            # Use explicit positions (matches client forward_full)
            position_ids = torch.arange(
                past_len, past_len + seq_len, device=input_ids.device, dtype=torch.long
            ).unsqueeze(0).expand(bsz, -1)

            # Convert to HF Cache object if available for newer transformers
            cache_obj = legacy_cache
            try:
                if StaticCache is not None and hasattr(StaticCache, "from_legacy_cache"):
                    cache_obj = StaticCache.from_legacy_cache(tuple(legacy_cache))
                elif DynamicCache is not None and hasattr(DynamicCache, "from_legacy_cache"):
                    cache_obj = DynamicCache.from_legacy_cache(tuple(legacy_cache))
            except Exception:
                cache_obj = legacy_cache

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

    # --- server.py (inside class FullLLMModel) ---
    def generate(self, input_ids, attention_mask=None, **kwargs):
        # Prefix-aware generate (matches your prefix-aware forward)
        if self._use_prefix_eval and self._server_kv_live is not None and self._client_kv_eval is not None:
            bsz, seq_len = input_ids.shape
            server_past = self._server_kv_live.get_local_past(bsz)
            client_past = self._client_kv_eval.get_local_past(bsz)
            legacy_cache = merge_past_key_values(self.total_layers, server_past, client_past)

            # infer prefix length P
            past_len = 0
            for kv in legacy_cache:
                if kv is not None:
                    past_len = kv[0].shape[-2]  # [B,H,P,D]
                    break

            position_ids = torch.arange(past_len, past_len + seq_len, device=input_ids.device, dtype=torch.long)
            position_ids = position_ids.unsqueeze(0).expand(bsz, -1)

            # Let HF generate with the prefilled cache; attention_mask becomes unnecessary
            cache_obj = legacy_cache
            try:
                if StaticCache is not None and hasattr(StaticCache, "from_legacy_cache"):
                    cache_obj = StaticCache.from_legacy_cache(tuple(legacy_cache))
                elif DynamicCache is not None and hasattr(DynamicCache, "from_legacy_cache"):
                    cache_obj = DynamicCache.from_legacy_cache(tuple(legacy_cache))
            except Exception:
                cache_obj = legacy_cache  # fallback

            return self.base_model.generate(
                input_ids=input_ids,
                position_ids=position_ids,
                past_key_values=cache_obj,
                **kwargs
            )

        # Fallback: no prefixes known
        return self.base_model.generate(input_ids=input_ids, attention_mask=attention_mask, **kwargs)

def _recv_exact(sock, n):
    buf = bytearray()
    while len(buf) < n:
        chunk = sock.recv(n - len(buf))
        if not chunk:
            raise EOFError(f"socket closed during recv (wanted {n}, got {len(buf)})")
        buf.extend(chunk)
    return bytes(buf)

# server.py
class Trainer:
    def __init__(self, conn):
        self.conn = conn
        try:
            self.conn.settimeout(300.0)  # ok
        except Exception:
            pass

    def _recvall(self, n: int):
        buf = bytearray()
        while len(buf) < n:
            try:
                chunk = self.conn.recv(n - len(buf))
            except (BlockingIOError, InterruptedError):
                continue
            if not chunk:
                return None  # peer closed / half-open
            buf.extend(chunk)
        return bytes(buf)

    def send_data(self, data) -> bool:
        try:
            serialized = pickle.dumps(data, protocol=pickle.HIGHEST_PROTOCOL)
            self.conn.sendall(len(serialized).to_bytes(4, 'big'))
            self.conn.sendall(serialized)
            return True
        except (BrokenPipeError, ConnectionResetError, OSError, socket.timeout) as e:
            print(f"⚠️ send_data failed: {e}")
            return False

    def receive_data(self):
        try:
            header = self._recvall(4)
            if header is None or len(header) != 4:
                return None
            length = int.from_bytes(header, 'big')
            payload = self._recvall(length)
            if payload is None or len(payload) != length:
                return None
            return pickle.loads(payload)
        except (EOFError, ConnectionResetError, BrokenPipeError, socket.timeout, OSError):
            return None

def calculate_metrics(outputs, labels, batch, tokenizer, model, device):
    """Calculate SQUAD-specific metrics - ROBUST VERSION"""
    try:
        # 1. Calculate standard next-token loss
        loss = outputs.loss.item() if outputs.loss is not None else 0.0
        
        # 2. Calculate token-level accuracy (for monitoring)
        logits = outputs.logits
        if logits.shape[1] != labels.shape[1]:
            min_len = min(logits.shape[1], labels.shape[1])
            logits = logits[:, :min_len, :]
            labels = labels[:, :min_len]
        
        # For next token prediction
        if logits.shape[1] > 1:
            shift_logits = logits[:, :-1, :].contiguous()
            shift_labels = labels[:, 1:].contiguous()
        else:
            shift_logits = logits
            shift_labels = labels
        
        predictions = torch.argmax(shift_logits, dim=-1)
        
        # Calculate answer token accuracy
        answer_accuracy = calculate_answer_token_accuracy(
            predictions, shift_labels, batch, tokenizer
        )
        
        # 3. Calculate F1/EM by generating answers (with error handling)
        f1_score = 0.0
        em_score = 0.0
        
        # Only try generation if we have the required data
        if ('original_example' in batch and 'formatted_text' in batch and 
            len(batch.get('original_example', [])) > 0 and 
            len(batch.get('formatted_text', [])) > 0):
            
            try:
                f1_score, em_score = calculate_generation_f1_em(
                    model, batch, tokenizer, device
                )
            except Exception as gen_error:
                print(f"⚠️ Generation metrics failed: {gen_error}")
                f1_score, em_score = 0.0, 0.0
        
        return loss, answer_accuracy, f1_score, em_score
        
    except Exception as e:
        print(f"❌ SQUAD metrics calculation failed: {e}")
        import traceback
        traceback.print_exc()
        return 0.0, 0.0, 0.0, 0.0


def calculate_answer_token_accuracy(predictions, labels, batch, tokenizer):
    """Calculate accuracy only on answer portion tokens"""
    try:
        if 'formatted_text' not in batch:
            # Fallback to general accuracy
            mask = (labels != -100)
            if mask.sum() == 0:
                return 0.0
            correct = (predictions == labels) & mask
            return correct.sum().float() / mask.sum().float()
        
        # Find answer tokens in each example
        accuracies = []
        for i in range(len(batch['formatted_text'])):
            text = batch['formatted_text'][i]
            
            # Find "Answer:" position
            answer_start = text.find("Answer:")
            if answer_start == -1:
                continue
                
            # Tokenize to find answer token positions
            context_question = text[:answer_start + len("Answer:")]
            answer_part = text[answer_start + len("Answer:"):]
            
            context_tokens = tokenizer.encode(context_question, add_special_tokens=False)
            answer_tokens = tokenizer.encode(answer_part, add_special_tokens=False)
            
            if len(answer_tokens) == 0:
                continue
            
            # Get accuracy for answer tokens only
            start_idx = len(context_tokens)
            end_idx = start_idx + len(answer_tokens)
            
            if end_idx <= predictions.shape[1] and end_idx <= labels.shape[1]:
                answer_preds = predictions[i, start_idx:end_idx]
                answer_labels = labels[i, start_idx:end_idx]
                
                if len(answer_preds) > 0:
                    correct = (answer_preds == answer_labels).sum().item()
                    total = len(answer_preds)
                    accuracies.append(correct / total)
        
        return sum(accuracies) / len(accuracies) if accuracies else 0.0
        
    except Exception as e:
        print(f"❌ Answer token accuracy failed: {e}")
        return 0.0


def squad_f1_score(prediction, ground_truth):
    """Calculate F1 score for SQUAD with better error handling"""
    try:
        from collections import Counter
        import string
        import re
        
        def normalize_answer(s):
            """Normalize answer text"""
            if not isinstance(s, str):
                s = str(s)
            
            def remove_articles(text):
                return re.sub(r'\b(a|an|the)\b', ' ', text)
            def white_space_fix(text):
                return ' '.join(text.split())
            def remove_punc(text):
                exclude = set(string.punctuation)
                return ''.join(ch for ch in text if ch not in exclude)
            def lower(text):
                return text.lower()
            
            return white_space_fix(remove_articles(remove_punc(lower(s))))
        
        # Normalize inputs
        pred_normalized = normalize_answer(prediction)
        truth_normalized = normalize_answer(ground_truth)
        
        prediction_tokens = pred_normalized.split()
        ground_truth_tokens = truth_normalized.split()
        
        # Handle empty cases
        if len(prediction_tokens) == 0 and len(ground_truth_tokens) == 0:
            return 1.0
        if len(prediction_tokens) == 0 or len(ground_truth_tokens) == 0:
            return 0.0
        
        # Calculate overlap
        common = Counter(prediction_tokens) & Counter(ground_truth_tokens)
        num_same = sum(common.values())
        
        if num_same == 0:
            return 0.0
        
        precision = num_same / len(prediction_tokens)
        recall = num_same / len(ground_truth_tokens)
        f1 = (2 * precision * recall) / (precision + recall)
        
        return float(f1)
        
    except Exception as e:
        print(f"❌ F1 calculation error: {e}")
        return 0.0

import torch

def right_trim_batch(input_ids, attention_mask, labels=None):
    """
    Right-trim padding so the model doesn't waste compute on padded tokens.
    Keeps shape-consistency across input_ids / attention_mask / labels.
    """
    # length per row
    lens = attention_mask.sum(dim=-1)
    max_len = int(lens.max().item())
    input_ids = input_ids[:, :max_len]
    attention_mask = attention_mask[:, :max_len]
    if labels is not None:
        labels = labels[:, :max_len]
        return input_ids, attention_mask, labels
    return input_ids, attention_mask, None

def decode_generated_suffix(model, tokenizer, input_ids, attention_mask,
                            max_new_tokens=32, eos_token_id=None):
    """
    Greedy generation, then decode ONLY the new suffix after the prompt length.
    - Works for causal LMs (OPT, etc.)
    - Strips at first newline to avoid rambling
    """
    eos_id = eos_token_id if eos_token_id is not None else tokenizer.eos_token_id
    gen = model.generate(
        input_ids=input_ids,
        attention_mask=attention_mask,
        max_new_tokens=max_new_tokens,
        do_sample=False,
        eos_token_id=eos_id,
        pad_token_id=tokenizer.pad_token_id,
    )
    prompt_lens = attention_mask.sum(dim=1)  # B
    outs = []
    for i, g in enumerate(gen):
        start = int(prompt_lens[i].item())
        suffix_ids = g[start:]
        txt = tokenizer.decode(suffix_ids, skip_special_tokens=True).strip()
        # Be conservative for QA/CLS: take first line/sentence
        txt = txt.split("\n", 1)[0].strip()
        outs.append(txt)
    return outs


@torch.no_grad()
def evaluate_model(model, test_loader, device, tokenizer, args, server_model=None, trainer=None):
    """
    Unified evaluation across tasks using true-split when available.

    Returns (avg_loss, main1, main2, main3) where:
      - For 'cls' (SST-2): (loss, accuracy, 0.0, 0.0)
      - For 'qa'  (SQuAD/DROP): (loss, f1, em, 0.0)
      - For 'sum' (XSum): (loss, rouge1, rouge2, rougeL)
    """
    task = getattr(args, "task", "squad").lower()
    task_type = {"sst2": "cls", "squad": "qa", "drop": "qa", "xsum": "sum"}.get(task, "qa")
    print(f"  Starting evaluation for task='{task}' (type='{task_type}')")

    # Smoke test (okay to fail for non-generative heads)
    try:
        print("  Testing generation capability (local model path)...")
        gen_works = test_generation_simple(model, tokenizer, device)
        print(f"  Generation test result: {'✅ PASS' if gen_works else '❌ FAIL'}")
    except Exception:
        print("  (Skipping generation smoke test for non-generative head.)")

    # Prefix refresh if applicable
    if server_model is not None and trainer is not None:
        _refreshed = _refresh_eval_prefixes(model, server_model, trainer)
        if not _refreshed:
            print("⚠️ Prefix-aware eval unavailable; continuing with current weights.")

    model.eval()
    if server_model is not None:
        server_model.eval()

    total_loss = 0.0
    n_batches  = 0

    # Accumulators
    pred_label_ids, true_label_ids = [], []   # CLS
    pred_texts = []                           # QA/SUM (and CLS fallback)
    all_meta   = []                           # one dict per example
    all_targets = []                          # text_target (used for SUM; ignored for QA)

    for batch_idx, batch in enumerate(tqdm(test_loader, desc=f"{task.upper()} Evaluation")):
        try:
            # --- Uniform batch adapter THEN right-trim ---
            input_ids, attention_mask, labels, prompt_text, text_target, meta = adapt_batch(batch, device)
            input_ids, attention_mask, labels = right_trim_batch(input_ids, attention_mask, labels)

            # ======== TRUE-SPLIT EVAL PATH ========
            if server_model is not None and trainer is not None:
                # Server forward to cut
                h_cut_live, _pkg = _server_forward_to_cut_payload(
                    server_model,
                    input_ids, attention_mask, labels,
                    send_fp16=True
                )
                trainer.send_data({
                    "type": "forward_cut",
                    "mode": "eval",
                    "data": {"h_cut": h_cut_live, "attention_mask": _pkg["attention_mask"], "labels": _pkg["labels"], "cut_layer": _pkg["cut_layer"]},
                    "meta": {
                        "task_type": task_type,
                        "max_new_tokens": getattr(args, "max_new_tokens", 20),
                    }
                })

                resp = trainer.receive_data()
                batch_loss = float(resp.get("loss", 0.0))
                total_loss += batch_loss
                n_batches  += 1

                if task_type == "cls":
                    # Prefer explicit pred_ids; else derive from logits
                    if "pred_ids" in resp:
                        preds = list(resp["pred_ids"])
                    elif "logits" in resp:
                        logits = torch.tensor(resp["logits"])
                        preds = logits.argmax(-1).tolist()
                    else:
                        # Fallback: ask client to return generations; map to labels later
                        preds = None

                    if preds is not None:
                        pred_label_ids.extend(preds)
                        true_label_ids.extend(batch["labels"].tolist())
                    else:
                        # Fallback to text-based mapping if only generations are available
                        gens = [g.strip() for g in resp.get("generations", [])]
                        pred_texts.extend(gens)
                        all_meta.extend(meta)
                        all_targets.extend(text_target)

                else:
                    gens = [g.strip() for g in resp.get("generations", [])]
                    pred_texts.extend(gens)
                    all_meta.extend(meta)
                    all_targets.extend(text_target)

            # ======== LOCAL (MONOLITHIC) EVAL PATH ========
            else:
                if task_type == "cls":
                    # If a classifier head exists:
                    try:
                        logits = model(input_ids=input_ids, attention_mask=attention_mask).logits
                        if isinstance(labels, torch.Tensor) and labels.ndim == 1 and labels.dtype in (torch.long, torch.int64):
                            loss = F.cross_entropy(logits, labels, reduction="mean")
                            total_loss += float(loss.item())
                        else:
                            total_loss += 0.0
                        n_batches += 1
                        preds = logits.argmax(-1).tolist()
                        pred_label_ids.extend(preds)
                        true_label_ids.extend(batch["labels"].tolist())
                    except Exception:
                        # No classifier head → use LM path and text mapping
                        outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels if labels is not None and labels.ndim==2 else None)
                        if hasattr(outputs, "loss") and outputs.loss is not None:
                            total_loss += float(outputs.loss.item())
                        n_batches += 1
                        gens = decode_generated_suffix(
                            model, tokenizer, input_ids, attention_mask,
                            max_new_tokens=getattr(args, "max_new_tokens", 20)
                        )
                        pred_texts.extend(gens)
                        all_meta.extend(meta)
                        all_targets.extend(text_target)

                else:
                    # Generative path (QA/SUM): compute LM loss if labels are provided
                    outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
                    if hasattr(outputs, "loss") and outputs.loss is not None:
                        total_loss += float(outputs.loss.item())
                    else:
                        # Manual CE if needed
                        logits = outputs.logits
                        shift_logits = logits[:, :-1, :].contiguous()
                        shift_labels = labels[:, 1:].contiguous() if labels is not None else None
                        if shift_labels is not None:
                            ce = F.cross_entropy(
                                shift_logits.view(-1, shift_logits.size(-1)),
                                shift_labels.view(-1),
                                ignore_index=-100,
                                reduction="mean",
                            )
                            total_loss += float(ce.item())
                    n_batches += 1

                    # Prompt-aware decoding (no duplicate generation)
                    gens = decode_generated_suffix(
                        model, tokenizer, input_ids, attention_mask,
                        max_new_tokens=getattr(args, "max_new_tokens", 20)
                    )
                    pred_texts.extend([g.strip() for g in gens])
                    all_meta.extend(meta)
                    all_targets.extend(text_target)

            # (Optional) light debug
            if batch_idx < 2:
                print(f"  Batch {batch_idx} ok.")

        except Exception as e:
            print(f"\n⚠️ Error in evaluation batch {batch_idx}: {e}")
            continue

    avg_loss = total_loss / max(n_batches, 1)

    # ---- Aggregate metrics in one place ----
    from metrics import compute_metrics_for_batch, _cls_map_prediction_to_label

    if task_type == "cls":
        if pred_label_ids and true_label_ids:
            acc = float((torch.tensor(pred_label_ids) == torch.tensor(true_label_ids)).float().mean().item())
        else:
            # text-based mapping fallback
            mapped = [_cls_map_prediction_to_label(t) for t in pred_texts]
            acc = float((torch.tensor(mapped) == torch.tensor([m.get("label_id", 0) for m in all_meta])).float().mean().item())
        print(f"\n  {task.upper()} Evaluation Complete:")
        print(f"   Average Loss: {avg_loss:.4f}")
        print(f"   Accuracy:     {acc:.6f}")
        return avg_loss, acc, 0.0, 0.0

    elif task_type == "qa":
        # For QA we want meta["refs"] per example. compute_metrics_for_batch will pull from meta.
        md = compute_metrics_for_batch(
            "qa",
            pred_texts=pred_texts,
            batch_meta=all_meta,
            gold_texts=all_targets  # ignored for QA, kept for signature unity
        )
        f1 = float(md.get("f1", 0.0))
        em = float(md.get("exact_match", md.get("em", 0.0)))
        print(f"\n  {task.upper()} Evaluation Complete:")
        print(f"   Average Loss: {avg_loss:.4f}")
        print(f"   F1:           {f1:.6f}")
        print(f"   EM:           {em:.6f}")
        return avg_loss, f1, em, 0.0

    else:  # "sum"
        # For summarization, use text targets as references
        md = compute_metrics_for_batch(
            "gen",
            pred_texts=pred_texts,
            batch_meta=all_meta,        # not used for GEN
            gold_texts=all_targets      # single-reference summaries
        )
        r1 = float(md.get("rouge1", 0.0))
        r2 = float(md.get("rouge2", 0.0))
        rL = float(md.get("rougeL", 0.0))
        print(f"\n  {task.upper()} Evaluation Complete:")
        print(f"   Average Loss: {avg_loss:.4f}")
        print(f"   ROUGE-1:      {r1:.6f}")
        print(f"   ROUGE-2:      {r2:.6f}")
        print(f"   ROUGE-L:      {rL:.6f}")
        return avg_loss, r1, r2, rL
        
# @torch.no_grad()
# def evaluate_model(model, test_loader, device, tokenizer, args, server_model=None, trainer=None):
#     """
#     Unified evaluation across tasks using true-split when available.

#     Returns (avg_loss, main1, main2, main3) where:
#       - For 'cls' (SST-2): (loss, accuracy, 0.0, 0.0)
#       - For 'qa'  (SQuAD/DROP): (loss, f1, em, 0.0)
#       - For 'sum' (XSum): (loss, rouge1, rouge2, rougeL)
#     """
#     task = getattr(args, "task", "squad").lower()
#     task_type = {"sst2": "cls", "squad": "qa", "drop": "qa", "xsum": "sum"}.get(task, "qa")

#     print(f"  Starting evaluation for task='{task}' (type='{task_type}')")

#     # Quick smoke test for generation capability on local model only
#     try:
#         print("  Testing generation capability (local model path)...")
#         gen_works = test_generation_simple(model, tokenizer, device)
#         print(f"  Generation test result: {'✅ PASS' if gen_works else '❌ FAIL'}")
#     except Exception as _:
#         # Some classifier heads won’t pass a text-gen probe; that’s fine.
#         print("  (Skipping generation smoke test for non-generative head.)")

#     # Prefix refresh if you use that mechanism
#     if server_model is not None and trainer is not None:
#         _refreshed = _refresh_eval_prefixes(model, server_model, trainer)
#         if not _refreshed:
#             print("⚠️ Prefix-aware eval unavailable; continuing with current weights.")

#     model.eval()
#     if server_model is not None:
#         server_model.eval()

#     total_loss = 0.0
#     n_batches  = 0

#     # Accumulators by task type
#     pred_label_ids, true_label_ids = [], []                 # cls
#     pred_texts, ref_texts_list = [], []                     # qa/sum

#     for batch_idx, batch in enumerate(tqdm(test_loader, desc=f"{task.upper()} Evaluation")):
#         try:
#             input_ids      = batch["input_ids"].to(device)
#             attention_mask = batch["attention_mask"].to(device)
#             labels = batch.get("labels", None)
#             if isinstance(labels, torch.Tensor):
#                 labels = labels.to(device)

#             # Right-trim for saner generation
#             input_ids, attention_mask, labels = right_trim_batch(input_ids, attention_mask, labels)
#             input_ids, attention_mask, labels, prompt_text, text_target, meta = adapt_batch(batch, device)

#             # Classification labels (SST-2) vs generative labels (QA/SUM)
            

#             # ======== TRUE-SPLIT EVAL PATH ========
#             if server_model is not None and trainer is not None:
#                 # Server forward to cut
#                 h_cut_live, _pkg = _server_forward_to_cut_payload(
#                     server_model,
#                     input_ids, attention_mask, labels,
#                     send_fp16=True
#                 )

#                 trainer.send_data({
#                     "type": "forward_cut",
#                     "mode": "eval",
#                     "data": {
#                         "h_cut": h_cut_live,                      # already packed by helper
#                         "attention_mask": attention_mask.cpu(),
#                         "labels": labels.cpu() if labels is not None else None,
#                     },
#                     "meta": {
#                         "task_type": task_type,                   # "cls" | "qa" | "sum"
#                         "max_new_tokens": getattr(args, "max_new_tokens", 20),
#                     }
#                 })

#                 resp = trainer.receive_data()
#                 batch_loss = float(resp.get("loss", 0.0))
#                 total_loss += batch_loss
#                 n_batches  += 1

#                 if task_type == "cls":
#                     if "pred_ids" in resp:
#                         preds = resp["pred_ids"]
#                     else:
#                         logits = torch.tensor(resp["logits"])
#                         preds = logits.argmax(-1).tolist()
#                     pred_label_ids.extend(preds)
#                     true_label_ids.extend(batch["labels"].tolist())

#                 else:
#                     gens = [g.strip() for g in resp.get("generations", [])]
#                     pred_texts.extend(gens)
#                     # refs: list of lists (provided by dataset collator)
#                     refs = batch.get("refs", [[] for _ in gens])
#                     ref_texts_list.extend(refs)

#             # ======== LOCAL (MONOLITHIC) EVAL PATH ========
#             else:
#                 if task_type == "cls":
#                     # Classifier head path: logits → CE loss and argmax
#                     logits = model(input_ids=input_ids, attention_mask=attention_mask).logits
#                     if labels is not None:
#                         loss = F.cross_entropy(logits, labels, reduction="mean")
#                         total_loss += float(loss.item())
#                     else:
#                         total_loss += 0.0
#                     n_batches += 1
#                     preds = logits.argmax(-1).tolist()
#                     pred_label_ids.extend(preds)
#                     true_label_ids.extend(batch["labels"].tolist())

#                 else:
#                     # Generative path: compute LM loss if labels given, and generate texts
#                     # Loss: shift logits vs shifted labels with ignore_index=-100
#                     outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
#                     if hasattr(outputs, "loss") and outputs.loss is not None:
#                         total_loss += float(outputs.loss.item())
#                     else:
#                         # Manual CE if needed
#                         logits = outputs.logits
#                         shift_logits = logits[:, :-1, :].contiguous()
#                         shift_labels = labels[:, 1:].contiguous() if labels is not None else None
#                         if shift_labels is not None:
#                             ce = F.cross_entropy(
#                                 shift_logits.view(-1, shift_logits.size(-1)),
#                                 shift_labels.view(-1),
#                                 ignore_index=-100,
#                                 reduction="mean",
#                             )
#                             total_loss += float(ce.item())
#                     n_batches += 1

#                     # Generation
#                     gen_ids = model.generate(
#                         input_ids=input_ids,
#                         attention_mask=attention_mask,
#                         max_new_tokens=getattr(args, "max_new_tokens", 20),
#                     )

#                     pred_texts = decode_generated_suffix(model, tokenizer, input_ids, attention_mask,
#                                      max_new_tokens=getattr(args, "max_new_tokens", 20))
#                     gens = tokenizer.batch_decode(gen_ids, skip_special_tokens=True)
#                     pred_texts.extend([g.strip() for g in gens])
#                     refs = batch.get("refs", [[] for _ in gens])
#                     ref_texts_list.extend(refs)

#             from metrics import compute_metrics_for_batch
#             m = compute_metrics_for_batch(task_type, pred_texts, batch["meta"], batch["text_target"])

#             # Optional: light debugging for first few batches
#             if batch_idx < 2:
#                 print(f"  Batch {batch_idx} ok.")

#         except Exception as e:
#             print(f"\n⚠️ Error in evaluation batch {batch_idx}: {e}")
#             continue

#     avg_loss = total_loss / max(n_batches, 1)

#     # ---- Aggregate metrics in one place ----
#     if task_type == "cls":
#         md = compute_metrics_for_task(
#             "cls",
#             pred_label_ids=pred_label_ids,
#             true_label_ids=true_label_ids,
#         )
#         acc = float(md["accuracy"])
#         print(f"\n  {task.upper()} Evaluation Complete:")
#         print(f"   Average Loss: {avg_loss:.4f}")
#         print(f"   Accuracy:     {acc:.6f}")
#         return avg_loss, acc, 0.0, 0.0

#     elif task_type == "qa":
#         md = compute_metrics_for_task(
#             "qa",
#             pred_texts=pred_texts,
#             ref_texts_list=ref_texts_list,
#         )
#         f1 = float(md.get("f1", 0.0))
#         em = float(md.get("em", 0.0))
#         print(f"\n  {task.upper()} Evaluation Complete:")
#         print(f"   Average Loss: {avg_loss:.4f}")
#         print(f"   F1:           {f1:.6f}")
#         print(f"   EM:           {em:.6f}")
#         return avg_loss, f1, em, 0.0

#     else:  # "sum"
#         md = compute_metrics_for_task(
#             "sum",
#             pred_texts=pred_texts,
#             ref_texts_list=ref_texts_list,
#         )
#         r1 = float(md.get("rouge1", 0.0))
#         r2 = float(md.get("rouge2", 0.0))
#         rL = float(md.get("rougeL", 0.0))
#         print(f"\n  {task.upper()} Evaluation Complete:")
#         print(f"   Average Loss: {avg_loss:.4f}")
#         print(f"   ROUGE-1:      {r1:.6f}")
#         print(f"   ROUGE-2:      {r2:.6f}")
#         print(f"   ROUGE-L:      {rL:.6f}")
#         return avg_loss, r1, r2, rL
        
# def evaluate_model(model, test_loader, device, tokenizer, args, server_model=None, trainer=None):
#     """SQUAD-specific evaluation with imported metrics"""
#     print("  Starting SQUAD evaluation...")
    
#     # Test generation capability first
#     print("  Testing generation capability...")
#     gen_works = test_generation_simple(model, tokenizer, device)
#     print(f"  Generation test result: {'✅ PASS' if gen_works else '❌ FAIL'}")
    
#     model.eval()
#     total_loss = 0.0
#     total_answer_accuracy = 0.0
#     total_f1 = 0.0
#     total_em = 0.0
#     num_batches = 0

#     with torch.no_grad():
#         # just before: for batch in eval_loader:
#         # _refreshed = _refresh_eval_prefixes(full_model, server_model, trainer)
#         if server_model is not None and trainer is not None:
#             _refreshed = _refresh_eval_prefixes(model, server_model, trainer)
#             if not _refreshed:
#                 print("⚠️ Prefix-aware eval unavailable; falling back to frozen no-prefix eval.")

#         for batch_idx, batch in enumerate(tqdm(test_loader, desc="SQUAD Evaluation")):
#             try:
#                 input_ids = batch['input_ids'].to(device)
#                 attention_mask = batch['attention_mask'].to(device)
#                 # labels = batch['labels'].to(device)
#                 # --- server.py (inside evaluate_model(...) before the forward) ---
#                 labels = input_ids.clone()
#                 labels[attention_mask == 0] = -100
#                 pad_id = getattr(tokenizer, 'pad_token_id', None)
#                 if pad_id is not None:
#                     labels[input_ids == pad_id] = -100
#                 labels = labels.long()

#                 outputs = model(input_ids, attention_mask, labels)  # proceed as before

                
#                 # Debug: Check data structure for first batch
#                 if batch_idx == 0:
#                     print(f"\n  First batch debug:")
#                     print(f"   Input shape: {input_ids.shape}")
#                     print(f"   Has formatted_text: {'formatted_text' in batch}")
#                     print(f"   Has original_example: {'original_example' in batch}")
#                     if 'formatted_text' in batch:
#                         print(f"   Formatted text count: {len(batch['formatted_text'])}")
#                         print(f"   Sample: {batch['formatted_text'][0][:100]}...")
                
#                 # Forward pass
#                 outputs = model(input_ids, attention_mask, labels)
                
#                 # Calculate SQUAD metrics using imported function
#                 loss, answer_acc, f1, em = calculate_squad_metrics(
#                     outputs, labels, batch, tokenizer, model, device
#                 )
                
#                 total_loss += loss
#                 total_answer_accuracy += answer_acc
#                 total_f1 += f1
#                 total_em += em
#                 num_batches += 1
                
#                 # Print progress for first few batches
#                 if batch_idx < 3:
#                     print(f"\nBatch {batch_idx}: Loss={loss:.4f}, Acc={answer_acc:.6f}, F1={f1:.6f}, EM={em:.6f}")
                
#             except Exception as e:
#                 print(f"\n⚠️ Error in evaluation batch {batch_idx}: {e}")
#                 continue
    
#     # Calculate averages
#     avg_loss = total_loss / num_batches if num_batches > 0 else 0.0
#     avg_answer_accuracy = total_answer_accuracy / num_batches if num_batches > 0 else 0.0
#     avg_f1 = total_f1 / num_batches if num_batches > 0 else 0.0
#     avg_em = total_em / num_batches if num_batches > 0 else 0.0
    
#     print(f"\n  SQUAD Evaluation Complete:")
#     print(f"   Average Loss: {avg_loss:.4f}")
#     print(f"   Answer Token Accuracy: {avg_answer_accuracy:.6f}")
#     print(f"   F1 Score: {avg_f1:.6f}")
#     print(f"   Exact Match: {avg_em:.6f}")
    
#     return avg_loss, avg_answer_accuracy, avg_f1, avg_em
    
def train_split_learning_zoo(server_model, full_model, train_loader, eval_loader,
                             optimizer, scheduler, trainer, device, tokenizer, args):
    """Train using split learning with Zeroth-Order Optimization (server-side ZOO)."""
    print("  Starting split learning training (ZOO)...")
    print(f"   ZOO Training configuration:")
    print(f"   Max steps: {args.max_steps}")
    print(f"   Batch size: {args.train_batch_size}")
    print(f"   Learning rate: {args.zoo_lr}")
    print(f"   Perturbation scale (eps): {args.mu}")
    print(f"   Number of perturbations: {args.num_pert}")
    print(f"   Eval every: {args.eval_steps} steps")

    # --- FIX 1: define params to perturb (server KV prefixes only) ---
    server_params = list(server_model.kv.parameters())

    # Create gradient estimator for server KV
    grad_estimator = StochasticGradientApproximator(
        model_params=server_params,
        perturbation_scale=args.mu,
        sample_count=args.num_pert,
        compute_device=device,
        data_type=torch.float32
    )

    server_model.train()
    losses, accs = [], []
    global_step = 0

    try:
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

                    # Build labels JUST like the SGD path: -100 on pads
                    labels = input_ids.clone()
                    labels[attention_mask == 0] = -100
                    pad_id = getattr(tokenizer, 'pad_token_id', None)
                    if pad_id is not None:
                        labels[input_ids == pad_id] = -100
                    labels = labels.long()

                    optimizer.zero_grad(set_to_none=True)

                    def objective_fn():
                        h_cut_live, pkg = _server_forward_to_cut_payload(
                            server_model,
                            input_ids, attention_mask, labels,
                            send_fp16=True
                        )
                        pkg["tgt_len"] = int(h_cut_live.shape[1])

                        # Server is ZOO; the client should skip g_cut and return loss only
                        trainer.send_data({"type": "forward_cut", "mode": "train",
                        "data": pkg, "meta": {"zoo_eval": True}})
                        resp = trainer.receive_data()  # {'loss': float}
                        return torch.as_tensor(resp["loss"], device=h_cut_live.device, dtype=h_cut_live.dtype)
                    
                    def objective_fn_c(*_args,**_kwargs):
                        return objective_fn()

                    
                    # Estimate gradients for server KV via ZOO
                    pbar.set_postfix_str("Computing ZOO gradients...")
                    try:
                        # Newer API (keyword)
                        grad_estimator.model_params = server_params
                        grad_estimator.estimate_gradients(
                            random_seed=global_step * 1000 + args.seed,
                            objective_fn=objective_fn_c
                        )
                    except TypeError:
                        # Legacy API (positional)
                        grad_estimator.estimate_gradients(
                            input_ids, labels, objective_fn_c,
                            random_seed=global_step * 1000 + args.seed
                        )

                    # Apply the ZOO estimated gradients to server KV
                    optimizer.step()

                    # --- monitor (safe) ---
                    try:
                        with torch.no_grad():
                            out = full_model(input_ids, attention_mask, labels)   # frozen full model, no prefixes
                            loss_val, acc, _, _ = calculate_metrics(
                                out, labels, batch, tokenizer, full_model, device
                            )
                    except Exception:
                        # if monitoring hiccups, keep training
                        loss_val, acc = float(objective_fn().item()), 0.0

                    losses.append(loss_val)
                    accs.append(acc)

                    global_step += 1
                    cur_loss = sum(losses[-10:]) / min(len(losses), 10)
                    cur_acc  = sum(accs[-10:]) / min(len(accs), 10)
                    pbar.set_postfix_str(f"{cur_loss:.4f}, Acc: {cur_acc:.6f}")
                    pbar.update(1)

                    # periodic eval (uses the safe FullLLMModel)
                    if global_step % args.eval_steps == 0:
                        print(f"\nStep {global_step}: Running ZOO evaluation...")
                        eval_loss, eval_acc, eval_f1, eval_em = evaluate_model(
                            full_model, eval_loader, device, tokenizer, args
                        )
                        print(f".  Step {global_step} ZOO Evaluation:")
                        print(f"   Loss: {eval_loss:.4f}")
                        print(f"   Answer Accuracy: {eval_acc:.6f}")
                        print(f"   F1 Score: {eval_f1:.6f}")
                        print(f"   Exact Match: {eval_em:.6f}")
                        server_model.train()

                except Exception as e:
                    print(f"\n✖ Error in ZOO step {global_step}: {e}")
                    traceback.print_exc()
                    continue

        pbar.close()
        scheduler.step()

        avg_loss = sum(losses) / len(losses) if losses else 0.0
        avg_acc  = sum(accs) / len(accs) if accs else 0.0
        print(f"\n✅ ZOO Training Complete - Final Loss: {avg_loss:.4f}, Final Acc: {avg_acc:.6f}")
        return avg_loss, avg_acc

    except Exception as e:
        print(f"✖ ZOO training failed: {e}")
        traceback.print_exc()
        return 0.0, 0.0


def train_split_learning_sgd(server_model, full_model, train_loader, eval_loader, optimizer, 
                           scheduler, trainer, device, tokenizer, args):
    """Train using split learning with SGD - FIXED VERSION"""
    print("   Starting split learning training (SGD)...")
    print(f"   Training configuration:")
    print(f"   Max steps: {args.max_steps}")
    print(f"   Batch size: {args.train_batch_size}")
    print(f"   Learning rate: {args.lr}")
    print(f"   Eval every: {args.eval_steps} steps")
    
    server_model.train()
    losses = []
    accs = []
    global_step = 0
    
    try:
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
                    
                    optimizer.zero_grad()
                    labels = input_ids.clone()
                    labels[attention_mask == 0] = -100
                    pad_id = getattr(tokenizer, 'pad_token_id', None)
                    if pad_id is not None:
                        labels[input_ids == pad_id] = -100
                    labels = labels.long()

                    # === True-split: compute and send h_cut ===
                    h_cut_live, pkg = _server_forward_to_cut_payload(
                        server_model,                 # ServerKVOnly instance
                        input_ids, attention_mask, labels,
                        send_fp16=True
                    )
                    pkg["tgt_len"] = int(h_cut_live.shape[1])

                    trainer.send_data({"type": "forward_cut", "mode": "train", "data": pkg, "meta": {"zoo_eval": False}})

                    # === Receive loss + g_cut; backprop on server ===
                    resp = trainer.receive_data()  # {'loss': float, 'g_cut': tensor}
                    # loss_val = float(resp["loss"])
                    # Harden against out-of-band messages
                    if not isinstance(resp, dict):
                        raise RuntimeError(f"Invalid client reply: {type(resp)}")
                    if "loss" not in resp:
                        # Surface client-side exceptions helpfully
                        if resp.get("type") == "client_error":
                            where = resp.get("where", "?")
                            msg   = resp.get("msg", "?")
                            raise RuntimeError(f"Client error during {where}: {msg}")
                        raise KeyError(f"Missing 'loss' in client reply. Keys={list(resp.keys())}")

                    loss_val = float(resp["loss"])
                    print(f"Loss: {loss_val:.4f}")

                    if "g_cut" not in resp:
                        raise RuntimeError("SGD training requires g_cut from client")

                    # Convert numpy array back to tensor and move to correct device
                    g_cut_np = resp["g_cut"]
                    g_cut = torch.from_numpy(g_cut_np).to(device=h_cut_live.device, dtype=h_cut_live.dtype)

                    # Verify shapes match
                    if g_cut.shape != h_cut_live.shape:
                        raise RuntimeError(f"g_cut shape {tuple(g_cut.shape)} != h_cut {tuple(h_cut_live.shape)}")

                    # Verify gradient requirements
                    if not h_cut_live.requires_grad:
                        raise RuntimeError("h_cut_live doesn't require grad - server prefixes may not be trainable")

                    optimizer.zero_grad(set_to_none=True)
                    h_cut_live.backward(g_cut)
                    optimizer.step()

                    # Calculate accuracy for monitoring using full model
                    with torch.no_grad():
                        outputs = full_model(input_ids, attention_mask, labels)
                        loss_check, accuracy, f1, _ = calculate_metrics(outputs, labels, batch, tokenizer, full_model, device)
                        losses.append(loss_val)
                        accs.append(accuracy)
                    
                    # Update progress bar with current loss, accuracy, and F1
                    global_step += 1
                    current_loss = sum(losses[-10:]) / min(len(losses), 10)  # Moving average of last 10
                    current_acc = sum(accs[-10:]) / min(len(accs), 10)
                    
                    pbar.set_postfix_str(f"{current_loss:.4f}, Acc: {current_acc:.3f}")
                    pbar.update(1)
                    
                    # Print detailed stats every 20 batches
                    if (i + 1) % 20 == 0:
                        avg_loss = sum(losses) / len(losses)
                        avg_acc = sum(accs) / len(accs)
                        print(f"\nStep {i+1}/{len(train_loader)} - Loss: {avg_loss:.4f}, Acc: {avg_acc:.3f}")

                    if global_step % args.eval_steps == 0:
                        print(f"\nStep {global_step}: Running evaluation...")
                        eval_loss, eval_acc, eval_f1, eval_em = evaluate_model(
                            full_model, eval_loader, device, tokenizer, args
                        )

                        print(f"   Step {global_step} Evaluation:")
                        print(f"   Loss: {eval_loss:.4f}")
                        print(f"   Answer Accuracy: {eval_acc:.6f}")
                        print(f"   F1 Score: {eval_f1:.6f}")
                        print(f"   Exact Match: {eval_em:.6f}")
                        
                        # Return to training mode
                        server_model.train()

                    # Print progress every 100 steps
                    if global_step % 100 == 0:
                        avg_loss = sum(losses[-100:]) / min(len(losses), 100)
                        avg_acc = sum(accs[-100:]) / min(len(accs), 100)
                        print(f"\nStep {global_step}/{args.max_steps} - Loss: {avg_loss:.4f}, Acc: {avg_acc:.6f}")
                    
                except Exception as e:
                    traceback.print_exc()
                    print(f"Batch {global_step} Error: {e}")
                    continue
        
        scheduler.step()
        
        avg_loss = sum(losses) / len(losses) if losses else 0.0
        avg_acc = sum(accs) / len(accs) if accs else 0.0
        
        print(f"\nSGD Training Complete - Final Loss: {avg_loss:.4f}, Final Acc: {avg_acc:.3f}")
        return avg_loss, avg_acc
        
    except Exception as e:
        print(f"âŒ Training failed: {e}")
        return 0.0, 0.0


def parse_args():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(description='Enhanced Split Learning LLM Server')
    
    # Model parameters
    parser.add_argument('--model_name', type=str, default='facebook/opt-125m', help='Model name')
    parser.add_argument('--num_prefix', type=int, default=10, help='Number of prefix tokens')
    parser.add_argument('--cut_layer', type=int, default=6, help='Split index: 0..L-1 goes to server; cut..L-1 to client')
    parser.add_argument('--max_length', type=int, default=512, help='Max sequence length')
    parser.add_argument("--host", type=str, default="127.0.0.1")
    parser.add_argument("--port", type=int, default=12345)

    
    # Training parameters
    parser.add_argument('--seed', type=int, default=42, help='Random seed')
    parser.add_argument('--zoo_lr', type=float, default=1e-3, help='Learning rate')
    parser.add_argument('--lr', type=float, default=1e-3, help='Learning rate') 
    parser.add_argument('--momentum', type=float, default=0.9, help='SGD momentum')
    parser.add_argument('--epochs', type=int, default=3, help='Number of epochs')
    parser.add_argument('--train_batch_size', type=int, default=4, help='Training batch size')
    parser.add_argument('--test_batch_size', type=int, default=4, help='Test batch size')
    parser.add_argument('--task', choices=["squad", "xsum", "drop", "sst2"], default="squad", help='Use ZOO for client')

    # Dataset sizes - NEW ARGUMENTS
    parser.add_argument('--train_examples', type=int, default=1000, help='Number of training examples')  # TRAIN=1000
    parser.add_argument('--dev_examples', type=int, default=500, help='Number of dev examples')  # DEV=500
    parser.add_argument('--eval_examples', type=int, default=1000, help='Number of eval examples')  # EVAL=1000
    
    # Training steps - NEW ARGUMENTS
    parser.add_argument('--max_steps', type=int, default=4000, help='Maximum training steps')  # STEPS=4000
    parser.add_argument('--eval_steps', type=int, default=4000, help='Evaluate every N steps')  # EVAL_STEPS=4000
    
    
    # ZOO parameters
    parser.add_argument('--mu', type=float, default=1e-1, help='ZOO perturbation scale')
    parser.add_argument('--num_pert', type=int, default=5, help='ZOO perturbations')
    parser.add_argument('--use_zeroth_order', action='store_true', help='Use ZOO for server')
    parser.add_argument('--use_zeroth_order_client', action='store_true', help='Use ZOO for client')
    
    # Evaluation
    parser.add_argument('--evaluate_every', type=int, default=1, help='Evaluate every N epochs')
    parser.add_argument('--f1_method', type=str, default='micro', 
                       choices=['micro', 'macro', 'sequence'],
                       help='F1 score calculation method')
    
    return parser.parse_args()

if __name__ == "__main__":
    try:
        print("STARTING ENHANCED SPLIT LEARNING SERVER")
        print("=" * 60)
        
        args = parse_args()
        torch.manual_seed(args.seed)
        np.random.seed(args.seed)
        
        print(f"  Configuration:")
        print(f"   Model: {args.model_name}")
        print(f"   Batch size: {args.train_batch_size}")
        print(f"   Max length: {args.max_length}")
        print(f"   ZOO server: {args.use_zeroth_order}")
        print(f"   ZOO client: {args.use_zeroth_order_client}")
        print(f"   F1 method: {args.f1_method}")
        if args.use_zeroth_order:
            print(f"   ZOO mu: {args.mu}")
            print(f"   ZOO perturbations: {args.num_pert}")
        
        # Device configuration
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"  Using device: {device}")
        
        # Load tokenizer
        print(f"  Loading tokenizer: {args.model_name}")
        tokenizer = safe_get_hf_tokenizer(args.model_name)
        print("  Tokenizer loaded successfully")
        
        # Create models
        print("  Creating models...")
        server_model = ServerKVOnly(args.model_name, cut_layer=args.cut_layer, num_prefix=args.num_prefix).to(device)
        
        # full_model = FullLLMModel(args.model_name, args.num_prefix).to(device)
        full_model = FullLLMModel(args.model_name, cut_layer=args.cut_layer, num_prefix=args.num_prefix).to(device)
        # give it a live pointer to server prefixes (no copy)
        full_model.attach_live_server_kv(server_model.kv)

        # Synchronize models
        print("  Models created and synchronized")
        
        # Create data loaders
        print(" Creating dataloaders...")
        train_loader, eval_loader = get_squad_dataloaders(args, tokenizer)
        print("  Dataloaders created successfully")
        # Setup optimizer
        print("  Setting up optimizer...")
        if args.use_zeroth_order:
            # ZOO needs higher learning rate and no momentum
            optimizer = optim.SGD(server_model.kv.parameters(), 
                                lr=args.zoo_lr, momentum=0.0)
        else:
            # Regular SGD can use lower learning rate with momentum
            optimizer = optim.SGD(server_model.kv.parameters(), 
                                lr=args.lr, momentum=args.momentum)
        
        
        scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=7, gamma=0.1)
        print("✅ Optimizer ready")
        
        # Setup network
        print("Setting up network...")
        server_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        server_socket.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        server_socket.bind((args.host, args.port))
        server_socket.listen(1)
        
        # Debug: Print model dimensions
        print(f"  Debug - Server model info:")
        
        print("=" * 60)
        print("SERVER READY - WAITING FOR CLIENT")
        print("=" * 60)
        print("Server listening on localhost:12345")
        print("Start client with same parameters")
        print("=" * 60)

        # Accept client connection(s) until we get a valid hello/config
        client_config = None
        # Accept client connection
        def _is_valid_client_config(msg: dict) -> bool:
            if not isinstance(msg, dict):
                return False
            # include the keys you actually use later during setup
            required = {"model_name", "num_prefix"}
            return required.issubset(msg.keys())

        client_config = None

        while True:
            conn, addr = server_socket.accept()
            print(f"✅ Client connected from {addr}")
            trainer = Trainer(conn)

            # Expect the client to speak first (hello or config)
            msg = trainer.receive_data()
            if msg is None:
                print("⚠️ Client disconnected before sending any message; closing and waiting for next client.")
                try:
                    conn.shutdown(socket.SHUT_RDWR)
                except Exception:
                    pass
                conn.close()
                continue  # back to accept()

            if isinstance(msg, dict) and msg.get("type") == "fatal":
                print(f"⚠️ Client reported fatal error during init: {msg.get('error')}")
                conn.close()
                continue  # wait for next client

            if isinstance(msg, dict) and msg.get("type") == "client_hello":
                # safe to send now
                if not trainer.send_data({"type": "server_hello", "version": 1}):
                    print("⚠️ Failed to send server_hello; closing.")
                    conn.close()
                    continue
                if not trainer.send_data({"type": "request_client_config"}):
                    print("⚠️ Failed to request client config; closing.")
                    conn.close()
                    continue
                cfg = trainer.receive_data()
                if _is_valid_client_config(cfg):
                    client_config = cfg
                else:
                    print(f"⚠️ Expected client config, got: {cfg}. Closing.")
                    conn.close()
                    continue

            elif _is_valid_client_config(msg):
                # Legacy client that sends config first
                client_config = msg

            else:
                print(f"⚠️ Unexpected message before config: {msg}. Closing.")
                conn.close()
                continue

            print(f"Received client config: {client_config}")
            # ----- proceed into your training setup using client_config -----
            print("Starting training...")
            break
            
        # Training loop
        for epoch in range(args.epochs):
            print(f"\nEpoch {epoch+1}/{args.epochs}")
            
            # Choose training method based on SERVER configuration only
            if args.use_zeroth_order:
                train_loss, train_acc = train_split_learning_zoo(
                    server_model, full_model, train_loader, eval_loader, optimizer, 
                    scheduler, trainer, device, tokenizer, args
                )
            else:
                train_loss, train_acc = train_split_learning_sgd(
                    server_model, full_model, train_loader, eval_loader, optimizer, 
                    scheduler, trainer, device, tokenizer, args
                )
            
            print(f"✅ Epoch {epoch+1} Training: Loss {train_loss:.4f}, Acc {train_acc:.6f}")
            
            # Evaluation
            if (epoch + 1) % args.evaluate_every == 0:
                print(f"Running evaluation...")
                # eval_loss, eval_acc, eval_f1 = evaluate_model(full_model, test_loader, device, tokenizer, args)
                eval_loss, eval_acc, eval_f1, eval_em = evaluate_model(full_model, eval_loader, device, tokenizer, args)

                
                print(f"\nEPOCH {epoch+1} RESULTS:")
                print(f"{'='*60}")
                print(f"TRAINING   - Loss: {train_loss:.4f}, Accuracy: {train_acc:.4f}")
                print(f"EVALUATION - Loss: {eval_loss:.4f}, Accuracy: {eval_acc:.4f}, F1: {eval_f1:.4f}")
                print(f"{'='*60}")
        
        # Training completed — final evaluation will run before notifying client
        
        # Final evaluation
        print("\nFinal model evaluation...")
        # final_loss, final_acc, final_f1 = evaluate_model(full_model, test_loader, device, tokenizer, args)
        # final_loss, final_acc, final_f1, eval_em = evaluate_model(full_model, eval_loader, device, tokenizer, args)
        # --- server.py (main) ---
        # Final evaluation — do this first so client is alive to send its KV
        final_loss, final_acc, final_f1, final_em = evaluate_model(
            full_model, eval_loader, device, tokenizer, args, server_model=server_model, trainer=trainer
        )
        # now signal completion (non-fatal if client already closed)
        try:
            trainer.send_data({'type': 'training_complete'})
        except Exception as _e:
            print(f"⚠️ Could not notify client of completion (likely closed): {_e}")

        
        print(f"\nFINAL RESULTS:")
        print(f"{'='*60}")
        print(f"Model: {args.model_name}")
        print(f"Epochs: {args.epochs}")
        print(f"Optimization: {'ZOO' if args.use_zeroth_order else 'SGD'}")
        print(f"Final Loss: {final_loss:.4f}")
        print(f"Final Accuracy: {final_acc:.4f}")
        print(f"Final F1 Score: {final_f1:.4f}")
        print(f"{'='*60}")
        
    except Exception as e:
        print(f"❌ CRITICAL SERVER ERROR: {e}")
        print("  Full traceback:")
        traceback.print_exc()
        sys.exit(1)
        
    finally:
        try:
            if 'conn' in locals():
                conn.close()
            if 'server_socket' in locals():
                server_socket.close()
        except:
            pass
        print("  Server shutdown complete")

# if __name__ == "__main__":
#     args = parse_args()
#     torch.manual_seed(args.seed)
#     np.random.seed(args.seed)
#     try:
#         print("STARTING ENHANCED SPLIT LEARNING SERVER")
#         print("=" * 60)

#         HOST = getattr(args, "host", "127.0.0.1")
#         PORT = int(getattr(args, "port", 12345))

#         server_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
#         server_socket.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
#         try:
#             server_socket.bind((HOST, PORT))
#         except OSError as e:
#             print(f"❌ Failed to bind {HOST}:{PORT}: {e}")
#             sys.exit(1)
#         server_socket.listen(1)
#         print(f"Server listening on {HOST}:{PORT}")

#         conn, addr = server_socket.accept()
#         print(f"✅ Client connected from {addr}")
#         trainer = Trainer(conn)
#         trainer.send_data({"type": "server_hello"})
        
#         print(f"  Configuration:")
#         print(f"   Model: {args.model_name}")
#         print(f"   Batch size: {args.train_batch_size}")
#         print(f"   Max length: {args.max_length}")
#         print(f"   ZOO server: {args.use_zeroth_order}")
#         print(f"   ZOO client: {args.use_zeroth_order_client}")
#         print(f"   F1 method: {args.f1_method}")
#         if args.use_zeroth_order:
#             print(f"   ZOO mu: {args.mu}")
#             print(f"   ZOO perturbations: {args.num_pert}")
        
#         # Device configuration
#         device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
#         print(f"  Using device: {device}")
        
#         # Load tokenizer
#         print(f"  Loading tokenizer: {args.model_name}")
#         tokenizer = safe_get_hf_tokenizer(args.model_name)
#         print("  Tokenizer loaded successfully")
        
#         # Create models
#         print("  Creating models...")
#         server_model = ServerKVOnly(args.model_name, cut_layer=args.cut_layer, num_prefix=args.num_prefix).to(device)
        
#         # full_model = FullLLMModel(args.model_name, args.num_prefix).to(device)
#         full_model = FullLLMModel(args.model_name, cut_layer=args.cut_layer, num_prefix=args.num_prefix).to(device)
#         # give it a live pointer to server prefixes (no copy)
#         full_model.attach_live_server_kv(server_model.kv)

#         # Synchronize models
#         print("  Models created and synchronized")
        
#         # Create data loaders
#         print(" Creating dataloaders...")
#         # train_loader, eval_loader = get_squad_dataloaders(args, tokenizer)
#         train_ds, val_ds, collate, task_type = build_task_datasets(args.task, tokenizer, args.max_length)

#         train_loader = DataLoader(train_ds, batch_size=args.train_batch_size, shuffle=True, collate_fn=collate)
#         eval_loader   = DataLoader(val_ds, batch_size=args.test_batch_size,  shuffle=False, collate_fn=collate)
#         print("  Dataloaders created successfully")
#         # Setup optimizer
#         print("  Setting up optimizer...")
#         if args.use_zeroth_order:
#             # ZOO needs higher learning rate and no momentum
#             optimizer = optim.SGD(server_model.kv.parameters(), 
#                                 lr=args.zoo_lr, momentum=0.0)
#         else:
#             # Regular SGD can use lower learning rate with momentum
#             optimizer = optim.SGD(server_model.kv.parameters(), 
#                                 lr=args.lr, momentum=args.momentum)
        
        
#         scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=7, gamma=0.1)
#         print("✅ Optimizer ready")
        
#         # Setup network
#         # print("Setting up network...")
#         # server_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
#         # server_socket.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
#         # server_socket.bind((args.host, args.port))
#         # server_socket.listen(1)

#         # print(f"Server listening on {args.host}:{args.port}")
        
#         # Debug: Print model dimensions
#         print(f"  Debug - Server model info:")
        
#         print("=" * 60)
#         print("SERVER READY - WAITING FOR CLIENT")
#         print("=" * 60)
#         print("Server listening on localhost:12345")
#         print("Start client with same parameters")
#         print("=" * 60)

#         client_cfg = None
#         ready_seen = False
#         while True:
#             try:
#                 msg = trainer.receive_data()
#             except Exception as e:
#                 print(f"⚠️ Preflight receive failed: {e}")
#                 break

#             if not isinstance(msg, dict):
#                 print(f"⚠️ Ignoring non-dict preflight message: {type(msg)}")
#                 continue

#             t = msg.get("type")
#             if t == "client_status":
#                 print(f"Client status: {msg.get('phase')}")
#                 if msg.get("phase") == "ready":
#                     ready_seen = True
#                 continue
            
#             if t == "client_config":
#                 client_cfg = msg
#                 print("Received client configuration from client.")
#                 continue

#             # Not a preflight message; stash nothing—break and let train loop handle it
#             # (All training receives are robust now, see below)
#             print(f"Leaving preflight on message: {t}")
#             break

#         if not ready_seen:
#             print("⚠️ Did not see client 'ready' status; continuing anyway…")

#         print("Starting training...", flush=True)
        
#         # Training loop
#         for epoch in range(args.epochs):
#             print(f"\nEpoch {epoch+1}/{args.epochs}")
            
#             # Choose training method based on SERVER configuration only
#             if args.use_zeroth_order:
#                 train_loss, train_acc = train_split_learning_zoo(
#                     server_model, full_model, train_loader, eval_loader, optimizer, 
#                     scheduler, trainer, device, tokenizer, args
#                 )
#             else:
#                 train_loss, train_acc = train_split_learning_sgd(
#                     server_model, full_model, train_loader, eval_loader, optimizer, 
#                     scheduler, trainer, device, tokenizer, args
#                 )
            
#             print(f"✅ Epoch {epoch+1} Training: Loss {train_loss:.4f}, Acc {train_acc:.6f}")
            
#             # Evaluation
#             if (epoch + 1) % args.evaluate_every == 0:
#                 print(f"Running evaluation...")
#                 # eval_loss, eval_acc, eval_f1 = evaluate_model(full_model, test_loader, device, tokenizer, args)
#                 eval_loss, eval_acc, eval_f1, eval_em = evaluate_model(full_model, eval_loader, device, tokenizer, args)

                
#                 print(f"\nEPOCH {epoch+1} RESULTS:")
#                 print(f"{'='*60}")
#                 print(f"TRAINING   - Loss: {train_loss:.4f}, Accuracy: {train_acc:.4f}")
#                 print(f"EVALUATION - Loss: {eval_loss:.4f}, Accuracy: {eval_acc:.4f}, F1: {eval_f1:.4f}")
#                 print(f"{'='*60}")
        
#         # Training completed — final evaluation will run before notifying client
        
#         # Final evaluation
#         print("\nFinal model evaluation...")
#         # final_loss, final_acc, final_f1 = evaluate_model(full_model, test_loader, device, tokenizer, args)
#         # final_loss, final_acc, final_f1, eval_em = evaluate_model(full_model, eval_loader, device, tokenizer, args)
#         # --- server.py (main) ---
#         # Final evaluation — do this first so client is alive to send its KV
#         final_loss, final_acc, final_f1, final_em = evaluate_model(
#             full_model, eval_loader, device, tokenizer, args, server_model=server_model, trainer=trainer
#         )
#         # now signal completion (non-fatal if client already closed)
#         try:
#             trainer.send_data({'type': 'training_complete'})
#         except Exception as _e:
#             print(f"⚠️ Could not notify client of completion (likely closed): {_e}")

        
#         print(f"\nFINAL RESULTS:")
#         print(f"{'='*60}")
#         print(f"Model: {args.model_name}")
#         print(f"Epochs: {args.epochs}")
#         print(f"Optimization: {'ZOO' if args.use_zeroth_order else 'SGD'}")
#         print(f"Final Loss: {final_loss:.4f}")
#         print(f"Final Accuracy: {final_acc:.4f}")
#         print(f"Final F1 Score: {final_f1:.4f}")
#         print(f"{'='*60}")
        
#     except Exception as e:
#         print(f"❌ CRITICAL SERVER ERROR: {e}")
#         print("  Full traceback:")
#         traceback.print_exc()
#         sys.exit(1)
        
#     finally:
#         try:
#             if 'conn' in locals():
#                 conn.close()
#             if 'server_socket' in locals():
#                 server_socket.close()
#         except:
#             pass
#         print("  Server shutdown complete")

