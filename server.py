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
from lora import apply_lora_to_opt, iter_lora_parameters, get_lora_state_dict, load_lora_state_dict
from dataset import get_enhanced_dataloaders as get_task_dataloaders


# Ensure merge_past_key_values is available for prefix-aware eval
try:
    from prefix_kv import merge_past_key_values
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

@torch.no_grad()
def print_squad_generations(model, tokenizer, batch, device, max_new_tokens=30, k=4):
    """Print generation examples during evaluation"""
    print("\n=== GENERATION SAMPLES ===")
    
    if "prompt_only" not in batch:
        print("No prompt_only field available for generation testing")
        return
    
    model.eval()
    prompts = batch["prompt_only"][:k]
    refs = batch["refs"][:k]
    
    # Tokenize prompts
    enc = tokenizer(
        prompts, 
        return_tensors="pt", 
        padding=True, 
        truncation=True,
        max_length=350
    ).to(device)

    input_ids, attention_mask = right_trim(enc["input_ids"], enc["attention_mask"])
    enc = {"input_ids": input_ids, "attention_mask": attention_mask}
    
    
    # Generate
    try:
        outputs = model.generate(
            enc["input_ids"],
            attention_mask=enc["attention_mask"],
            do_sample=False,
            num_beams=1,
            max_new_tokens=max_new_tokens,
            eos_token_id=tokenizer.eos_token_id,
            pad_token_id=tokenizer.pad_token_id
        )
    except Exception as e:
        print(f"Generation failed: {e}")
        return
    
    # Process outputs
    for i in range(min(k, outputs.size(0))):
        try:
            # Decode full generated sequence
            full_text = tokenizer.decode(outputs[i], skip_special_tokens=True)
            
            # Remove prompt to get just the answer
            prompt_text = tokenizer.decode(enc["input_ids"][i], skip_special_tokens=True)
            if len(full_text) > len(prompt_text):
                pred_answer = full_text[len(prompt_text):].strip()
            else:
                pred_answer = ""
            
            # Clean up prediction
            pred_answer = pred_answer.split('\n')[0].strip('.,!?"\'')
            
            # Get ground truth
            gold_answers = refs[i] if i < len(refs) else [""]
            gold_answer = gold_answers[0] if gold_answers else ""
            
            # Extract question for display
            question_part = prompts[i].split('Question:', 1)[-1].split('Answer:', 1)[0].strip()
            
            # Calculate metrics
            from metrics import squad_f1_score, squad_exact_match
            f1 = squad_f1_score(pred_answer, gold_answers)
            em = squad_exact_match(pred_answer, gold_answers)
            
            print(f"\nExample {i+1}:")
            print(f"Q: {question_part}")
            print(f"Pred: '{pred_answer}'")
            print(f"Gold: '{gold_answer}'")
            print(f"EM={em:.3f} F1={f1:.3f}")
            
        except Exception as e:
            print(f"Error processing example {i}: {e}")
    
    print("=== END SAMPLES ===\n")

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
            if side == "server":
                # Only server KV prefixes should be trainable
                is_allowed = n.startswith("kv.")
            elif side == "client":
                # Only client prefixes should be trainable
                is_allowed = n.startswith("client_kv.")
            else:
                # Generic fallback: any KV/prefix-like params are allowed
                is_allowed = (n.startswith("kv.") or n.startswith("client_kv.") or ("prefix" in n))

            ok = ("lora_A" not in n and "lora_B" not in n) and ((is_allowed) == p.requires_grad)

        elif mode == "lora":
            # Only LoRA adapters should be trainable
            is_lora = ("lora_A" in n) or ("lora_B" in n)
            ok = (is_lora == p.requires_grad) and ("kv." not in n) and ("client_kv." not in n)

        else:  # none
            ok = (p.requires_grad is False)

        assert ok, f"Unexpected trainable param in {mode} mode{f' ({side})' if side else ''}: {n} requires_grad={p.requires_grad}"


def _neg_inf(dtype: torch.dtype) -> float:
    # Use the representable minimum as the additive mask value
    return torch.finfo(dtype).min

def _refresh_eval_prefixes(full_model, server_model, trainer, args):
    """
    Pull the latest client prefix snapshot for eval,
    attach live server prefixes, and enable prefix-aware eval.
    Safe fallback to legacy eval if anything fails.
    """
    # LoRA mode: do not use prefix-aware eval or request client KV
    if args.tuning == "lora":
        # In LoRA mode, skip split eval handshake entirely; use local frozen eval only
        full_model.enable_prefix_eval(False)
        return False
    
    full_model.attach_live_server_kv(server_model.kv)
    try:
        trainer.send_data({"type": "get_client_kv_state"})
        resp = trainer.receive_data()
        if isinstance(resp, dict) and resp.get("type") == "client_kv_state":
            state = resp["state"]
            # Guard against LoRA/empty state
            if state and state.get("k") is not None and state.get("v") is not None:
                full_model.load_client_kv_state(state)
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

class LoRAServerModel(nn.Module):
    """
    Server-side when tuning=LoRA. Keeps full base_model, injects LoRA only into layers [0..cut-1].
    Presents no-prefix stubs so forward/masks pipeline stays unified.
    """
    def __init__(self, model_name: str, cut_layer: int,
                 r: int = 8, alpha: int = 16, dropout: float = 0.0,
                 targets=("q_proj","v_proj")):
        super().__init__()
        self.base_model = AutoModelForCausalLM.from_pretrained(
            model_name, torch_dtype=torch.float32, device_map=None
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
        self.server_kv = _NoPrefixStub()
        self.client_kv_mirror = _NoPrefixStub()

        # Provide a minimal KV stub so shared code paths can reference server_model.kv safely
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
        """Return an empty KV state to satisfy interfaces expecting a server KV snapshot."""
        try:
            # Prefer the stub tensors so shapes/types are tensors
            return {"k": self.kv.k.detach().cpu(), "v": self.kv.v.detach().cpu()}
        except Exception:
            # Ultimate fallback
            return {"k": torch.zeros(0), "v": torch.zeros(0)}

    def trainable_parameters(self):
        return iter_lora_parameters(self.base_model, layer_range=(0, self.cut_layer-1))

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
        for p in self.kv.parameters():
            p.requires_grad = True
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
        # Freeze all base model params on server in prefix mode; only KV prefixes train
        for p in self.base_model.parameters():
            p.requires_grad = False

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
            # Optional fields (some codepaths expect server_kv_state even in LoRA mode)
            formatted_texts.append(item.get('formatted_text', ""))
            original_examples.append(item.get('original_example', {}))
            if 'server_kv_state' in item:
                pass
        
        # Stack tensors
        batch_dict = {
            'input_ids': torch.stack(input_ids),
            'attention_mask': torch.stack(attention_masks),
            'labels': torch.stack(labels),
        }
        
        # Always include non-tensor metadata lists for downstream logging/metrics
        batch_dict['formatted_text'] = formatted_texts
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
        return tokenizer
    except Exception as e:
        print(f"Failed to load tokenizer for {model_name}: {e}")
        raise

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
            # Ensure the suffix containing "Answer:" + answer is retained
            old_side = getattr(self.tokenizer, 'truncation_side', 'right')
            try:
                self.tokenizer.truncation_side = 'left'
            except Exception:
                pass
            encoding = self.tokenizer(
                text,
                truncation=True,
                padding='max_length',
                max_length=self.max_length,
                return_tensors='pt'
            )
            try:
                self.tokenizer.truncation_side = old_side
            except Exception:
                pass
            
            input_ids = encoding['input_ids'].squeeze()
            attention_mask = encoding['attention_mask'].squeeze()
            
            # Ensure consistent tensor shapes
            if input_ids.dim() == 0:
                input_ids = input_ids.unsqueeze(0)
            if attention_mask.dim() == 0:
                attention_mask = attention_mask.unsqueeze(0)
            
            # Build labels that supervise ONLY the answer tokens (ignore prefix + padding)
            labels = input_ids.clone()
            valid_len = int(attention_mask.sum().item())
            ids_list = input_ids.tolist()

            # Helper to find token subsequence
            def _find_subseq(hay, needle):
                n = len(needle)
                if n == 0 or n > len(hay):
                    return -1
                for i in range(0, len(hay) - n + 1):
                    if hay[i:i+n] == needle:
                        return i
                return -1

            # Try several encodings of the marker to be robust to whitespace/newlines
            candidates = []
            for marker in ["\nAnswer:", " Answer:", "Answer:"]:
                try:
                    tok = self.tokenizer.encode(marker, add_special_tokens=False)
                    if tok:
                        candidates.append(tok)
                except Exception:
                    pass

            pos = -1
            match_len = 0
            for cand in candidates:
                p = _find_subseq(ids_list, cand)
                if p != -1 and (pos == -1 or p < pos):
                    pos = p
                    match_len = len(cand)

            if pos != -1:
                start_idx = pos + match_len
            else:
                # Fallback via character split
                astart = text.find("Answer:")
                if astart == -1:
                    start_idx = 0
                else:
                    prefix = text[:astart + len("Answer:")]
                    old_side2 = getattr(self.tokenizer, 'truncation_side', 'right')
                    try:
                        self.tokenizer.truncation_side = 'left'
                    except Exception:
                        pass
                    prefix_tokens = self.tokenizer.encode(prefix, add_special_tokens=False)
                    try:
                        self.tokenizer.truncation_side = old_side2
                    except Exception:
                        pass
                    start_idx = len(prefix_tokens)

            # Clamp to valid sequence length
            start_idx = max(0, min(start_idx, valid_len))
            # Ignore everything before the answer prefix
            if start_idx > 0:
                labels[:start_idx] = -100
            # Ignore padding
            labels[attention_mask == 0] = -100
            pad_id = getattr(self.tokenizer, 'pad_token_id', None)
            if pad_id is not None:
                labels[input_ids == pad_id] = -100
            labels = labels.long()
            
            result = {
                'input_ids': input_ids,
                'cut_layer': args.cut_layer,
                'attention_mask': attention_mask,
                'labels': labels
            }
            # Only include server_kv_state in prefix mode
            try:
                if args.tuning != 'lora':
                    result['server_kv_state'] = server_model.state_dict_kv()
            except Exception:
                pass
            
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


class Trainer:
    """Handles communication with the client"""
    def __init__(self, conn):
        self.conn = conn
    
    def send_data(self, data):
        try:
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
            return pickle.loads(data)
        except Exception as e:
            print(f"❌ Failed to receive data: {e}")
            raise


def _classification_metrics(outputs, labels, batch, tokenizer):
    """Compute simple classification accuracy when 'class_label' is present."""
    try:
        # Predict the next token after the marker as a word and map to class
        logits = outputs.logits
        # position for first answer token: find via formatted_text
        preds = []
        trues = []
        for i, text in enumerate(batch.get('formatted_text', [])):
            if 'Answer:' not in text:
                continue
            ctx = text[: text.find('Answer:') + len('Answer:')]
            ctx_tokens = tokenizer.encode(ctx, add_special_tokens=False)
            pos = max(len(ctx_tokens) - 1, 0)
            if pos >= logits.shape[1]-1:
                continue
            token_id = int(torch.argmax(logits[i, pos, :]).item())
            token_str = tokenizer.decode([token_id]).strip().lower()
            mapped = 1 if token_str in {"great", "positive", "good"} else 0
            preds.append(mapped)
            if 'class_label' in batch:
                trues.append(int(batch['class_label'][i].item()))
        if preds and trues and len(preds) == len(trues):
            correct = sum(int(p==t) for p,t in zip(preds, trues))
            return float(correct/len(trues))
    except Exception:
        return 0.0
    return 0.0


def calculate_metrics(outputs, labels, batch, tokenizer, model, device):
    """Task-aware metrics: classification (SST-2) vs generation (QA/summary)."""
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
        # Classification branch: if batch has class labels, compute classification acc
        if 'class_label' in batch:
            answer_accuracy = _classification_metrics(outputs, labels, batch, tokenizer)
        else:
            # Calculate answer token accuracy for generation tasks
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

def evaluate_model(model, test_loader, device, tokenizer, args, server_model=None, trainer=None):
    """Task-aware evaluation; generation metrics for QA/summary, class acc for SST-2."""
    print("  Starting SQUAD evaluation...")
    
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
        split_eval = False
        if server_model is not None and trainer is not None:
            if getattr(args, 'tuning', 'prefix') == 'lora':
                # In LoRA mode, use split pipeline (no prefix handshake needed)
                split_eval = True
            else:
                _refreshed = _refresh_eval_prefixes(model, server_model, trainer, args)
                if not _refreshed:
                    print("⚠️ Prefix-aware eval unavailable; falling back to frozen no-prefix eval.")
                else:
                    split_eval = True

        for batch_idx, batch in enumerate(tqdm(test_loader, desc="SQUAD Evaluation")):
            try:
                input_ids, attention_mask, labels, prompt_text, text_target, meta = adapt_batch(batch, device)
                input_ids, attention_mask, labels = right_trim(input_ids, attention_mask, labels)

                # LoRA eval: combine server+client LoRA on a fresh full model and evaluate locally
                if split_eval and getattr(args, 'tuning', 'prefix') == 'lora':
                    try:
                        # 1) Build a fresh full model
                        full_eval = AutoModelForCausalLM.from_pretrained(args.model_name, torch_dtype=torch.float32, device_map=None).to(device)
                        for p in full_eval.parameters():
                            p.requires_grad = False

                        total_layers_local = full_eval.config.num_hidden_layers

                        # 2) Apply LoRA to both halves
                        apply_lora_to_opt(full_eval, targets=tuple(args.lora_targets.split(',')), layer_range=(0, args.cut_layer-1), r=args.lora_r, lora_alpha=args.lora_alpha, lora_dropout=args.lora_dropout)
                        apply_lora_to_opt(full_eval, targets=tuple(args.lora_targets.split(',')), layer_range=(args.cut_layer, total_layers_local-1), r=args.lora_r, lora_alpha=args.lora_alpha, lora_dropout=args.lora_dropout)

                        # 3) Load server LoRA into [0..cut-1]
                        server_state = get_lora_state_dict(getattr(server_model, 'base_model', server_model), layer_range=(0, args.cut_layer-1))
                        _ = load_lora_state_dict(full_eval, server_state)

                        # 4) Request client LoRA for [cut..L-1] and load
                        trainer.send_data({"type": "get_client_lora_state"})
                        resp = trainer.receive_data()
                        if isinstance(resp, dict) and resp.get("type") == "client_lora_state" and resp.get("ok", False):
                            client_state = resp.get("state", {})
                            _ = load_lora_state_dict(full_eval, client_state)
                        else:
                            print("⚠️ Could not fetch client LoRA state; proceeding with server half only")

                        # 5) Compute outputs and metrics locally on combined model
                        outputs = full_eval(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
                        loss, answer_acc, f1, em = calculate_squad_metrics(outputs, labels, batch, tokenizer, full_eval, device)

                        total_loss += loss
                        total_answer_accuracy += answer_acc
                        total_f1 += f1
                        total_em += em
                        num_batches += 1
                        if batch_idx < 3:
                            print(f"\nBatch {batch_idx}: Loss={loss:.4f}, Acc={answer_acc:.6f}, F1={f1:.6f}, EM={em:.6f}")
                        continue
                    except Exception as e:
                        print(f"⚠️ Combined LoRA eval failed, falling back to client eval path: {e}")
                        # Fall through to legacy path below if needed

                # Default eval: compute local outputs for metrics
                outputs = model(input_ids, attention_mask, labels)

                # Debug: Check data structure for first batch
                if batch_idx == 0:
                    print(f"\n  First batch debug:")
                    print(f"   Input shape: {input_ids.shape}")
                    print(f"   Has formatted_text: {'formatted_text' in batch}")
                    print(f"   Has original_example: {'original_example' in batch}")
                    if 'formatted_text' in batch:
                        print(f"   Formatted text count: {len(batch['formatted_text'])}")
                        print(f"   Sample: {batch['formatted_text'][0][:100]}...")

                # If split eval context exists, ask client to compute loss on its half once
                if split_eval:
                    h_cut_live, pkg = _server_forward_to_cut_payload(
                        server_model,
                        input_ids, attention_mask, labels,
                        send_fp16=True
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

                    resp = trainer.receive_data()  # client returns eval stats
                    if not (isinstance(resp, dict) and resp.get("type") == "eval_stats"):
                        raise RuntimeError(f"Bad eval resp: {type(resp)}")

                    # Client responded; we rely on local metrics for aggregation

                # Calculate task-aware metrics locally (authoritative)
                # Choose generation max tokens dynamically per task
                task = getattr(args, 'task', 'squad')
                gen_max = 5
                if task == 'squad':
                    gen_max = 16
                elif task == 'drop':
                    gen_max = 8
                elif task == 'xsum':
                    gen_max = 20
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
    
    print(f"\n  SQUAD Evaluation Complete:")
    print(f"   Average Loss: {avg_loss:.4f}")
    print(f"   Answer Token Accuracy: {avg_answer_accuracy:.6f}")
    print(f"   F1 Score: {avg_f1:.6f}")
    print(f"   Exact Match: {avg_em:.6f}")
    
    return avg_loss, avg_answer_accuracy, avg_f1, avg_em
    
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

                    def objective_fn():
                        # Ensure deterministic probe: disable dropout temporarily
                        was_training = server_model.training
                        server_model.eval()
                        with torch.no_grad():
                            h_cut_live, pkg = _server_forward_to_cut_payload(
                                server_model,
                                input_ids, attention_mask, labels,
                                send_fp16=True
                            )
                        pkg["tgt_len"] = int(h_cut_live.shape[1])
                        trainer.send_data({"type": "forward_cut", "mode": "train",
                        "data": pkg, "meta": {"zoo_eval": True}})
                        # Server is ZOO; the client should skip g_cut and return loss only
                        resp = trainer.receive_data()  # {'type': 'loss_report', 'loss': float}
                        loss_val = float(resp.get("loss", 0.0))
                        if was_training:
                            server_model.train()
                        return torch.as_tensor(loss_val, device=h_cut_live.device, dtype=h_cut_live.dtype)

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
                        # eval_loss, eval_acc, eval_f1, eval_em = evaluate_model(
                        #     full_model, eval_loader, device, tokenizer, args
                        # )
                        eval_loss, eval_acc, eval_f1, eval_em = evaluate_model(
                            full_model, eval_loader, device, tokenizer, args,
                            server_model=server_model, trainer=trainer
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
                    h_cut_live, pkg = _server_forward_to_cut_payload(
                        server_model,                 # ServerKVOnly instance
                        input_ids, attention_mask, labels,
                        send_fp16=True
                    )
                    pkg["tgt_len"] = int(h_cut_live.shape[1])

                    trainer.send_data({"type": "forward_cut", "mode": "train", "data": pkg, "meta": {"zoo_eval": False}})

                    # === Receive loss + g_cut; backprop on server ===
                    resp = trainer.receive_data()  # {'loss': float, 'g_cut': tensor}
                    loss_val = float(resp["loss"])
                    print(f"Loss: {loss_val:.4f}")

                    g_cut = torch.as_tensor(resp["g_cut"])
                    if g_cut.shape != h_cut_live.shape:
                        raise RuntimeError(f"g_cut shape {tuple(g_cut.shape)} != h_cut {tuple(h_cut_live.shape)}")
                    g_cut = g_cut.to(device=h_cut_live.device, dtype=h_cut_live.dtype)

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
                            full_model, eval_loader, device, tokenizer, args,
                            server_model=server_model, trainer=trainer
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
    parser.add_argument('--num_prefix', type=int, default=20, help='Number of prefix tokens')
    parser.add_argument('--cut_layer', type=int, default=6, help='Split index: 0..L-1 goes to server; cut..L-1 to client')
    parser.add_argument('--max_length', type=int, default=512, help='Max sequence length')
    
    # Training parameters
    parser.add_argument('--seed', type=int, default=42, help='Random seed')
    parser.add_argument('--zoo_lr', type=float, default=1e-3, help='Learning rate')
    parser.add_argument('--lr', type=float, default=1e-3, help='Learning rate') 
    parser.add_argument('--momentum', type=float, default=0.9, help='SGD momentum')
    parser.add_argument('--epochs', type=int, default=3, help='Number of epochs')
    parser.add_argument('--train_batch_size', type=int, default=4, help='Training batch size')
    parser.add_argument('--test_batch_size', type=int, default=4, help='Test batch size')

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

    parser.add_argument('--task', choices=["squad", "xsum", "drop", "sst2"], default="squad", help='Use ZOO for client')
    parser.add_argument('--tuning', choices=['prefix','lora','none'], default='prefix')
    parser.add_argument('--lora_r', type=int, default=8)
    parser.add_argument('--lora_alpha', type=int, default=16)
    parser.add_argument('--lora_dropout', type=float, default=0.0)
    parser.add_argument('--lora_targets', type=str, default='q_proj,v_proj')
    
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
        if args.tuning == 'lora':
            server_model = LoRAServerModel(
                args.model_name, args.cut_layer,
                r=args.lora_r, alpha=args.lora_alpha, dropout=args.lora_dropout,
                targets=tuple(args.lora_targets.split(','))
            ).to(device)
            trainable_params = list(server_model.trainable_parameters())
            print(f"Server owns layers [0, {args.cut_layer-1}] with LoRA r={args.lora_r}, alpha={args.lora_alpha}")
            _assert_only_expected_trainables(server_model, args.tuning, side="server")

        else:
            # existing PrefixKV server path (keep yours)
            server_model = ServerKVOnly(args.model_name, cut_layer=args.cut_layer, num_prefix=args.num_prefix).to(device)
            trainable_params = list(server_model.kv.parameters())  # unchanged
            _assert_only_expected_trainables(server_model, args.tuning, side="server")


        if args.tuning == 'lora':
            assert all((not p.requires_grad) for n,p in server_model.named_parameters()
                    if ("lora_A" not in n and "lora_B" not in n)), "Only LoRA params must be trainable in LoRA mode!"
        
        full_model = FullLLMModel(args.model_name, cut_layer=args.cut_layer, num_prefix=args.num_prefix).to(device)
        # Only attach server KV in prefix mode; in LoRA there is no live server KV to use
        if args.tuning != 'lora':
            full_model.attach_live_server_kv(server_model.kv)

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
        print("  Dataloaders created successfully")
        # Setup optimizer
        print("  Setting up optimizer...")
        if args.use_zeroth_order:
            # ZOO needs higher learning rate and no momentum (optimizer used for applying estimated grads)
            optimizer = optim.SGD(trainable_params, lr=args.zoo_lr, momentum=0.0)
        else:
            # Regular SGD can use lower learning rate with momentum
            optimizer = optim.SGD(trainable_params, lr=args.lr, momentum=args.momentum)
        
        
        scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=7, gamma=0.1)
        print("✅ Optimizer ready")
        
        # Setup network
        print("Setting up network...")
        server_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        server_socket.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        server_socket.bind(('localhost', 12345))
        server_socket.listen(1)
        
        # Debug: Print model dimensions
        print(f"  Debug - Server model info:")
        
        print("=" * 60)
        print("SERVER READY - WAITING FOR CLIENT")
        print("=" * 60)
        print("Server listening on localhost:12345")
        print("Start client with same parameters")
        print("=" * 60)
        
        # Accept client connection
        conn, addr = server_socket.accept()
        print(f"✅ Client connected from {addr}")
        
        trainer = Trainer(conn)
        client_config = trainer.receive_data()
        print(f"Received client config: {client_config}")
        
        print("Starting training...")
        
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
                eval_loss, eval_acc, eval_f1, eval_em = evaluate_model(
                    full_model, eval_loader, device, tokenizer, args,
                    server_model=server_model, trainer=trainer
                )
                print(f"\nEPOCH {epoch+1} RESULTS:")
                print(f"{'='*60}")
                print(f"TRAINING   - Loss: {train_loss:.4f}, Accuracy: {train_acc:.4f}")
                print(f"EVALUATION - Loss: {eval_loss:.4f}, Accuracy: {eval_acc:.4f}, F1: {eval_f1:.4f}")
                print(f"{'='*60}")
                
        # Final evaluation
        print("\nFinal model evaluation...")
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