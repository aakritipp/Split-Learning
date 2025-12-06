import contextlib
import time
import logging
import numpy as np
import random
import torch
import torch.nn as nn
from torch.nn import CrossEntropyLoss
import torch.nn.functional as F
from transformers.modeling_outputs import CausalLMOutputWithPast
from typing import Any, Callable, Dict, List, NewType, Optional, Tuple, Union
from transformers import PreTrainedTokenizerBase
from transformers.tokenization_utils_base import PaddingStrategy
from dataclasses import dataclass
import torch.nn as nn
from lora import LoRALayer, Linear


logger = logging.getLogger(__name__)


@contextlib.contextmanager
def count_time(name):
    logger.info("%s..." % name)
    start_time = time.time()
    try:
        yield
    finally:
        logger.info("Done with %.2fs" % (time.time() - start_time))


@dataclass
class Prediction:
    correct_candidate: Union[int, str]
    predicted_candidate: Union[int, str]

        
@contextlib.contextmanager
def temp_seed(seed):
    state = np.random.get_state()
    np.random.seed(seed)
    try:
        yield
    finally:
        np.random.set_state(state)


def encode_prompt(task, template, train_samples, eval_sample, tokenizer, max_length):
    """
    Encode prompts for eval_sample
    Input: 
    - task, template: task and template class
    - train_samples, eval_sample: demonstrations and the actual sample
    - tokenizer, max_length: tokenizer and max length
    Output:
    - encodings: a list of N lists of tokens. N is the number of options for classification/multiple-choice.
    - option_lens: a list of N integers indicating the number of option tokens.
    """

    # Demonstrations for ICL
    train_prompts = [template.verbalize(sample, sample.correct_candidate).strip() for sample in train_samples]
    train_prompts = task.train_sep.join(train_prompts).strip()
    

    encode_fn = template.encode; verbalize_fn = template.verbalize 
            
    unverbalized_eval_prompt = encode_fn(eval_sample).strip(' ')
    # We generate one prompt for each candidate (different classes in classification)
    # or different choices in multiple-choice tasks
    verbalized_eval_prompts = [verbalize_fn(eval_sample, cand).strip(' ') for cand in eval_sample.candidates]
    unverbalized_eval_prompt_length = len(tokenizer.encode(unverbalized_eval_prompt))
    option_lens = [(len(tokenizer.encode(verbalized_eval_prompt)) - unverbalized_eval_prompt_length) for verbalized_eval_prompt in verbalized_eval_prompts]

    final_prompts = [(train_prompts + task.train_sep + eval_prompt).lstrip().strip(' ') for eval_prompt in verbalized_eval_prompts] 
    
    # Tokenize 
    encodings = [tokenizer.encode(final_prompt) for final_prompt in final_prompts]

    if any([len(encoding) > max_length for encoding in encodings]):
        logger.warn("Exceed max length")
    if tokenizer.add_bos_token:
        encodings = [encoding[0:1] + encoding[1:][-(max_length-1):] for encoding in encodings]  
    else:
        encodings = [encoding[-max_length:] for encoding in encodings]  
   
    return encodings, option_lens


def forward_wrap_with_option_len(self, input_ids=None, labels=None, option_len=None, num_options=None, return_dict=None, **kwargs):
    """
    This is to replace the original forward function of Transformer models to enable:
    (1) Partial target sequence: loss will only be calculated on part of the sequence
    (2) Classification-style training: a classification loss (CE) will be calculated over several options
    Input:
    - input_ids, labels: same as the original forward function
    - option_len: a list of int indicating the option lengths, and loss will be calculated only on the
      last option_len tokens 
    - num_options: a list of int indicating the number of options for each example (this will be #label
      words for classification tasks and #choices for multiple choice tasks), and a classification loss
      will be calculated.
    """
    outputs = self.original_forward(input_ids=input_ids, **kwargs)
    if labels is None:
        return outputs
    logits = outputs.logits

    loss = None
    # Shift so that tokens < n predict n
    shift_logits = logits[..., :-1, :].contiguous()
    # Here we use input_ids (which should always = labels) bc sometimes labels are correct candidate IDs
    shift_labels = torch.clone(input_ids)[..., 1:].contiguous()

    # Some models (e.g., GPT-2) don't define a pad_token_id in their config.
    # In that case, there is no dedicated padding token to mask out, so we skip this step.
    pad_token_id = getattr(self.config, "pad_token_id", None)
    if pad_token_id is not None:
        shift_labels[shift_labels == pad_token_id] = -100

    # Apply option len (do not calculate loss on the non-option part)
    for _i, _len in enumerate(option_len):
        shift_labels[_i, :-_len] = -100

    # Calculate the loss
    loss_fct = CrossEntropyLoss(ignore_index=-100)
    if num_options is not None: 
        # Train as a classification tasks
        log_probs = F.log_softmax(shift_logits, dim=-1)
        mask = shift_labels != -100 # Option part
        shift_labels[~mask] = 0 # So that it doesn't mess up with indexing

        selected_log_probs = torch.gather(log_probs, dim=-1, index=shift_labels.unsqueeze(-1)).squeeze(-1) # (bsz x num_options, len)
        selected_log_probs = (selected_log_probs * mask).sum(-1) / mask.sum(-1) # (bsz x num_options)

        if any([x != num_options[0] for x in num_options]):
            # Multi choice tasks with different number of options
            loss = 0
            start_id = 0
            count = 0
            while start_id < len(num_options):
                end_id = start_id + num_options[start_id]
                _logits = selected_log_probs[start_id:end_id].unsqueeze(0) # (1, num_options)
                _labels = labels[start_id:end_id][0].unsqueeze(0) # (1)
                loss = loss_fct(_logits, _labels) + loss
                count += 1
                start_id = end_id
            loss = loss / count
        else:
            num_options = num_options[0]
            selected_log_probs = selected_log_probs.view(-1, num_options) # (bsz, num_options)
            labels = labels.view(-1, num_options)[:, 0] # Labels repeat so we only take the first one
            loss = loss_fct(selected_log_probs, labels)
    else:
        loss = loss_fct(shift_logits.view(-1, self.config.vocab_size), shift_labels.view(-1))

    if not return_dict:
        output = (logits,) + outputs[1:]
        return (loss,) + output if loss is not None else output

    return CausalLMOutputWithPast(
        loss=loss,
        logits=logits,
        past_key_values=outputs.past_key_values,
        hidden_states=outputs.hidden_states,
        attentions=outputs.attentions,
    )


def mark_only_lora_as_trainable(model: nn.Module, bias: str = 'none') -> None:
    for n, p in model.named_parameters():
        if 'lora_' not in n:
            p.requires_grad = False
    if bias == 'none':
        return
    elif bias == 'all':
        for n, p in model.named_parameters():
            if 'bias' in n:
                p.requires_grad = True
    elif bias == 'lora_only':
        for m in model.modules():
            if isinstance(m, LoRALayer) and \
                hasattr(m, 'bias') and \
                m.bias is not None:
                    m.bias.requires_grad = True
    else:
        raise NotImplementedError


def apply_lora_to_opt(model, lora_r, lora_alpha, lora_dropout):
    """
    Apply LoRA to the attention layers (q_proj, v_proj) of an OPT model.
    """
    # We need to target the server side model which contains the OPT decoder layers.
    # In SplitOPT, the server has 'opt' which is OPTForCausalLM.
    # The layers are in model.server.opt.model.decoder.layers
    
    # But this function might be called on SplitOPT or OPTForCausalLM directly.
    # Let's traverse modules.
    
    target_modules = ["q_proj", "v_proj"]
    
    print(f"Applying LoRA to OPT model with r={lora_r}, alpha={lora_alpha}")
    
    # Helper to replace modules
    def replace_module(module):
        for name, child in module.named_children():
            if name in target_modules and isinstance(child, nn.Linear):
                # Replace with LoRA Linear
                new_layer = Linear(
                    in_features=child.in_features,
                    out_features=child.out_features,
                    r=lora_r,
                    lora_alpha=lora_alpha,
                    lora_dropout=lora_dropout,
                    merge_weights=False # Important for MeZO to control merge state
                )
                
                # Initialize weights to match original (copy pretrained weights)
                new_layer.weight.data.copy_(child.weight.data)
                if child.bias is not None:
                    new_layer.bias.data.copy_(child.bias.data)
                
                setattr(module, name, new_layer)
                print(f"Replaced {name} with LoRA Linear")
            else:
                replace_module(child)

    replace_module(model)


@dataclass
class DataCollatorWithPaddingAndNesting:
    """
    Collator for training
    """

    tokenizer: PreTrainedTokenizerBase
    padding: Union[bool, str, PaddingStrategy] = True
    max_length: Optional[int] = None
    pad_to_multiple_of: Optional[int] = None
    return_tensors: str = "pt"

    def __call__(self, features: List[Dict[str, Any]]) -> Dict[str, Any]:
        features = [ff for f in features for ff in f]
        
        # Separate labels if they exist to handle padding manually (standard pad doesn't pad labels with -100 usually)
        labels = [f.pop("labels") for f in features] if "labels" in features[0] else None
        
        batch = self.tokenizer.pad(
            features,
            padding=self.padding,
            max_length=self.max_length,
            pad_to_multiple_of=self.pad_to_multiple_of,
            return_tensors=self.return_tensors,
        )
        
        if labels is not None:
            # MeZO Fix: Handle case where labels are integers (classification indices)
            if isinstance(labels[0], int):
                # If labels are integers, we just convert them to tensor directly, no padding needed
                batch["labels"] = torch.tensor(labels, dtype=torch.long)
            else:
                max_label_length = max(len(l) for l in labels)
                if self.pad_to_multiple_of:
                    max_label_length = (max_label_length + self.pad_to_multiple_of - 1) // self.pad_to_multiple_of * self.pad_to_multiple_of
                
                padding_side = self.tokenizer.padding_side
                padded_labels = []
                for l in labels:
                    # Ensure l is a list
                    if not isinstance(l, list):
                        l = l.tolist() if hasattr(l, "tolist") else [l]
                    
                    remainder = [ -100 ] * (max_label_length - len(l))
                    if padding_side == "right":
                        padded_labels.append(l + remainder)
                    else:
                        padded_labels.append(remainder + l)
                
                batch["labels"] = torch.tensor(padded_labels, dtype=torch.long)
            
        if "label" in batch:
            batch["labels"] = batch["label"]
            del batch["label"]
        if "label_ids" in batch:
            batch["labels"] = batch["label_ids"]
            del batch["label_ids"]
        return batch
