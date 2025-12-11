import logging
import os

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

import argparse
import torch
import random
import numpy as np
from dataclasses import dataclass, field
from typing import List, Optional
from datasets import Dataset
import torch.nn.functional as F
import time
from tqdm import tqdm
from torch.utils.data import Dataset
from transformers import Trainer, HfArgumentParser, TrainingArguments, AutoConfig, AutoTokenizer, AutoModelForCausalLM, DataCollatorForTokenClassification
from transformers.modeling_outputs import CausalLMOutput
from torch.distributed.fsdp.fully_sharded_data_parallel import FullyShardedDataParallel as FSDP

from utils import *
from metrics import calculate_metric
from trainer import OurTrainer
from dataset import get_task
from splitmodel import GPT2Config, GPT2LMModel_Server, GPT2LMModel_Client, SplitGPT2, SplitOPT
from lora import *
from split_communication import format_payload_size


class NondiffCollator:
    """
    Collator for non-differentiable objective training.
    Handles padding and preserves 'gold' labels for F1/acc computation.
    """
    def __init__(self, tokenizer, pad_to_multiple_of=None):
        self.tokenizer = tokenizer
        self.pad_to_multiple_of = pad_to_multiple_of
    
    def __call__(self, features):
        # Extract gold labels for non-diff training
        golds = [f.pop("gold") if "gold" in f else None for f in features]
        
        # Get max length
        max_length = max(len(f["input_ids"]) for f in features)
        if self.pad_to_multiple_of:
            max_length = ((max_length + self.pad_to_multiple_of - 1) // self.pad_to_multiple_of) * self.pad_to_multiple_of
        
        # Pad input_ids and labels
        padded_input_ids = []
        padded_labels = []
        option_lens = []
        
        padding_side = getattr(self.tokenizer, 'padding_side', 'left')
        pad_token_id = self.tokenizer.pad_token_id or 0
        
        for f in features:
            input_ids = f["input_ids"]
            labels = f.get("labels", input_ids)
            option_len = f.get("option_len", 0)
            
            if padding_side == "left":
                padded = [pad_token_id] * (max_length - len(input_ids)) + list(input_ids)
                padded_lab = [-100] * (max_length - len(labels)) + list(labels)
            else:
                padded = list(input_ids) + [pad_token_id] * (max_length - len(input_ids))
                padded_lab = list(labels) + [-100] * (max_length - len(labels))
            
            padded_input_ids.append(padded)
            padded_labels.append(padded_lab)
            option_lens.append(option_len)
        
        batch = {
            "input_ids": torch.tensor(padded_input_ids, dtype=torch.long),
            "labels": torch.tensor(padded_labels, dtype=torch.long),
            "option_len": option_lens,
        }
        
        if any(g is not None for g in golds):
            batch["gold"] = golds
        
        return batch


def get_gpu_memory_mb():
    """Get current GPU memory usage in MB."""
    if torch.cuda.is_available():
        return torch.cuda.memory_allocated() / (1024 ** 2)
    return 0.0


def get_gpu_max_memory_mb():
    """Get peak GPU memory usage in MB."""
    if torch.cuda.is_available():
        return torch.cuda.max_memory_allocated() / (1024 ** 2)
    return 0.0


def reset_gpu_memory_stats():
    """Reset GPU memory statistics."""
    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()
        torch.cuda.empty_cache()


def get_model_memory_mb(model):
    """Estimate model memory usage in MB based on parameter sizes."""
    total_bytes = 0
    for param in model.parameters():
        total_bytes += param.nelement() * param.element_size()
    for buffer in model.buffers():
        total_bytes += buffer.nelement() * buffer.element_size()
    return total_bytes / (1024 ** 2)


def get_trainable_memory_mb(model):
    """Estimate memory needed for trainable parameters only (in MB)."""
    total_bytes = 0
    for param in model.parameters():
        if param.requires_grad:
            total_bytes += param.nelement() * param.element_size()
    return total_bytes / (1024 ** 2)


def get_gradient_memory_estimate_mb(model, optimizer_type="sgd", momentum=0.0):
    """
    Estimate memory needed for gradients and optimizer states (in MB).
    
    For each trainable parameter:
    - Gradients: same size as parameter (for FO only)
    - SGD with momentum: one buffer per parameter
    - Adam: two buffers per parameter (m and v)
    """
    trainable_bytes = 0
    for param in model.parameters():
        if param.requires_grad:
            trainable_bytes += param.nelement() * param.element_size()
    
    gradient_memory = trainable_bytes  # Same size as trainable params
    
    if optimizer_type == "sgd":
        if momentum > 0:
            optimizer_memory = trainable_bytes  # Momentum buffer
        else:
            optimizer_memory = 0
    elif optimizer_type == "adam":
        optimizer_memory = 2 * trainable_bytes  # m and v buffers
    else:
        optimizer_memory = 0
    
    return (gradient_memory + optimizer_memory) / (1024 ** 2)


def estimate_activation_memory_mb(batch_size, seq_len, hidden_dim, num_layers, 
                                   bytes_per_element=4, store_for_backward=True):
    """
    Estimate activation memory for transformer forward pass.
    
    For each layer, we store:
    - Input activations: batch × seq × hidden
    - Attention scores: batch × heads × seq × seq (approx)
    - FFN intermediate: batch × seq × 4*hidden
    
    Args:
        batch_size: Batch size
        seq_len: Sequence length
        hidden_dim: Hidden dimension
        num_layers: Number of transformer layers
        bytes_per_element: Bytes per tensor element (4 for fp32, 2 for fp16)
        store_for_backward: If True, store activations for backprop (FO mode)
    """
    if not store_for_backward:
        # ZO mode: only need current layer's activations
        # Just input + output of current computation
        per_layer = batch_size * seq_len * hidden_dim * bytes_per_element * 2
        return per_layer / (1024 ** 2)
    
    # FO mode: store all activations for backward pass
    # Per layer: input, attention, FFN intermediate
    per_layer = batch_size * seq_len * hidden_dim * bytes_per_element  # input
    per_layer += batch_size * seq_len * hidden_dim * 4 * bytes_per_element  # FFN
    per_layer += batch_size * 12 * seq_len * seq_len * bytes_per_element  # attention (approx)
    
    total = per_layer * num_layers
    return total / (1024 ** 2)


def get_split_model_memory_info(model, client_optimizer_mode="fo", server_optimizer_mode="fo", 
                                   optimizer_type="sgd", momentum=0.0,
                                   batch_size=64, seq_len=512):
    """
    Get detailed memory information for a split model (client/server).
    
    Returns dict with:
    - client_model_mb: Total model memory for client
    - client_trainable_mb: Trainable parameter memory for client
    - client_gradient_mb: Gradient + optimizer memory for client (0 if ZO)
    - server_model_mb: Total model memory for server  
    - server_trainable_mb: Trainable parameter memory for server
    - server_gradient_mb: Gradient + optimizer memory for server (0 if ZO)
    - total_model_mb: Total model memory
    - total_training_mb: Total memory needed during training
    - client_standalone_estimate_mb: Estimated GPU for client alone
    - server_standalone_estimate_mb: Estimated GPU for server alone
    """
    info = {
        "client_model_mb": 0.0,
        "client_trainable_mb": 0.0,
        "client_gradient_mb": 0.0,
        "server_model_mb": 0.0,
        "server_trainable_mb": 0.0,
        "server_gradient_mb": 0.0,
        "total_model_mb": 0.0,
        "total_training_mb": 0.0,
        "client_standalone_estimate_mb": 0.0,
        "server_standalone_estimate_mb": 0.0,
    }
    
    # Get model config for activation estimation
    hidden_dim = getattr(model.config, 'hidden_size', getattr(model.config, 'n_embd', 768))
    
    if hasattr(model, 'client'):
        info["client_model_mb"] = get_model_memory_mb(model.client)
        info["client_trainable_mb"] = get_trainable_memory_mb(model.client)
        if client_optimizer_mode == "fo":
            info["client_gradient_mb"] = get_gradient_memory_estimate_mb(
                model.client, optimizer_type, momentum)
        
        # Client standalone: model + output activation (embedding output)
        # Client only does embedding, so activation is batch × seq × hidden
        client_activation = batch_size * seq_len * hidden_dim * 4 / (1024 ** 2)
        info["client_standalone_estimate_mb"] = (
            info["client_model_mb"] + 
            info["client_gradient_mb"] + 
            client_activation
        )
        
    if hasattr(model, 'server'):
        info["server_model_mb"] = get_model_memory_mb(model.server)
        info["server_trainable_mb"] = get_trainable_memory_mb(model.server)
        if server_optimizer_mode == "fo":
            info["server_gradient_mb"] = get_gradient_memory_estimate_mb(
                model.server, optimizer_type, momentum)
        
        # Server standalone: model + gradients + activations
        # Estimate number of layers on server
        if hasattr(model.config, 'num_hidden_layers'):
            server_layers = model.config.num_hidden_layers  # All layers on server for OPT
        elif hasattr(model.config, 'n_layer'):
            server_layers = model.config.n_layer - 3  # GPT-2 split at layer 3
        else:
            server_layers = 9  # Default estimate
            
        server_activation = estimate_activation_memory_mb(
            batch_size, seq_len, hidden_dim, server_layers,
            store_for_backward=(server_optimizer_mode == "fo")
        )
        info["server_standalone_estimate_mb"] = (
            info["server_model_mb"] + 
            info["server_gradient_mb"] + 
            server_activation
        )
    
    info["total_model_mb"] = info["client_model_mb"] + info["server_model_mb"]
    info["total_training_mb"] = (info["client_model_mb"] + info["client_gradient_mb"] +
                                  info["server_model_mb"] + info["server_gradient_mb"])
    
    return info

def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

@dataclass
class OurArguments(TrainingArguments):
    # dataset and sampling strategy
    task_name: str = "SST2" # task name should match the string before Dataset in the Dataset class name. We support the following task_name: SST2, RTE, CB, BoolQ, WSC, WIC, MultiRC, Copa, ReCoRD, SQuAD, DROP
    # Number of examples
    num_train: int = 0 # ICL mode: number of demonstrations; training mode: number of training samples
    num_dev: int = None # (only enabled with training) number of development samples
    num_eval: int = None # number of evaluation samples
    num_train_sets: int = None # how many sets of training samples/demos to sample; if None and train_set_seed is None, then we will sample one set for each evaluation sample
    train_set_seed: int = None # designated seed to sample training samples/demos

    # Model loading
    model_name: str = "facebook/opt-125m" # HuggingFace model name
    load_float16: bool = False # load model parameters as float16
    load_bfloat16: bool = False # load model parameters as bfloat16
    load_int8: bool = False # load model parameters as int8
    max_length: int = 2048 # max length the model can take

    # If False (default), use Hugging Face `device_map="auto"` loading to shard the model
    # across available GPUs and reduce host RAM usage while loading checkpoint shards.
    # Set to True if you plan to manage device placement manually (e.g., custom FSDP).
    no_auto_device: bool = False

    # Training
    # Default to standard first-order (FO) training; MeZO (ZO) must be requested explicitly
    trainer: str = "regular" 
    ## options
    ## - none: no training -- for zero-shot or in-context learning (ICL)
    ## - regular: regular huggingface trainer -- for fine-tuning
    ## - zo: zeroth-order (MeZO) training

    # MeZO
    zo_eps: float = 1e-3 # eps in MeZO
    zo_continuous_rng: bool = True  # Use continuous RNG across client/server (faithful MeZO)
    # If False (default): client and server both reset to same seed (z values repeat)
    # If True: client perturbs first, saves RNG state, server continues (unique z per param)
    
    zo_variant: str = "central"  # "central" (two-point) or "forward" (one-point) gradient estimation
    # - "central": g = (f(θ+εz) - f(θ-εz)) / (2ε) -- 2 forward passes, lower bias
    # - "forward": g = (f(θ+εz) - f(θ)) / ε -- 1 forward pass, higher bias but faster
    
    zo_perturbation: str = "coordinate"  # "coordinate" (MeZO) or "layer" (DeComFL-style)
    # - "coordinate": each element gets independent random noise (original MeZO, default)
    # - "layer": each layer gets a normalized random direction (DeComFL-style)

    # Multiple perturbations (DeComFL-style variance reduction)
    num_pert: int = 1  # Number of perturbation vectors to use
    # - 1: Original MeZO (single perturbation)
    # - >1: DeComFL-style (average over K perturbations, reduces variance but K× more forward passes)

    # Per-machine optimizer modes (for split learning)
    # - "auto": follow `trainer` (FO when trainer=\"regular\", ZO when trainer=\"zo\")
    # - "fo": first-order (standard gradient-based)
    # - "zo": zeroth-order (MeZO-style)
    client_optimizer: str = "auto"
    server_optimizer: str = "auto"
    optimizer: str = "sgd"
    sgd_momentum: float = 0.0
    
    # Separate learning rates for client and server (for hybrid ZO/FO modes)
    # - If None, uses the global learning_rate for both
    # - Typically: ZO needs larger LR (1e-3 to 1e-4), FO needs smaller LR (1e-5 to 5e-5)
    client_learning_rate: Optional[float] = None
    server_learning_rate: Optional[float] = None

    only_train_option: bool = True # whether to only train the option part of the input
    train_as_classification: bool = False # take the log likelihood of all options and train as classification 
    non_diff: bool = False

    # Prefix tuning
    prefix_tuning: bool = False # whether to use prefix tuning
    num_prefix: int = 5 # number of prefixes to use
    no_reparam: bool = True # do not use reparameterization trick
    prefix_init_by_real_act: bool = True # initialize prefix by real activations of random words

    # LoRA
    lora: bool = False # whether to use LoRA
    lora_alpha: int = 16 # alpha in LoRA
    lora_r: int = 8 # r in LoRA
    
    # Split Learning
    init_checkpoint: str = None # path to pretrained checkpoint
    model_card: str = "gpt2.sm" # model card for split learning

    head_tuning: bool = False # head tuning: only tune the LM head

    # Display
    verbose: bool = False # verbose output

    # Logging / integrations
    # By default, disable external reporting (e.g., wandb/tensorboard) so that
    # no config JSON is sent anywhere unless explicitly requested on the CLI.
    report_to: Optional[List[str]] = field(default_factory=lambda: [])




class Framework:

    def __init__(self, args, task):
        self.args = args
        self.task = task
        
        # Reset GPU memory stats before loading model
        reset_gpu_memory_stats()
        
        self.model, self.tokenizer = self.load_model()
        
        # Determine optimizer modes
        global_trainer = getattr(self.args, "trainer", "regular")
        client_mode = getattr(self.args, "client_optimizer", "auto")
        server_mode = getattr(self.args, "server_optimizer", "auto")
        
        if client_mode == "auto":
            client_mode = "zo" if global_trainer == "zo" else "fo"
        if server_mode == "auto":
            server_mode = "zo" if global_trainer == "zo" else "fo"
        
        optimizer_type = getattr(self.args, "optimizer", "sgd")
        momentum = getattr(self.args, "sgd_momentum", 0.0)
        
        # Track model memory after loading with optimizer modes
        batch_size = getattr(self.args, "per_device_train_batch_size", 64)
        seq_len = getattr(self.args, "max_length", 512)
        
        self.model_memory_info = get_split_model_memory_info(
            self.model, 
            client_optimizer_mode=client_mode,
            server_optimizer_mode=server_mode,
            optimizer_type=optimizer_type,
            momentum=momentum,
            batch_size=batch_size,
            seq_len=seq_len
        )
        self.gpu_memory_after_model_load = get_gpu_memory_mb()
        
        logger.info(f"Optimizer Modes - Client: {client_mode.upper()}, Server: {server_mode.upper()}")
        logger.info(f"Model Memory - Client: {self.model_memory_info['client_model_mb']:.2f} MB, "
                   f"Server: {self.model_memory_info['server_model_mb']:.2f} MB")
        logger.info(f"Trainable Memory - Client: {self.model_memory_info['client_trainable_mb']:.2f} MB, "
                   f"Server: {self.model_memory_info['server_trainable_mb']:.2f} MB")
        logger.info(f"Gradient+Optimizer Memory - Client: {self.model_memory_info['client_gradient_mb']:.2f} MB, "
                   f"Server: {self.model_memory_info['server_gradient_mb']:.2f} MB")
        logger.info(f"Standalone GPU Estimate - Client: {self.model_memory_info['client_standalone_estimate_mb']:.2f} MB, "
                   f"Server: {self.model_memory_info['server_standalone_estimate_mb']:.2f} MB")
        logger.info(f"GPU Memory after model load: {self.gpu_memory_after_model_load:.2f} MB")

    
    def load_model(self):
        """
        Load Split Learning Models
        """
        with count_time("Loading split model"):
            if "opt" in self.args.model_name.lower():
                # Split OPT path: use HuggingFace OPTForCausalLM under the hood.
                logger.info(f"Loading split OPT model with pretrained weights: {self.args.model_name}")
                model = SplitOPT(self.args.model_name)
                # For OPT we currently fine-tune all parameters; LoRA masking is
                # not wired into the OPT architecture in this codebase.
                if self.args.lora:
                    logger.info(f"Enabling LoRA for OPT split (r={self.args.lora_r}, alpha={self.args.lora_alpha})")
                    # Apply to the server model (which contains the decoder layers)
                    apply_lora_to_opt(model.server, self.args.lora_r, self.args.lora_alpha, 0.0)
                    
                    # Mark only LoRA parameters as trainable
                    mark_only_lora_as_trainable(model.client)
                    mark_only_lora_as_trainable(model.server)
            else:
                # Split GPT-2 style models with built-in LoRA in the attention projections.
                if self.args.model_card == "gpt2.sm":
                    config = GPT2Config(
                        n_embd=768,
                        n_layer=12,
                        n_head=12,
                        lora_attn_dim=self.args.lora_r,
                        lora_attn_alpha=self.args.lora_alpha,
                        lora_dropout=0.0,
                    )
                elif self.args.model_card == "gpt2.md":
                    config = GPT2Config(
                        n_embd=1024,
                        n_layer=24,
                        n_head=16,
                        lora_attn_dim=self.args.lora_r,
                        lora_attn_alpha=self.args.lora_alpha,
                        lora_dropout=0.0,
                    )
                elif self.args.model_card == "gpt2.lg":
                    config = GPT2Config(
                        n_embd=1280,
                        n_layer=36,
                        n_head=20,
                        lora_attn_dim=self.args.lora_r,
                        lora_attn_alpha=self.args.lora_alpha,
                        lora_dropout=0.0,
                    )
                else:
                    # Fallback small config
                    config = GPT2Config(
                        lora_attn_dim=self.args.lora_r,
                        lora_attn_alpha=self.args.lora_alpha,
                        lora_dropout=0.0,
                    )

                model = SplitGPT2(config)
                
                # Load pretrained weights if model_name is a GPT-2 variant
                if "gpt2" in self.args.model_name.lower():
                    logger.info(f"Loading pretrained weights: {self.args.model_name}")
                    model.load_weight(self.args.model_name, split_layer=3)
                else:
                    logger.info("No pretrained weights loaded (random initialization)")

                # LoRA setup for the split GPT-2 model:
                # only LoRA parameters are trainable when LoRA is enabled.
                if self.args.lora and self.args.lora_r > 0:
                    mark_only_lora_as_trainable(model.client)
                    mark_only_lora_as_trainable(model.server)

            model.eval()
            
            # Handle dtype manually since we aren't using AutoModel.from_pretrained
            if self.args.load_float16:
                model.half()
            elif self.args.load_bfloat16:
                model.bfloat16()

            # Move to device
            if torch.cuda.is_available():
                model = model.cuda()

        # Load tokenizer
        tokenizer = AutoTokenizer.from_pretrained(self.args.model_name, use_fast=False)

        # HF tokenizer bug fix
        if "opt" in self.args.model_name:
            tokenizer.bos_token_id = 0
        
        if "llama" in self.args.model_name:
            # LLaMA padding token
            tokenizer.pad_token_id = 0 # technically <unk>

        # GPT-2 has no pad token by default; required for batch padding in collators.
        if "gpt2" in self.args.model_name.lower() and tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token

        # Prefix tuning can be applied to both HuggingFace models (OPT/Roberta/LLaMA)
        # and our split GPT-2 model (`SplitGPT2`) now that `prefix.PrefixTuning`
        # understands a GPT-2 style config (model_type=\"gpt2_split\").
        if self.args.prefix_tuning:
            from prefix import PrefixTuning
            PrefixTuning(model, num_prefix=self.args.num_prefix, reparam=not self.args.no_reparam, float16=self.args.load_float16, init_by_real_act=self.args.prefix_init_by_real_act)
        # if self.args.lora:
        #     from lora import LoRA
        #     LoRA(model, r=self.args.lora_r, alpha=self.args.lora_alpha, float16=self.args.load_float16)
        if self.args.head_tuning:
            if model.config.model_type == "opt":
                head_name = "lm_head"
            else:
                raise NotImplementedError
            for n, p in model.named_parameters():
                if head_name not in n:
                    p.requires_grad = False
                else:
                    logger.info(f"Only tuning {n}")

        return model, tokenizer


    def forward(self, input_ids, option_len=None):
        """
        Given input_ids and the length of the option, return the log-likelihood of each token in the option.
        This function is only for inference
        """
        input_ids = torch.tensor([input_ids]).to(self.model.device)

        with torch.inference_mode():
            self.model.eval()
            logits = self.model(input_ids=input_ids).logits
        labels = input_ids[0, 1:]  #ground truth
        logits = logits[0, :-1]    #prediction
        log_probs = F.log_softmax(logits, dim=-1)

        selected_log_probs = log_probs[torch.arange(len(labels)).to(labels.device), labels]
        selected_log_probs = selected_log_probs.cpu().detach()
        # Only return the option (candidate) part
        return selected_log_probs[-option_len:]


    def one_step_pred(self, train_samples, eval_sample, verbose=False):
        """
        Return the prediction on the eval sample. In ICL, use train_samples as demonstrations
        """
        verbose = verbose or self.args.verbose
        if verbose:
            logger.info("========= Example =========")
            logger.info(f"Candidate: {eval_sample.candidates}")
            logger.info(f"Correct candidate: {eval_sample.correct_candidate}")


        # Encode (add prompt and tokenize) the sample; if multiple-choice/classification, encode all candidates (options)
        encoded_candidates, option_lens = encode_prompt(
            self.task, self.task.get_template(), train_samples, eval_sample, self.tokenizer, max_length=self.args.max_length
        )

        outputs = []
        # For classification/multiple-choice, calculate the probabilities of all candidates
        for candidate_id, encoded_candidate in enumerate(encoded_candidates):
            selected_log_probs = self.forward(encoded_candidate, option_len=option_lens[candidate_id])
            if verbose:
                if candidate_id == 0:
                    logger.info("=== Candidate %d ===" % candidate_id)
                    logger.info(self.tokenizer.decode(encoded_candidate))
                else:
                    logger.info("=== Candidate %d (without context)===" % candidate_id)
                    logger.info(self.tokenizer.decode(encoded_candidate).split(self.task.train_sep)[-1])
                logger.info(f"Log probabilities of the option tokens: {selected_log_probs}")

            outputs.append(selected_log_probs)

        # log p(candidate | input) = log p_lm(candidate | input) / |candidate #tokens|
        scores = [x.mean().item() for x in outputs]

        if verbose:
            logger.info(f"Prediction scores: {scores}")

        if isinstance(eval_sample.correct_candidate, list):
            # For some datasets there are multiple correct answers
            correct_candidate_id = [eval_sample.candidates.index(c) for c in eval_sample.correct_candidate]
        else:
            correct_candidate_id = eval_sample.candidates.index(eval_sample.correct_candidate)

        return Prediction(correct_candidate=correct_candidate_id, predicted_candidate=int(np.argmax(scores)))


    def evaluate(self, train_samples, eval_samples):
        """
        Evaluate function. If one_train_set_per_eval_sample is True, then each eval sample has its own training (demonstration) set.
        """
        logger.info(f"There are {len(train_samples)} training samples and {len(eval_samples)} validation samples")

        # Prediction loop
        predictions = []
        for eval_id, eval_sample in enumerate(tqdm(eval_samples)):
            
            predictions.append(
                self.one_step_pred(train_samples, eval_sample, verbose=(eval_id < 3))
            )

        # Calculate metrics 
        metrics = {"accuracy": calculate_metric(predictions)}
        
        return metrics


    def train(self, train_samples, eval_samples):
        """
        Training function
        """
        # Reset peak memory stats before training
        reset_gpu_memory_stats()
        
        # Enable memory tracking on the model
        if hasattr(self.model, 'enable_memory_tracking'):
            self.model.enable_memory_tracking(True)
        
        # Set tokenizer to left padding (so that all the options are right aligned)
        self.tokenizer.padding_side = "left"

        class HFDataset(Dataset):

            def __init__(self, data):
                self.data = data

            def __len__(self):
                return len(self.data)

            def __getitem__(self, idx):
                return self.data[idx]


        def _convert(samples):
            """
            Convert samples to HF-compatible dataset
            """
            data = []
            for sample in samples:
                encoded_candidates, option_lens = encode_prompt(
                    self.task, self.task.get_template(), [], sample, self.tokenizer, 
                    max_length=self.args.max_length)
                if isinstance(sample.correct_candidate, list):
                    correct_candidate_id = sample.candidates.index(sample.correct_candidate[0])
                else:
                    correct_candidate_id = sample.candidates.index(sample.correct_candidate)
                
                if self.args.train_as_classification:
                    # For classification, we provide the label as the correct candidate id
                    data.append([{"input_ids": encoded_candidates[_i], "labels": correct_candidate_id, "option_len": option_lens[_i], "num_options": len(sample.candidates)} for _i in range(len(encoded_candidates))])
                elif self.args.only_train_option:
                    # Otherwise, it is just LM-style teacher forcing
                    if self.args.non_diff:
                        # For non-differentiable objective, we need to provide the gold answer to calculate F1/acc
                        data.append({"input_ids": encoded_candidates[correct_candidate_id], "labels": encoded_candidates[correct_candidate_id], "option_len": option_lens[correct_candidate_id], "gold": sample.correct_candidate})
                    else:
                        data.append({"input_ids": encoded_candidates[correct_candidate_id], "labels": encoded_candidates[correct_candidate_id], "option_len": option_lens[correct_candidate_id]})
                else:
                    data.append({"input_ids": encoded_candidates[correct_candidate_id], "labels": encoded_candidates[correct_candidate_id]})
            return data

        with count_time("Tokenizing training samples"):
            train_dataset = HFDataset(_convert(train_samples))
            eval_dataset = HFDataset(_convert(eval_samples))

        if self.args.only_train_option and not self.args.non_diff:
            # If --only_train_option and not with a non-differentiable objective, we wrap the forward function
            self.model.original_forward = self.model.forward
            self.model.forward = forward_wrap_with_option_len.__get__(self.model, type(self.model))

        if self.args.non_diff:
            collator = NondiffCollator
        else:
            collator = DataCollatorForTokenClassification

        
        trainer = OurTrainer(
            model=self.model, 
            args=self.args,
            train_dataset=train_dataset, 
            eval_dataset=eval_dataset,
            tokenizer=self.tokenizer,
            data_collator=DataCollatorWithPaddingAndNesting(self.tokenizer, pad_to_multiple_of=8) if self.args.train_as_classification else collator(self.tokenizer, pad_to_multiple_of=8),
        )

        # Override trainer.evaluate to report accuracy
        original_evaluate = trainer.evaluate
        def evaluate_with_accuracy(eval_dataset=None, ignore_keys=None, metric_key_prefix="eval"):
            # Run standard evaluation (gets loss)
            metrics = original_evaluate(eval_dataset, ignore_keys, metric_key_prefix)
            
            # Run custom evaluation (gets accuracy)
            # Ensure self.model points to the trainer's model
            self.model = trainer.model
            logger.info("Running custom accuracy evaluation...")
            custom_metrics = self.evaluate([], eval_samples)
            
            # Merge metrics
            metrics[f"{metric_key_prefix}_accuracy"] = custom_metrics["accuracy"]
            logger.info(f"***** Eval Accuracy: {custom_metrics['accuracy']:.4f} *****")
            
            return metrics
        
        trainer.evaluate = evaluate_with_accuracy

        trainer.train() 
        
        # Track peak GPU memory after training
        self.peak_gpu_memory_mb = get_gpu_max_memory_mb()
        self.num_training_rounds = self.args.max_steps
        
        # Calculate communication rounds for split learning
        # Each forward pass = 1 communication (client sends activations to server)
        # For MeZO: central variant uses 2 forwards per perturbation, forward variant uses 1 baseline + 1 per perturbation
        num_pert = getattr(self.args, 'num_pert', 1)
        zo_variant = getattr(self.args, 'zo_variant', 'central')
        trainer_type = getattr(self.args, 'trainer', 'none')
        
        if trainer_type == 'zo':
            if zo_variant == 'forward':
                # 1 baseline forward + 1 forward per perturbation
                forwards_per_step = 1 + num_pert
            else:  # central (default)
                # 2 forwards per perturbation (for +ε and -ε)
                forwards_per_step = 2 * num_pert
            self.num_communication_rounds = self.num_training_rounds * forwards_per_step
        else:
            # First-order: 1 forward + 1 backward per step = 2 communications
            self.num_communication_rounds = self.num_training_rounds * 2
        
        # Get client/server peak memory if tracked
        if hasattr(self.model, 'client_peak_memory_mb'):
            self.client_peak_memory_mb = self.model.client_peak_memory_mb
            self.server_peak_memory_mb = self.model.server_peak_memory_mb
        else:
            self.client_peak_memory_mb = 0.0
            self.server_peak_memory_mb = 0.0
        
        # Calculate standalone GPU estimates (what each device would need separately)
        # Server peak currently includes client model in memory, so subtract it
        client_model_mb = self.model_memory_info.get("client_model_mb", 0)
        server_model_mb = self.model_memory_info.get("server_model_mb", 0)
        
        # Standalone estimates: subtract the other model's memory from peak
        self.client_standalone_mb = self.client_peak_memory_mb  # Client doesn't include server
        self.server_standalone_mb = max(0, self.server_peak_memory_mb - client_model_mb)  # Subtract client model
        
        logger.info(f"Peak GPU Memory during training: {self.peak_gpu_memory_mb:.2f} MB")
        logger.info(f"Client Peak Memory (measured): {self.client_peak_memory_mb:.2f} MB")
        logger.info(f"Server Peak Memory (measured): {self.server_peak_memory_mb:.2f} MB")
        logger.info(f"Client Standalone Estimate: {self.client_standalone_mb:.2f} MB")
        logger.info(f"Server Standalone Estimate: {self.server_standalone_mb:.2f} MB")
        logger.info(f"Number of training rounds: {self.num_training_rounds}")
        logger.info(f"Number of communication rounds: {self.num_communication_rounds}")
        
        # FSDP compatibility
        self.model = trainer.model 

        if type(self.model) == FSDP:
            logger.info("This is an FSDP model now. Be careful when assigning back the original forward function")
            self.model._fsdp_wrapped_module.forward = self.model._fsdp_wrapped_module.original_forward
        elif isinstance(self.model, torch.nn.parallel.DistributedDataParallel):
            # Handle DDP case
            logger.info("Restoring forward function for DDP model")
            self.model.module.forward = self.model.module.original_forward
        else:
            self.model.forward = self.model.original_forward


def parse_args():
    parser = argparse.ArgumentParser()
    parser = HfArgumentParser(OurArguments)
    args = parser.parse_args_into_dataclasses()[0]
    print(args)
    return args
    
def main():
    args = parse_args()

    set_seed(args.seed)
    task = get_task(args.task_name)
    train_sets = task.sample_train_sets(num_train=args.num_train, num_dev=args.num_dev, num_eval=args.num_eval, num_train_sets=args.num_train_sets, seed=args.train_set_seed)

    # Initialize trainer and load model
    framework = Framework(args, task)

    if args.train_set_seed is not None or args.num_train_sets is not None:
        # Eval samples share one (or multiple) training set(s)
        for train_set_id, train_samples in enumerate(train_sets):
            train_set_seed = train_set_id if args.train_set_seed is None else args.train_set_seed

            # Sample eval samples
            if args.num_eval is not None:
                eval_samples = task.sample_subset(data_split="valid", seed=train_set_seed, num=args.num_eval)
            else:
                eval_samples = task.valid_samples

            if args.trainer != "none":
                if args.num_dev is not None:
                    # Dev samples
                    dev_samples = train_samples[-args.num_dev:] 
                    train_samples = train_samples[:-args.num_dev]
                else:
                    dev_samples = None

                # Training
                framework.train(train_samples, dev_samples if dev_samples is not None else eval_samples)

                metrics = framework.evaluate([], eval_samples) # No in-context learning if there is training
                if dev_samples is not None:
                    dev_metrics = framework.evaluate([], dev_samples) 
                    for m in dev_metrics:
                        metrics["dev_" + m] = dev_metrics[m]
                
                # Add GPU memory and training round metrics
                metrics["client_model_memory_mb"] = framework.model_memory_info["client_model_mb"]
                metrics["server_model_memory_mb"] = framework.model_memory_info["server_model_mb"]
                metrics["total_model_memory_mb"] = framework.model_memory_info["total_model_mb"]
                # Gradient/optimizer memory (0 for ZO mode - this shows the ZO savings)
                metrics["client_gradient_memory_mb"] = framework.model_memory_info["client_gradient_mb"]
                metrics["server_gradient_memory_mb"] = framework.model_memory_info["server_gradient_mb"]
                metrics["total_training_memory_mb"] = framework.model_memory_info["total_training_mb"]
                metrics["peak_gpu_memory_mb"] = framework.peak_gpu_memory_mb
                # Separate client/server peak memory (actual GPU required for each)
                metrics["client_peak_memory_mb"] = getattr(framework, 'client_peak_memory_mb', 0.0)
                metrics["server_peak_memory_mb"] = getattr(framework, 'server_peak_memory_mb', 0.0)
                # Standalone estimates (what each device needs when running separately)
                metrics["client_standalone_mb"] = getattr(framework, 'client_standalone_mb', 0.0)
                metrics["server_standalone_mb"] = getattr(framework, 'server_standalone_mb', 0.0)
                metrics["num_training_rounds"] = framework.num_training_rounds
                metrics["num_communication_rounds"] = framework.num_communication_rounds

            logger.info("===== Train set %d =====" % train_set_seed)
            logger.info(metrics)
            print("results: ", metrics)
            
            # Print summary in a more readable format
            print("\n" + "=" * 60)
            print("TRAINING SUMMARY")
            print("=" * 60)
            print(f"Accuracy:                    {metrics.get('accuracy', 'N/A'):.4f}" if isinstance(metrics.get('accuracy'), float) else f"Accuracy:                    {metrics.get('accuracy', 'N/A')}")
            print(f"Number of Training Rounds:   {metrics.get('num_training_rounds', 'N/A')}")
            print(f"Number of Comm Rounds:       {metrics.get('num_communication_rounds', 'N/A')}")
            print(f"Client Model Memory:         {metrics.get('client_model_memory_mb', 0):.2f} MB")
            print(f"Client Gradient Memory:      {metrics.get('client_gradient_memory_mb', 0):.2f} MB (0 if ZO)")
            print(f"Server Model Memory:         {metrics.get('server_model_memory_mb', 0):.2f} MB")
            print(f"Server Gradient Memory:      {metrics.get('server_gradient_memory_mb', 0):.2f} MB (0 if ZO)")
            print(f"Total Model Memory:          {metrics.get('total_model_memory_mb', 0):.2f} MB")
            print(f"Total Training Memory:       {metrics.get('total_training_memory_mb', 0):.2f} MB")
            print("-" * 60)
            print("PEAK GPU MEMORY (actual GPU required):")
            print(f"  Client Peak Memory:        {metrics.get('client_peak_memory_mb', 0):.2f} MB")
            print(f"  Server Peak Memory:        {metrics.get('server_peak_memory_mb', 0):.2f} MB")
            print(f"  Total Peak Memory:         {metrics.get('peak_gpu_memory_mb', 0):.2f} MB")
            print("-" * 60)
            
            # Estimate communication costs (DeComFL-style)
            # Get model info for estimation
            total_params = sum(p.numel() for p in framework.model.parameters())
            trainable_params = sum(p.numel() for p in framework.model.parameters() if p.requires_grad)
            
            # Estimate hidden state size at split point (batch_size × seq_len × hidden_dim)
            # Using typical values for estimation
            batch_size = args.per_device_train_batch_size
            max_seq_len = getattr(args, 'max_length', 512)
            hidden_dim = getattr(framework.model.client_model.config if hasattr(framework.model, 'client_model') else framework.model.config, 
                                'hidden_size', 768)
            
            # dtype size (4 bytes for fp32, 2 for fp16)
            dtype_bytes = 2 if args.load_float16 or args.load_bfloat16 else 4
            
            # Per-forward communication cost (Split Learning)
            # Forward: hidden_states (batch × seq × hidden)
            activation_size = batch_size * max_seq_len * hidden_dim * dtype_bytes
            # Backward in ZO: just loss scalar (4 bytes) + seed (8 bytes)
            zo_backward_size = 12  # loss (float) + seed (int64)
            # Backward in FO: gradients same size as activations
            fo_backward_size = activation_size
            
            num_comm_rounds = metrics.get('num_communication_rounds', 0)
            is_zo = args.trainer == "zo"
            
            if is_zo:
                # ZO: forward activations + loss scalar + seed
                per_round_bytes = activation_size + zo_backward_size
            else:
                # FO: forward activations + backward gradients  
                per_round_bytes = activation_size + fo_backward_size
            
            total_bytes = per_round_bytes * num_comm_rounds
            
            # Traditional FL comparison (would send full model each round)
            traditional_fl_per_round = 2 * total_params * dtype_bytes  # send + receive full model
            traditional_fl_total = traditional_fl_per_round * metrics.get('num_training_rounds', 0)
            
            # DeComFL pure (only scalars + seeds)
            num_pert = getattr(args, 'num_pert', 1)
            decomfl_per_round = num_pert * 4 + 8  # P scalars + seed
            decomfl_total = decomfl_per_round * num_comm_rounds
            
            print("ESTIMATED COMMUNICATION COST (DeComFL-style):")
            print(f"  Per Forward+Backward:      {format_payload_size(per_round_bytes)}")
            print(f"  Total ({num_comm_rounds} rounds):      {format_payload_size(total_bytes)}")
            print(f"  Traditional FL would use:  {format_payload_size(traditional_fl_total)}")
            print(f"  DeComFL pure would use:    {format_payload_size(decomfl_total)}")
            if traditional_fl_total > 0:
                savings = (1 - total_bytes / traditional_fl_total) * 100
                print(f"  Savings vs Traditional FL: {savings:.2f}%")
            print("=" * 60 + "\n")
            
            # Add estimated communication metrics
            metrics["estimated_total_bytes"] = total_bytes
            metrics["estimated_total_bytes_formatted"] = format_payload_size(total_bytes)
            metrics["traditional_fl_bytes"] = traditional_fl_total
            metrics["decomfl_pure_bytes"] = decomfl_total
            if traditional_fl_total > 0:
                metrics["savings_vs_traditional_fl"] = 1 - (total_bytes / traditional_fl_total)

if __name__ == "__main__": 
    main()