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
    lora: bool = True # whether to use LoRA
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
        self.model, self.tokenizer = self.load_model()

    
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

            logger.info("===== Train set %d =====" % train_set_seed)
            logger.info(metrics)
            print("results: ", metrics)

if __name__ == "__main__": 
    main()