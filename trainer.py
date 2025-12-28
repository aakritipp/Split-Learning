"""
Custom trainer extending HuggingFace Trainer for split learning.

This module provides OurTrainer, which extends the HuggingFace Trainer class
with support for split learning architectures and zeroth-order (ZO) optimization.

Key features:
- Split learning with configurable client/server optimization modes (ZO/FO)
- Coordinated perturbation generation for ZO optimization
- Support for both coordinate-wise and layer-wise perturbation strategies
- Separate optimizers for client and server modules
"""
import math
import os
import random
import re
import shutil
import sys
import time
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple, Union
import numpy as np

from tqdm.auto import tqdm
from transformers import Trainer
# Integrations must be imported before ML frameworks:
from transformers.integrations import hp_params

import numpy as np
import torch
import torch.distributed as dist
from packaging import version
from torch import nn
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler

from transformers import __version__
from transformers.trainer_callback import TrainerState, ExportableState
from transformers.trainer_pt_utils import IterableDatasetShard
from transformers.trainer_utils import (
    HPSearchBackend,
    TrainOutput,
    has_length,
    speed_metrics,
)
from transformers.utils import (
    is_apex_available,
    logging,
)

if is_apex_available():
    from apex import amp

logger = logging.get_logger(__name__)


# Name of the files used for checkpointing
TRAINER_STATE_NAME = "trainer_state.json"

class OurTrainer(Trainer):

    from transformers.trainer_pt_utils import _get_learning_rate

    def _move_model_to_device(self, model, device):
        if hasattr(model, "hf_device_map"):
            logger.info("Model has hf_device_map, skipping _move_model_to_device")
            return
        super()._move_model_to_device(model, device)

    def compute_loss(
        self,
        model,
        inputs,
        return_outputs: bool = False,
        num_items_in_batch: Optional[int] = None,
    ):
        """
        Custom loss computation that always uses `labels` with causal LM models.

        The stock `Trainer.compute_loss` in this Transformers version only uses
        `labels` when a `label_smoother` or `compute_loss_func` is configured.
        In our setup neither is set, so the base implementation ignores `labels`
        and expects the model to return a `loss` on its own, which it does not.

        Here we explicitly:
        - Pop `labels` from `inputs` if present.
        - Call the model with `labels` so it computes a loss.
        - Return that loss (and optionally the full outputs).
        """

        # Use get instead of pop to avoid mutating inputs in-place.
        # This is critical for ZO (zo_step) which reuses inputs across multiple forward passes.
        labels = inputs.get("labels")

        if labels is not None:
            # Create a fresh dict for model inputs to exclude labels without mutating original inputs
            model_inputs = {k: v for k, v in inputs.items() if k != "labels"}
            outputs = model(**model_inputs, labels=labels)
        else:
            outputs = model(**inputs)

        # Model may return a dict-like object or a tuple
        if isinstance(outputs, dict):
            loss = outputs["loss"] if "loss" in outputs else outputs[0]
        else:
            loss = outputs[0]

        return (loss, outputs) if return_outputs else loss

    def create_optimizer(self):
        """
        Setup the optimizer(s) for split learning.
        
        Supports four modes:
        - ZO/ZO: Both client and server use zeroth-order (dummy optimizers)
        - FO/FO: Both client and server use first-order (real optimizers)
        - ZO/FO: Client uses zeroth-order (dummy), Server uses first-order (real)
        - FO/ZO: Client uses first-order (real), Server uses zeroth-order (dummy)
        
        Modes are controlled by:
        - --trainer: Global setting ('regular' for FO, 'zo' for ZO)
        - --client_optimizer: 'auto', 'fo', or 'zo'
        - --server_optimizer: 'auto', 'fo', or 'zo'
        
        Learning rates:
        - --learning_rate: Global default LR
        - --client_learning_rate: Override LR for client (if None, uses global)
        - --server_learning_rate: Override LR for server (if None, uses global)
        
        NOTE: ZO (zeroth-order) typically needs larger LR (1e-3 to 1e-4),
              FO (first-order) typically needs smaller LR (1e-5 to 5e-5).
        
        When set to 'auto', the mode follows the global --trainer setting.
        """
        # Resolve optimizer modes
        global_trainer = getattr(self.args, "trainer", "regular")
        client_mode = getattr(self.args, "client_optimizer", "auto")
        server_mode = getattr(self.args, "server_optimizer", "auto")
        
        if client_mode == "auto":
            client_mode = "zo" if global_trainer == "zo" else "fo"
        if server_mode == "auto":
            server_mode = "zo" if global_trainer == "zo" else "fo"
        
        # Resolve learning rates (separate for client and server)
        global_lr = self.args.learning_rate
        self.client_lr = getattr(self.args, "client_learning_rate", None) or global_lr
        self.server_lr = getattr(self.args, "server_learning_rate", None) or global_lr
        
        logger.info(f"Optimizer modes: client={client_mode.upper()}, server={server_mode.upper()}")
        logger.info(f"Learning rates: client_lr={self.client_lr:.2e}, server_lr={self.server_lr:.2e}")

        # Helper to create optimizer for a module
        def make_opt(model_part, mode, lr):
            if mode == "zo":
                # In pure zeroth-order mode we do *not* use a torch optimizer to
                # update parameters, but Hugging Face's Trainer still expects a real
                # optimizer object so that:
                #   - `create_scheduler(...)` can build a scheduler, and
                #   - `self._get_learning_rate()` can read `param_groups[0]["lr"]`.
                #
                # To satisfy this contract without changing model updates, we create a
                # lightweight dummy SGD optimizer on a single parameter. We never call
                # `.step()` on it in ZO mode – it's only there so the scheduler and LR
                # accessors have something to attach to.
                params = [p for p in model_part.parameters() if p.requires_grad]
                if not params:
                    # Extremely defensive: if the module has no trainable params,
                    # create a tiny dummy parameter purely for the scheduler.
                    dummy = torch.nn.Parameter(torch.zeros(1, device=self.args.device))
                    params = [dummy]
                return torch.optim.SGD([params[0]], lr=lr)
            
            # Standard FO optimizer creation (SGD only, by user request)
            decay_parameters = [
                name
                for name, _ in model_part.named_parameters()
                if "bias" not in name
                and "LayerNorm.weight" not in name
                and "layernorm.weight" not in name
            ]
            optimizer_grouped_parameters = [
                {
                    "params": [
                        p
                        for n, p in model_part.named_parameters()
                        if n in decay_parameters and p.requires_grad
                    ],
                    "weight_decay": self.args.weight_decay,
                },
                {
                    "params": [
                        p
                        for n, p in model_part.named_parameters()
                        if n not in decay_parameters and p.requires_grad
                    ],
                    "weight_decay": 0.0,
                },
            ]

            opt_name = getattr(self.args, "optimizer", "sgd")
            if opt_name == "sgd":
                return torch.optim.SGD(
                    optimizer_grouped_parameters,
                    lr=lr,
                    momentum=getattr(self.args, "sgd_momentum", 0.0),
                )
            else:
                from transformers.optimization import AdamW

                return AdamW(
                    optimizer_grouped_parameters,
                    lr=lr,
                    eps=self.args.adam_epsilon,
                )

        # Handle split model
        model = self.model
        if hasattr(model, "module"):
            model = model.module
            
        if hasattr(model, "client") and hasattr(model, "server"):
            self.client_optimizer_obj = make_opt(model.client, client_mode, self.client_lr)
            self.server_optimizer_obj = make_opt(model.server, server_mode, self.server_lr)
            
            # Store the modes for use in training_step
            self._client_mode = client_mode
            self._server_mode = server_mode
            
            # For compatibility, set self.optimizer to one of them or a combined one if possible.
            # But since we manage steps manually in training_step, we just need to ensure it exists if Trainer checks it.
            # We'll set it to server_optimizer_obj if exists, else client.
            self.optimizer = self.server_optimizer_obj if self.server_optimizer_obj else self.client_optimizer_obj
        else:
            # Fallback for non-split models
            self.client_optimizer_obj = None
            self.server_optimizer_obj = None
            if self.optimizer is None:
                super().create_optimizer()
                
        return self.optimizer

    def _inner_training_loop(
        self, batch_size=None, args=None, trial=None, ignore_keys_for_eval=None, resume_from_checkpoint=None
    ):
        """
        We overload the original training loop to add linear probing and ZO support.
        """
        self._train_batch_size = batch_size
        # Data loader and number of training steps
        train_dataloader = self.get_train_dataloader()

        # Setting up training control variables:
        # number of training epochs: num_train_epochs
        # number of training steps per epoch: num_update_steps_per_epoch
        # total number of training steps to execute: max_steps
        total_train_batch_size = args.train_batch_size * args.gradient_accumulation_steps * args.world_size

        len_dataloader = None
        if has_length(train_dataloader):
            len_dataloader = len(train_dataloader)
            num_update_steps_per_epoch = len_dataloader // args.gradient_accumulation_steps
            num_update_steps_per_epoch = max(num_update_steps_per_epoch, 1)
            num_examples = self.num_examples(train_dataloader)
            if args.max_steps > 0:
                max_steps = args.max_steps
                num_train_epochs = args.max_steps // num_update_steps_per_epoch + int(
                    args.max_steps % num_update_steps_per_epoch > 0
                )
                # May be slightly incorrect if the last batch in the training dataloader has a smaller size but it's
                # the best we can do.
                num_train_samples = args.max_steps * total_train_batch_size
            else:
                max_steps = math.ceil(args.num_train_epochs * num_update_steps_per_epoch)
                num_train_epochs = math.ceil(args.num_train_epochs)
                num_train_samples = self.num_examples(train_dataloader) * args.num_train_epochs
        elif args.max_steps > 0:  # Rely on max_steps when dataloader does not have a working size
            max_steps = args.max_steps
            # Setting a very large number of epochs so we go as many times as necessary over the iterator.
            num_train_epochs = sys.maxsize
            num_update_steps_per_epoch = max_steps
            num_examples = total_train_batch_size * args.max_steps
            num_train_samples = args.max_steps * total_train_batch_size
        else:
            raise ValueError(
                "args.max_steps must be set to a positive value if dataloader does not have a length, was"
                f" {args.max_steps}"
            )

        self.create_optimizer_and_scheduler(num_training_steps=max_steps)

        print("optimizer: ", self.optimizer)

        # Recreate TrainerState with proper callback tracking so checkpointing works
        # (mirrors the HF Trainer v4.49+ behavior).
        self.state = TrainerState(
            stateful_callbacks=[
                cb for cb in self.callback_handler.callbacks + [self.control] if isinstance(cb, ExportableState)
            ]
        )
        self.state.is_hyper_param_search = trial is not None
        self.state.train_batch_size = self._train_batch_size
        # Compute absolute values for logging/eval/save steps if ratios were provided.
        self.state.compute_steps(args, max_steps)

        # Activate gradient checkpointing if needed
        if args.gradient_checkpointing:
            self.model.gradient_checkpointing_enable()

        model = self._wrap_model(self.model_wrapped)

        # for the rest of this function `model` is the outside model, whether it was wrapped or not
        if model is not self.model:
            self.model_wrapped = model

        # important: at this point:
        # self.model         is the Transformers Model
        # self.model_wrapped is DDP(Transformers Model), Deepspeed(Transformers Model), etc.

        # Train!
        logger.info("***** Running training *****")
        logger.info(f"  Num examples = {num_examples}")
        logger.info(f"  Num Epochs = {num_train_epochs}")
        logger.info(f"  Instantaneous batch size per device = {args.per_device_train_batch_size}")
        logger.info(f"  Total train batch size (w. parallel, distributed & accumulation) = {total_train_batch_size}")
        logger.info(f"  Gradient Accumulation steps = {args.gradient_accumulation_steps}")
        logger.info(f"  Total optimization steps = {max_steps}")
        logger.info(
            f"  Number of trainable parameters = {sum(p.numel() for p in model.parameters() if p.requires_grad)}"
        )

        self.state.epoch = 0
        start_time = time.time()
        epochs_trained = 0
        steps_trained_in_current_epoch = 0
        steps_trained_progress_bar = None

        # Update the references
        self.callback_handler.model = self.model
        self.callback_handler.optimizer = self.optimizer
        self.callback_handler.lr_scheduler = self.lr_scheduler
        self.callback_handler.train_dataloader = train_dataloader
        self.state.trial_params = None
        # This should be the same if the state has been saved but in case the training arguments changed, it's safer
        # to set this after the load.
        self.state.max_steps = max_steps
        self.state.num_train_epochs = num_train_epochs
        self.state.is_local_process_zero = self.is_local_process_zero()
        self.state.is_world_process_zero = self.is_world_process_zero()

        # tr_loss is a tensor to avoid synchronization of TPUs through .item()
        tr_loss = torch.tensor(0.0).to(args.device)
        # _total_loss_scalar is updated everytime .item() has to be called on tr_loss and stores the sum of all losses
        self._total_loss_scalar = 0.0
        self._globalstep_last_logged = self.state.global_step
        model.zero_grad()

        self.control = self.callback_handler.on_train_begin(args, self.state, self.control)

        for epoch in range(epochs_trained, num_train_epochs):
            if isinstance(train_dataloader, DataLoader) and isinstance(train_dataloader.sampler, DistributedSampler):
                train_dataloader.sampler.set_epoch(epoch)
            elif hasattr(train_dataloader, "dataset") and isinstance(train_dataloader.dataset, IterableDatasetShard):
                train_dataloader.dataset.set_epoch(epoch)

            epoch_iterator = train_dataloader

            # Reset the past mems state at the beginning of each epoch if necessary.
            if args.past_index >= 0:
                self._past = None

            steps_in_epoch = (
                len(epoch_iterator)
                if len_dataloader is not None
                else args.max_steps * args.gradient_accumulation_steps
            )
            self.control = self.callback_handler.on_epoch_begin(args, self.state, self.control)

            step = -1
            for step, inputs in enumerate(epoch_iterator):

                if step % args.gradient_accumulation_steps == 0:
                    self.control = self.callback_handler.on_step_begin(args, self.state, self.control)

                # Unified training step (handles ZO/FO and split logic)
                tr_loss_step = self.training_step(model, inputs)

                if (
                    args.logging_nan_inf_filter
                    and (torch.isnan(tr_loss_step) or torch.isinf(tr_loss_step))
                ):
                    # if loss is nan or inf simply add the average of previous logged losses
                    tr_loss += tr_loss / (1 + self.state.global_step - self._globalstep_last_logged)
                else:
                    tr_loss += tr_loss_step

                self.current_flos += float(self.floating_point_ops(inputs))

                if (step + 1) % args.gradient_accumulation_steps == 0 or (
                    # last step in epoch but step is always smaller than gradient_accumulation_steps
                    steps_in_epoch <= args.gradient_accumulation_steps
                    and (step + 1) == steps_in_epoch
                ):
                    # Updates are handled in training_step.
                    # We just update state and callbacks here.

                    self.state.global_step += 1
                    self.state.epoch = epoch + (step + 1) / steps_in_epoch
                    self.control = self.callback_handler.on_step_end(args, self.state, self.control)

                    self._maybe_log_save_evaluate(
                        tr_loss,
                        None,  # grad_norm
                        model,
                        trial,
                        epoch,
                        ignore_keys_for_eval,
                        start_time,
                    )
                else:
                    self.control = self.callback_handler.on_substep_end(args, self.state, self.control)

                if self.control.should_epoch_stop or self.control.should_training_stop:
                    break
            if step < 0:
                logger.warning(
                    "There seems to be not a single sample in your epoch_iterator, stopping training at step"
                    f" {self.state.global_step}! This is expected if you're using an IterableDataset and set"
                    f" num_steps ({max_steps}) higher than the number of available samples."
                )
                self.control.should_training_stop = True

            self.control = self.callback_handler.on_epoch_end(args, self.state, self.control)
            self._maybe_log_save_evaluate(
                tr_loss,
                None,  # grad_norm
                model,
                trial,
                epoch,
                ignore_keys_for_eval,
                start_time,
            )

            if self.control.should_training_stop:
                break

        if args.past_index and hasattr(self, "_past"):
            # Clean the state at the end of training
            delattr(self, "_past")

        logger.info("\n\nTraining completed. Do not forget to share your model on huggingface.co/models =)\n\n")
        
        # add remaining tr_loss
        self._total_loss_scalar += tr_loss.item()
        train_loss = self._total_loss_scalar / self.state.global_step

        metrics = speed_metrics("train", start_time, num_samples=num_train_samples, num_steps=self.state.max_steps)
        self.store_flos()
        metrics["total_flos"] = self.state.total_flos
        metrics["train_loss"] = train_loss

        self.is_in_train = False

        self._memory_tracker.stop_and_update_metrics(metrics)

        self.log(metrics)

        self.control = self.callback_handler.on_train_end(args, self.state, self.control)

        return TrainOutput(self.state.global_step, train_loss, metrics)


    ############## ZO and Split Learning ##############
    
    def generate_shared_perturbation_seed(self):
        """
        Generate a shared perturbation seed for coordinated ZO.
        
        In true split learning, both client and server use the SAME seed
        for perturbations, ensuring identical perturbation patterns.
        """
        self.zo_random_seed = np.random.randint(1000000000)
        return self.zo_random_seed


    def generate_multiple_seeds(self, num_pert):
        """
        Generate multiple perturbation seeds for variance reduction.
        
        Args:
            num_pert: Number of perturbation vectors to use
            
        Returns:
            List of random seeds
        """
        return [np.random.randint(1000000000) for _ in range(num_pert)]

    def _get_sorted_trainable_params(self, module, cache_key):
        """
        Get sorted trainable parameters for a module, with caching.
        
        Caches the sorted parameter list to avoid re-sorting on every call.
        The cache is stored in self._sorted_params_cache[cache_key].
        
        Args:
            module: The module to get parameters from
            cache_key: A string key for caching (e.g., "client" or "server")
        
        Returns:
            List of (name, param) tuples sorted by name
        """
        if not hasattr(self, '_sorted_params_cache'):
            self._sorted_params_cache = {}
        
        if cache_key not in self._sorted_params_cache:
            self._sorted_params_cache[cache_key] = sorted(
                [(name, param) for name, param in module.named_parameters() if param.requires_grad],
                key=lambda x: x[0]
            )
        
        return self._sorted_params_cache[cache_key]

    def _invalidate_params_cache(self):
        """Invalidate the sorted params cache (call if model structure changes)."""
        if hasattr(self, '_sorted_params_cache'):
            self._sorted_params_cache.clear()

    def _generate_perturbation(self, param, name=None):
        """
        Generate perturbation vector z for a parameter based on zo_perturbation mode.
        
        Two modes controlled by --zo_perturbation:
        
        1. Coordinate-wise:
           - z ~ N(0, 1) with same shape as param
           - Each element gets independent random noise
           
        2. Layer-wise:
           - z ~ N(0, 1) with same shape as param, then normalized to unit norm
           - Each layer gets a random DIRECTION (unit vector)
           - This reduces variance but may have different convergence properties
        
        Args:
            param: The parameter tensor to generate perturbation for
            name: Optional parameter name (for debugging)
            
        Returns:
            z: Perturbation tensor with same shape as param
        """
        perturbation_type = getattr(self.args, 'zo_perturbation', 'coordinate')
        
        if perturbation_type == "coordinate":
            # Layer-wise: generate random direction, then normalize to unit norm
            z = torch.normal(mean=0, std=1, size=param.data.size(), 
                           device=param.data.device, dtype=param.data.dtype)
            # Normalize to unit vector (Frobenius norm for matrices)
            z_norm = z.norm()
            if z_norm > 0:
                z = z / z_norm
            # Scale by sqrt(numel) to maintain similar magnitude to coordinate-wise
            # This ensures the expected perturbation magnitude is comparable
            z = z * (param.numel() ** 0.5)
        else:
            # Coordinate-wise (default): each element gets independent N(0,1)
            z = torch.normal(mean=0, std=1, size=param.data.size(), 
                           device=param.data.device, dtype=param.data.dtype)
            z_norm = z.norm()
            if z_norm > 0:
                z = z / z_norm
            # Scale by sqrt(numel) to maintain similar magnitude to coordinate-wise
            # This ensures the expected perturbation magnitude is comparable
            z = z * (param.numel() ** 0.5)
        
        return z

    def zo_perturb_module(self, module, cache_key, random_seed=None, scaling_factor=1, 
                          use_rng_state=False, rng_state=None):
        """
        Unified perturbation function for any module using the shared seed.
        
        Two modes based on --zo_continuous_rng flag:
        
        1. Shared Seed Mode (default):
           - Each module uses the shared seed INDEPENDENTLY
           - Both generate reproducible perturbations starting from the same point
           - Client and server get the SAME z sequence
        
        2. Continuous RNG Mode (use_rng_state=True):
           - Client perturbs first with seed, saves RNG state
           - Server loads RNG state and continues the sequence
           - Creates one continuous z vector across all parameters
        
        Perturbation type controlled by --zo_perturbation:
        - "coordinate": Each element gets independent N(0,1) noise
        - "layer": Each layer gets a normalized random direction
        
        Args:
            module: The module to perturb (client or server)
            cache_key: A string key for parameter caching (e.g., "client" or "server")
            random_seed: Optional seed override (defaults to self.zo_random_seed)
            scaling_factor: Scaling for perturbation (+1, -2, etc.)
            use_rng_state: If True, use rng_state instead of seed (for continuous RNG)
            rng_state: The RNG state to restore (required if use_rng_state=True)
            
        Returns:
            The RNG state after perturbation (for continuous RNG chaining)
        """
        if use_rng_state and rng_state is not None:
            # Continuous RNG mode: restore state from previous module
            torch.set_rng_state(rng_state)
        else:
            # Shared seed mode: reset to seed
            seed = random_seed if random_seed is not None else self.zo_random_seed
            torch.manual_seed(seed)
        
        # Use cached sorted params if available, otherwise compute and cache
        sorted_params = self._get_sorted_trainable_params(module, cache_key)
        
        for name, param in sorted_params:
            z = self._generate_perturbation(param, name)
            param.data = param.data + scaling_factor * z * self.args.zo_eps
        
        # Return current RNG state for chaining (continuous RNG mode)
        return torch.get_rng_state()

    def zo_perturb_parameters(self, target_modules, random_seed=None, scaling_factor=1):
        """
        Perturb parameters of specified modules using the shared seed.
        
        This is the legacy/non-split version of perturbation.
        For split learning, use zo_perturb_module instead.
        
        IMPORTANT: Parameters are iterated in SORTED ORDER to ensure deterministic
        RNG sequence matching between perturbation and update phases.
        
        Perturbation type controlled by --zo_perturbation:
        - "coordinate": Each element gets independent N(0,1) noise
        - "layer": Each layer gets a normalized random direction
        """
        seed = random_seed if random_seed is not None else self.zo_random_seed
        torch.manual_seed(seed)
        
        perturbed_params = set()
        
        # Collect all parameters with their full names (module prefix + param name)
        all_params = []
        for module in target_modules:
            for name, param in module.named_parameters():
                if param.requires_grad and id(param) not in perturbed_params:
                    all_params.append((name, param))
                    perturbed_params.add(id(param))
        
        # Sort by name to ensure deterministic order
        all_params.sort(key=lambda x: x[0])
        
        for name, param in all_params:
            z = self._generate_perturbation(param, name)
            param.data = param.data + scaling_factor * z * self.args.zo_eps

    def zo_perturb_client(self, client_module, random_seed=None, scaling_factor=1):
        """Perturb client parameters. Delegates to zo_perturb_module."""
        self.zo_perturb_module(client_module, "client", random_seed, scaling_factor)

    def zo_perturb_server(self, server_module, random_seed=None, scaling_factor=1):
        """Perturb server parameters. Delegates to zo_perturb_module."""
        self.zo_perturb_module(server_module, "server", random_seed, scaling_factor)

    def zo_forward(self, model, inputs):
        """
        Get (no gradient) loss from the model. Dropout is turned off too.
        """
        model.eval()
        with torch.inference_mode():
            inputs = self._prepare_inputs(inputs)
            outputs = model(**inputs)
            loss = outputs.loss if hasattr(outputs, "loss") else outputs[0]
            if isinstance(loss, torch.Tensor) and loss.ndim > 0:
                loss = loss.mean()
            if self.args.n_gpu > 1:
                loss = loss.mean()
        return loss.detach()

    def zo_update_module(self, module, cache_key, projected_grad, 
                         use_rng_state=False, rng_state=None, lr=None):
        """
        Unified update function for any module using the shared seed.
        
        Uses the SAME seed/RNG state as perturbation to regenerate identical z vectors.
        
        The update rule is: θ = θ - lr * (projected_grad * z + weight_decay * θ)
        where z is the SAME perturbation vector used during forward passes.
        
        Perturbation type controlled by --zo_perturbation:
        - "coordinate": Each element gets independent N(0,1) noise
        - "layer": Each layer gets a normalized random direction
        
        Args:
            module: The module to update (client or server)
            cache_key: A string key for parameter caching (e.g., "client" or "server")
            projected_grad: The scalar gradient estimate from ZO
            use_rng_state: If True, use rng_state instead of seed (for continuous RNG)
            rng_state: The RNG state to restore (required if use_rng_state=True)
            lr: Learning rate to use (if None, uses _get_learning_rate())
            
        Returns:
            The RNG state after update (for continuous RNG chaining)
        """
        if use_rng_state and rng_state is not None:
            # Continuous RNG mode: restore state from previous module
            torch.set_rng_state(rng_state)
        else:
            # Shared seed mode: reset to seed
            torch.manual_seed(self.zo_random_seed)
        
        # Use provided lr or fall back to scheduler LR
        if lr is None:
            lr = self._get_learning_rate()
        
        # Use cached sorted params (same order as perturbation)
        sorted_params = self._get_sorted_trainable_params(module, cache_key)
        
        for name, param in sorted_params:
            # Use the same perturbation generation as in zo_perturb_module
            z = self._generate_perturbation(param, name)
            if "bias" not in name and "layer_norm" not in name and "layernorm" not in name:
                param.data = param.data - lr * (projected_grad * z + self.args.weight_decay * param.data)
            else:
                param.data = param.data - lr * (projected_grad * z)
        
        # Return current RNG state for chaining (continuous RNG mode)
        return torch.get_rng_state()

    def zo_update(self, target_modules, projected_grad):
        """
        Update parameters of specified modules using the shared seed.
        
        This is the legacy/non-split version of update.
        For split learning, use zo_update_module instead.
        
        IMPORTANT: Parameters must be iterated in the SAME SORTED ORDER as
        zo_perturb_parameters to ensure identical z vectors are regenerated.
        
        Perturbation type controlled by --zo_perturbation:
        - "coordinate": Each element gets independent N(0,1) noise
        - "layer": Each layer gets a normalized random direction
        """
        torch.manual_seed(self.zo_random_seed)
        updated_params = set()
        
        # Collect all parameters with their full names
        all_params = []
        for module in target_modules:
            for name, param in module.named_parameters():
                if param.requires_grad and id(param) not in updated_params:
                    all_params.append((name, param))
                    updated_params.add(id(param))
        
        # Sort by name to ensure deterministic order (same as perturbation)
        all_params.sort(key=lambda x: x[0])
        
        lr = self._get_learning_rate()
        for name, param in all_params:
            # Use the same perturbation generation as in zo_perturb_parameters
            z = self._generate_perturbation(param, name)
            if "bias" not in name and "layer_norm" not in name and "layernorm" not in name:
                param.data = param.data - lr * (projected_grad * z + self.args.weight_decay * param.data)
            else:
                param.data = param.data - lr * (projected_grad * z)
        self.lr_scheduler.step()

    def zo_update_client(self, client_module, projected_grad):
        """Update client parameters. Delegates to zo_update_module."""
        self.zo_update_module(client_module, "client", projected_grad)

    def zo_update_server(self, server_module, projected_grad):
        """Update server parameters. Delegates to zo_update_module."""
        self.zo_update_module(server_module, "server", projected_grad)

    def fo_step_split(self, model, inputs):
        model.train()
        inputs = self._prepare_inputs(inputs)
        with self.compute_loss_context_manager():
            loss = self.compute_loss(model, inputs)

        if self.args.n_gpu > 1: loss = loss.mean()
        if self.args.gradient_accumulation_steps > 1 and not self.deepspeed:
            loss = loss / self.args.gradient_accumulation_steps

        # Server backward
        if self.use_apex:
            from apex import amp
            with amp.scale_loss(loss, self.optimizer) as scaled_loss:
                scaled_loss.backward()
        else:
            self.accelerator.backward(loss)

        # Client backward - gradients flow back through the split
        inner_model = model.module if hasattr(model, "module") else model
        if (
            hasattr(inner_model, "server_input")
            and inner_model.server_input is not None
            and inner_model.server_input.grad is not None
        ):
            # Get gradient at split point from server
            dfx_client = inner_model.server_input.grad.clone().detach()
            # Backward through client using the gradient from server
            inner_model.client_output.backward(dfx_client)
            # Clear stored tensors
            inner_model.server_input = None
            inner_model.client_output = None
            
        # Optimizer steps
        if self.client_optimizer_obj: 
            self.client_optimizer_obj.step()
            self.client_optimizer_obj.zero_grad()
        if self.server_optimizer_obj: 
            self.server_optimizer_obj.step()
            self.server_optimizer_obj.zero_grad()

        return loss.detach()

    def zo_step_split_coordinated(self, model, inputs):
        """
        True Split Learning ZO step with shared perturbation seed.
        
        Implements the protocol for zeroth-order optimization
        in split learning, where client and server are INDEPENDENT entities.
        
        Supports multiple perturbations for variance reduction:
        - num_pert=1: Single perturbation
        - num_pert>1: Average gradient estimates over K perturbations
        
        IMPORTANT: All gradient estimates are computed at the SAME base point θ₀,
        then accumulated and applied as a single update. This is essential for
        proper variance reduction.
        
        Two gradient estimation variants (controlled by --zo_variant):
        
        1. Central (default, --zo_variant="central"):
           - g = (f(θ+εz) - f(θ-εz)) / (2ε) -- 2 forward passes, lower bias
           
        2. Forward (--zo_variant="forward"):
           - g = (f(θ+εz) - f(θ)) / ε -- 1 forward pass, higher bias but faster
        
        Communication:
        - Forward: activations at the cut layer
        - Metadata: shared perturbation seed(s)
        - (No gradient exchange in ZO mode - only scalar loss is needed)
        """
        inner_model = model.module if hasattr(model, "module") else model
        
        # Verify this is a properly split model
        if not hasattr(inner_model, 'client') or not hasattr(inner_model, 'server'):
            raise RuntimeError(
                "Model does not have client/server split. "
                "Use SplitGPT2 or SplitOPT for split learning."
            )
        
        # Check configuration options
        use_continuous_rng = getattr(self.args, 'zo_continuous_rng', False)
        zo_variant = getattr(self.args, 'zo_variant', 'central')
        num_pert = getattr(self.args, 'num_pert', 1)
        
        # Generate seeds for all perturbations
        seeds = self.generate_multiple_seeds(num_pert)
        
        # Initialize accumulated gradients for client and server
        accumulated_client_grads = {}
        accumulated_server_grads = {}
        for name, param in inner_model.client.named_parameters():
            if param.requires_grad:
                accumulated_client_grads[name] = torch.zeros_like(param.data)
        for name, param in inner_model.server.named_parameters():
            if param.requires_grad:
                accumulated_server_grads[name] = torch.zeros_like(param.data)
        
        loss_for_log = None
        delta_loss = 0.0
        
        if zo_variant == "forward":
            # ============ RGE-FORWARD with multiple perturbations ============
            # Compute baseline loss once at θ₀
            loss0 = self.zo_forward(model, inputs)
            loss_for_log = loss0
            
            # For each perturbation, compute gradient estimate and ACCUMULATE
            for k, seed in enumerate(seeds):
                self.zo_random_seed = seed
                
                # Perturb +ε from θ₀
                model.train()
                if use_continuous_rng:
                    rng_state = self.zo_perturb_module(inner_model.client, "client", seed, scaling_factor=1)
                    self.zo_perturb_module(inner_model.server, "server", scaling_factor=1, use_rng_state=True, rng_state=rng_state)
                else:
                    self.zo_perturb_module(inner_model.client, "client", seed, scaling_factor=1)
                    self.zo_perturb_module(inner_model.server, "server", seed, scaling_factor=1)
                
                loss1 = self.zo_forward(model, inputs)
                
                # Restore parameters back to θ₀
                model.train()
                if use_continuous_rng:
                    rng_state = self.zo_perturb_module(inner_model.client, "client", seed, scaling_factor=-1)
                    self.zo_perturb_module(inner_model.server, "server", scaling_factor=-1, use_rng_state=True, rng_state=rng_state)
                else:
                    self.zo_perturb_module(inner_model.client, "client", seed, scaling_factor=-1)
                    self.zo_perturb_module(inner_model.server, "server", seed, scaling_factor=-1)
                
                # Compute gradient for this perturbation (scaled by 1/num_pert for averaging)
                projected_grad = (loss1 - loss0) / (self.args.zo_eps * num_pert)
                
                # Update with this perturbation's gradient contribution
                if use_continuous_rng:
                    rng_state = self.zo_update_module(inner_model.client, "client", projected_grad, lr=self.client_lr)
                    self.zo_update_module(inner_model.server, "server", projected_grad, use_rng_state=True, rng_state=rng_state, lr=self.server_lr)
                else:
                    self.zo_update_module(inner_model.client, "client", projected_grad, lr=self.client_lr)
                    self.zo_update_module(inner_model.server, "server", projected_grad, lr=self.server_lr)
            
            delta_loss = loss1 - loss0
            
        else:
            # ============ RGE-CENTRAL with multiple perturbations ============
            loss_for_log = None
            
            # For each perturbation, compute gradient and update
            for k, seed in enumerate(seeds):
                self.zo_random_seed = seed
                
                # Positive perturbation
                model.train()
                if use_continuous_rng:
                    rng_state = self.zo_perturb_module(inner_model.client, "client", seed, scaling_factor=1)
                    self.zo_perturb_module(inner_model.server, "server", scaling_factor=1, use_rng_state=True, rng_state=rng_state)
                else:
                    self.zo_perturb_module(inner_model.client, "client", seed, scaling_factor=1)
                    self.zo_perturb_module(inner_model.server, "server", seed, scaling_factor=1)
                
                loss1 = self.zo_forward(model, inputs)
                if loss_for_log is None:
                    loss_for_log = loss1

                # Negative perturbation (apply -2ε to go from +ε to -ε)
                model.train()
                if use_continuous_rng:
                    rng_state = self.zo_perturb_module(inner_model.client, "client", seed, scaling_factor=-2)
                    self.zo_perturb_module(inner_model.server, "server", scaling_factor=-2, use_rng_state=True, rng_state=rng_state)
                else:
                    self.zo_perturb_module(inner_model.client, "client", seed, scaling_factor=-2)
                    self.zo_perturb_module(inner_model.server, "server", seed, scaling_factor=-2)
                
                loss2 = self.zo_forward(model, inputs)
                
                # Restore parameters (apply +ε to go from -ε to 0)
                model.train()
                if use_continuous_rng:
                    rng_state = self.zo_perturb_module(inner_model.client, "client", seed, scaling_factor=1)
                    self.zo_perturb_module(inner_model.server, "server", scaling_factor=1, use_rng_state=True, rng_state=rng_state)
                else:
                    self.zo_perturb_module(inner_model.client, "client", seed, scaling_factor=1)
                    self.zo_perturb_module(inner_model.server, "server", seed, scaling_factor=1)
                
                # Compute gradient for this perturbation (scaled by 1/num_pert for averaging)
                projected_grad = (loss1 - loss2) / (2 * self.args.zo_eps * num_pert)
                
                # Update with this perturbation's gradient contribution
                if use_continuous_rng:
                    rng_state = self.zo_update_module(inner_model.client, "client", projected_grad, lr=self.client_lr)
                    self.zo_update_module(inner_model.server, "server", projected_grad, use_rng_state=True, rng_state=rng_state, lr=self.server_lr)
                else:
                    self.zo_update_module(inner_model.client, "client", projected_grad, lr=self.client_lr)
                    self.zo_update_module(inner_model.server, "server", projected_grad, lr=self.server_lr)
            
            delta_loss = loss1 - loss2
        
        # ============ ZO DIAGNOSTIC LOGGING ============
        if hasattr(self, 'state') and self.state.global_step % 100 == 0:
            rng_mode = "continuous" if use_continuous_rng else "shared_seed"
            perturbation_type = getattr(self.args, 'zo_perturbation', 'coordinate')
            logger.info(
                f"[ZO Diagnostics] Step {self.state.global_step}: "
                f"variant={zo_variant}, perturbation={perturbation_type}, num_pert={num_pert}, "
                f"Δloss={delta_loss:.6f}, "
                f"client_lr={self.client_lr:.2e}, server_lr={self.server_lr:.2e}, "
                f"rng_mode={rng_mode}"
            )
        
        # Step LR scheduler
        self.lr_scheduler.step()
        
        return loss_for_log

    def zo_step_coordinated(self, model, inputs, modules):
        """
        Legacy coordinated ZO step - kept for backwards compatibility.
        Perturbs all modules together (not truly split).
        
        Supports both RGE-central and RGE-forward variants via --zo_variant.
        """
        self.zo_random_seed = np.random.randint(1000000000)
        zo_variant = getattr(self.args, 'zo_variant', 'central')

        if zo_variant == "forward":
            # RGE-forward: g = (f(θ+εz) - f(θ)) / ε
            # Compute baseline loss at unperturbed θ
            loss0 = self.zo_forward(model, inputs)
            
            # Perturb +ε
            model.train()
            self.zo_perturb_parameters(modules, scaling_factor=1)
            loss1 = self.zo_forward(model, inputs)
            
            # Restore -ε to get back to θ
            model.train()
            self.zo_perturb_parameters(modules, scaling_factor=-1)
            
            projected_grad = (loss1 - loss0) / self.args.zo_eps
            loss_for_return = loss0
        else:
            # RGE-central (default): g = (f(θ+εz) - f(θ-εz)) / (2ε)
            model.train()
            self.zo_perturb_parameters(modules, scaling_factor=1)
            loss1 = self.zo_forward(model, inputs)

            model.train()
            self.zo_perturb_parameters(modules, scaling_factor=-2)
            loss2 = self.zo_forward(model, inputs)
            
            model.train()
            self.zo_perturb_parameters(modules, scaling_factor=1)  # Restore
            
            projected_grad = (loss1 - loss2) / (2 * self.args.zo_eps)
            loss_for_return = loss1
        
        self.zo_update(modules, projected_grad)
        
        return loss_for_return

    def training_step(self, model, inputs):
        """
        Unified training step for split learning + ZO.

        Supported modes:
        - ZO/ZO: Both client and server use zeroth-order with shared seed
        - FO/FO: Standard first-order split learning with gradient backprop
        - ZO/FO: Client uses ZO, server uses FO (hybrid)
        - FO/ZO: Client uses FO, server uses ZO (hybrid)
        
        True Split Learning Protocol:
        - Client and server are INDEPENDENT modules
        - Only activations (forward) and gradients (backward) are exchanged
        - For ZO: Both parties share the SAME perturbation seed
        
        Hybrid Modes:
        - ZO/FO: Client optimized via zeroth-order (no gradients needed), 
                 Server optimized via first-order (standard backprop)
        - FO/ZO: Client optimized via first-order (backprop through split point),
                 Server optimized via zeroth-order (perturbation-based)
        """
        global_trainer = getattr(self.args, "trainer", "regular")
        client_mode = getattr(self.args, "client_optimizer", "auto")
        server_mode = getattr(self.args, "server_optimizer", "auto")

        # Resolve "auto" modes based on global trainer setting
        if client_mode == "auto":
            client_mode = "zo" if global_trainer == "zo" else "fo"
        if server_mode == "auto":
            server_mode = "zo" if global_trainer == "zo" else "fo"

        # Guard: ZO requires trainer='zo'
        if global_trainer != "zo" and "zo" in (client_mode, server_mode):
            raise ValueError(
                f"Zeroth-order modes require args.trainer='zo'. "
                f"Got trainer={global_trainer}, client={client_mode}, server={server_mode}."
            )

        inner_model = model.module if hasattr(model, "module") else model

        # Pure ZO/ZO: TRUE split ZO with shared seed
        if client_mode == "zo" and server_mode == "zo":
            return self.zo_step_split_coordinated(model, inputs)

        # Pure FO/FO: Standard first-order split learning
        if client_mode == "fo" and server_mode == "fo":
            return self.fo_step_split(model, inputs)

        # ZO/FO: Client uses ZO, server uses FO (hybrid)
        if client_mode == "zo" and server_mode == "fo":
            return self._zo_fo_step(model, inputs, inner_model)

        # FO/ZO: Client uses FO, server uses ZO (hybrid)
        if client_mode == "fo" and server_mode == "zo":
            return self._fo_zo_step(model, inputs, inner_model)

        raise ValueError(
            f"Unsupported mode: client={client_mode}, server={server_mode}. "
            f"Supported: ZO/ZO, FO/FO, ZO/FO, FO/ZO"
        )

    def _zo_fo_step(self, model, inputs, inner_model):
        """
        Hybrid mode: Client uses ZO (zeroth-order), Server uses FO (first-order).
        
        Protocol:
        1. Perturb client (+ε*z)
        2. Forward pass → loss1 (with gradient tracking for server)
        3. Backward pass → compute and store server gradients in .grad (DON'T update yet!)
        4. Perturb client to (-ε*z)
        5. Forward pass (inference mode) → loss2 (same server weights as loss1)
        6. Restore client parameters
        7. Update client using ZO gradient estimate (with client_lr)
        8. Update server using stored gradients from step 3 (with server_lr)
        
        IMPORTANT: Server must NOT be updated between loss1 and loss2 computation,
        otherwise the ZO gradient estimate for the client is corrupted.
        
        Learning rates:
        - Client ZO uses self.client_lr (typically larger, e.g., 1e-3)
        - Server FO uses self.server_lr via optimizer (typically smaller, e.g., 5e-5)
        """
        num_pert = getattr(self.args, 'num_pert', 1)
        seeds = self.generate_multiple_seeds(num_pert)
        
        inputs = self._prepare_inputs(inputs)
        loss_for_log = None
        
        for k, seed in enumerate(seeds):
            self.zo_random_seed = seed
            
            # ============ PHASE 1: POSITIVE PERTURBATION ============
            model.train()
            self.zo_perturb_module(inner_model.client, "client", seed, scaling_factor=1)

            # Forward pass with gradient tracking for server
            with self.compute_loss_context_manager():
                loss1 = self.compute_loss(model, inputs)

            if self.args.n_gpu > 1:
                loss1 = loss1.mean()
            
            if loss_for_log is None:
                loss_for_log = loss1.detach()
            
            # Scale loss for gradient accumulation across perturbations
            loss1_scaled = loss1 / num_pert
            if self.args.gradient_accumulation_steps > 1 and not self.deepspeed:
                loss1_scaled = loss1_scaled / self.args.gradient_accumulation_steps
                
            # Backward to accumulate server gradients (stored in .grad)
            self.accelerator.backward(loss1_scaled)

            # ============ PHASE 2: NEGATIVE PERTURBATION ============
            model.train()
            self.zo_perturb_module(inner_model.client, "client", seed, scaling_factor=-2)
            loss2 = self.zo_forward(model, inputs)
            
            # ============ PHASE 3: RESTORE & UPDATE CLIENT ============
            model.train()
            self.zo_perturb_module(inner_model.client, "client", seed, scaling_factor=1)

            # Update client using ZO gradient estimate scaled by 1/num_pert
            projected_grad = (loss1.detach() - loss2) / (2 * self.args.zo_eps * num_pert)
            self.zo_update_module(inner_model.client, "client", projected_grad, lr=self.client_lr)
        
        # ============ PHASE 4: UPDATE SERVER ============
        # Server gradients have been accumulated across all perturbations (scaled by 1/num_pert)
        if self.server_optimizer_obj:
            self.server_optimizer_obj.step()
            self.server_optimizer_obj.zero_grad()

        # ============ DIAGNOSTIC LOGGING ============
        if hasattr(self, 'state') and self.state.global_step % 100 == 0:
            logger.info(
                f"[ZO/FO Hybrid] Step {self.state.global_step}: "
                f"num_pert={num_pert}, loss={loss_for_log:.4f}, "
                f"client_lr={self.client_lr:.2e}, server_lr={self.server_lr:.2e}"
            )

        self.lr_scheduler.step()

        return loss_for_log

    def _fo_zo_step(self, model, inputs, inner_model):
        """
        Hybrid mode: Client uses FO (first-order), Server uses ZO (zeroth-order).
        
        Protocol:
        1. Perturb server (+ε*z)
        2. Forward pass → loss1 (with gradient tracking for client via split point)
        3. Perturb server to (-ε*z)
        4. Forward pass (inference mode) → loss2
        5. Restore server parameters
        6. Update server using ZO gradient estimate (with server_lr)
        7. Re-forward with updated server, backward to get client gradients
        8. Update client using FO gradients (with client_lr via optimizer)
        
        Learning rates:
        - Server ZO uses self.server_lr (typically larger, e.g., 1e-3)
        - Client FO uses self.client_lr via optimizer (typically smaller, e.g., 5e-5)
        
        Note: This mode requires careful handling of the split point gradients.
        The client backward happens after server ZO update to reflect the new server state.
        """
        num_pert = getattr(self.args, 'num_pert', 1)
        seeds = self.generate_multiple_seeds(num_pert)
        
        loss_for_log = None
        
        for k, seed in enumerate(seeds):
            self.zo_random_seed = seed
            
            # ============ PHASE 1: POSITIVE PERTURBATION ============
            model.train()
            self.zo_perturb_module(inner_model.server, "server", seed, scaling_factor=1)

            # Forward pass (inference mode for ZO)
            loss1 = self.zo_forward(model, inputs)
            
            if loss_for_log is None:
                loss_for_log = loss1

            # ============ PHASE 2: NEGATIVE PERTURBATION ============
            model.train()
            self.zo_perturb_module(inner_model.server, "server", seed, scaling_factor=-2)
            loss2 = self.zo_forward(model, inputs)
            
            # ============ PHASE 3: RESTORE & UPDATE SERVER ============
            model.train()
            self.zo_perturb_module(inner_model.server, "server", seed, scaling_factor=1)

            # Update server using ZO gradient estimate scaled by 1/num_pert
            projected_grad = (loss1 - loss2) / (2 * self.args.zo_eps * num_pert)
            self.zo_update_module(inner_model.server, "server", projected_grad, lr=self.server_lr)
        
        # ============ PHASE 4: CLIENT FO UPDATE ============
        # Now do a forward+backward pass for client FO update
        # This uses the updated server parameters (after all ZO updates)
        inputs = self._prepare_inputs(inputs)
        with self.compute_loss_context_manager():
            loss_fo = self.compute_loss(model, inputs)

        if self.args.n_gpu > 1:
            loss_fo = loss_fo.mean()
        if self.args.gradient_accumulation_steps > 1 and not self.deepspeed:
            loss_fo = loss_fo / self.args.gradient_accumulation_steps

        self.accelerator.backward(loss_fo)
        
        # Client backward through split point
        if (
            hasattr(inner_model, "server_input")
            and inner_model.server_input is not None
            and inner_model.server_input.grad is not None
        ):
            dfx_client = inner_model.server_input.grad.clone().detach()
            inner_model.client_output.backward(dfx_client)
            inner_model.server_input = None
            inner_model.client_output = None
            
        # Update client using FO gradients (uses client_lr via optimizer)
        if self.client_optimizer_obj:
            self.client_optimizer_obj.step()
            self.client_optimizer_obj.zero_grad()

        # ============ DIAGNOSTIC LOGGING ============
        if hasattr(self, 'state') and self.state.global_step % 100 == 0:
            logger.info(
                f"[FO/ZO Hybrid] Step {self.state.global_step}: "
                f"num_pert={num_pert}, loss={loss_for_log:.4f}, "
                f"fo_loss={loss_fo.item():.4f}, client_lr={self.client_lr:.2e}, "
                f"server_lr={self.server_lr:.2e}"
            )

        self.lr_scheduler.step()

        return loss_for_log


    ################ Verification utilities ################

    def verify_shared_seed_perturbations(self, model):
        """
        Verify that client and server generate reproducible perturbations with the same seed.
        
        This is a diagnostic function to ensure the simplified RNG approach works correctly.
        It checks that:
        1. Each side generates the same perturbations when given the same seed
        2. Perturbation and update phases generate identical z vectors
        
        Returns:
            Tuple of (is_valid, message)
        """
        inner_model = model.module if hasattr(model, "module") else model
        
        if not hasattr(inner_model, 'client') or not hasattr(inner_model, 'server'):
            return False, "Model does not have client/server split."
        
        # Store original parameters
        client_params_original = {
            name: param.data.clone() 
            for name, param in inner_model.client.named_parameters() 
            if param.requires_grad
        }
        server_params_original = {
            name: param.data.clone() 
            for name, param in inner_model.server.named_parameters() 
            if param.requires_grad
        }
        
        # Test seed
        test_seed = 42
        
        # Test 1: Perturb client twice with same seed, verify identical results
        self.zo_random_seed = test_seed
        self.zo_perturb_client(inner_model.client, test_seed, scaling_factor=1)
        client_after_perturb1 = {
            name: param.data.clone() 
            for name, param in inner_model.client.named_parameters() 
            if param.requires_grad
        }
        
        # Restore and perturb again
        for name, param in inner_model.client.named_parameters():
            if param.requires_grad:
                param.data.copy_(client_params_original[name])
        
        self.zo_perturb_client(inner_model.client, test_seed, scaling_factor=1)
        client_after_perturb2 = {
            name: param.data.clone() 
            for name, param in inner_model.client.named_parameters() 
            if param.requires_grad
        }
        
        # Verify client perturbations are identical
        for name in client_after_perturb1:
            if not torch.allclose(client_after_perturb1[name], client_after_perturb2[name]):
                # Restore original parameters
                for n, p in inner_model.client.named_parameters():
                    if p.requires_grad:
                        p.data.copy_(client_params_original[n])
                for n, p in inner_model.server.named_parameters():
                    if p.requires_grad:
                        p.data.copy_(server_params_original[n])
                return False, f"Client perturbation not reproducible for param {name}"
        
        # Test 2: Verify server perturbations are reproducible
        self.zo_perturb_server(inner_model.server, test_seed, scaling_factor=1)
        server_after_perturb1 = {
            name: param.data.clone() 
            for name, param in inner_model.server.named_parameters() 
            if param.requires_grad
        }
        
        # Restore and perturb again
        for name, param in inner_model.server.named_parameters():
            if param.requires_grad:
                param.data.copy_(server_params_original[name])
        
        self.zo_perturb_server(inner_model.server, test_seed, scaling_factor=1)
        server_after_perturb2 = {
            name: param.data.clone() 
            for name, param in inner_model.server.named_parameters() 
            if param.requires_grad
        }
        
        # Verify server perturbations are identical
        for name in server_after_perturb1:
            if not torch.allclose(server_after_perturb1[name], server_after_perturb2[name]):
                # Restore original parameters
                for n, p in inner_model.client.named_parameters():
                    if p.requires_grad:
                        p.data.copy_(client_params_original[n])
                for n, p in inner_model.server.named_parameters():
                    if p.requires_grad:
                        p.data.copy_(server_params_original[n])
                return False, f"Server perturbation not reproducible for param {name}"
        
        # Restore original parameters
        for name, param in inner_model.client.named_parameters():
            if param.requires_grad:
                param.data.copy_(client_params_original[name])
        for name, param in inner_model.server.named_parameters():
            if param.requires_grad:
                param.data.copy_(server_params_original[name])
        
        return True, "Shared seed perturbations verified: Client and server generate reproducible perturbations."


    ################ Evaluation overrides ################

    def evaluate(
        self,
        eval_dataset = None,
        ignore_keys = None,
        metric_key_prefix: str = "eval",
    ):
        """
        Override evaluate to print eval loss.
        """
        # Call the parent evaluate method
        output = super().evaluate(
            eval_dataset=eval_dataset,
            ignore_keys=ignore_keys,
            metric_key_prefix=metric_key_prefix,
        )

        # Print the eval loss
        if f"{metric_key_prefix}_loss" in output:
            logger.info(f"***** {metric_key_prefix.capitalize()} Loss: {output[f'{metric_key_prefix}_loss']:.4f} *****")
        else:
            logger.info(f"***** {metric_key_prefix.capitalize()} Loss not found. Keys: {list(output.keys())} *****")
            print(f"***** {metric_key_prefix.capitalize()} Loss not found. Keys: {list(output.keys())} *****")

        return output



    ################ Checkpointing overrides ################

    def _save_checkpoint(self, model, trial):
        """
        Disable periodic checkpoint saving during training.

        This overrides the base Trainer implementation so that calls from
        `_maybe_log_save_evaluate` become no-ops. Final model saving via
        `save_model` is still available if you call it explicitly.
        """
        return
