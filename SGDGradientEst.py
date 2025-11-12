import torch
import numpy as np
import math

class StochasticGradientApproximator:
    def __init__(self, model_params, perturbation_scale=1e-3, sample_count=1,
                 compute_device=None, data_type=torch.float32, estimator_type: str = 'central',
                 perturbation_distribution: str = 'rademacher', bernoulli_p: float = 0.5, center_bernoulli: bool = True):
        self.trainable_params = [param for param in model_params if param.requires_grad]
        self.param_count = sum([param.numel() for param in self.trainable_params])
        self.perturbation_scale = perturbation_scale
        self.sample_count = sample_count
        self.compute_device = compute_device if compute_device is not None else torch.device('cpu')
        self.data_type = data_type
        self.estimator_type = str(estimator_type).lower()
        self.perturbation_distribution = str(perturbation_distribution).lower()
        self.bernoulli_p = float(bernoulli_p)
        self.center_bernoulli = bool(center_bernoulli)
        # Runtime context exposed during objective evaluations (populated in estimate_gradients)
        self._current_context = None
        
        # Store original parameters for restoration
        self.original_params = []
        for param in self.trainable_params:
            self.original_params.append(param.data.clone())

    # Allow external callers to update the parameter set dynamically
    @property
    def model_params(self):
        return self.trainable_params

    @model_params.setter
    def model_params(self, params):
        # Reset trainable parameter references and derived metadata
        self.trainable_params = [p for p in params if p.requires_grad]
        self.param_count = sum(p.numel() for p in self.trainable_params)
        self.original_params = []
        for param in self.trainable_params:
            self.original_params.append(param.data.clone())

    @property
    def current_context(self):
        """Context describing the in-flight finite-difference probe (seed, sign, etc.)."""
        return self._current_context

    def _sample_direction_for_param(self, param, seeded_value: int):
        """
        Generate a perturbation direction tensor of the same shape as `param` using
        the configured distribution and a deterministic per-parameter seed.
        Supported distributions:
          - 'rademacher': values in {-1, +1}
          - 'gaussian'  : values ~ N(0, 1)
          - 'bernoulli' : if center_bernoulli=True, mean-centered and variance-normalized;
                          otherwise raw {0, 1} with probability p
        """
        gen = torch.Generator(device=param.device)
        gen.manual_seed(int(seeded_value))
        dist = self.perturbation_distribution
        if dist == 'rademacher':
            direction = torch.randint(0, 2, param.shape, dtype=torch.float32, device=param.device, generator=gen) * 2.0 - 1.0
            return direction
        if dist == 'gaussian':
            return torch.randn(param.shape, dtype=torch.float32, device=param.device, generator=gen)
        if dist == 'bernoulli':
            p = max(1e-7, min(1.0 - 1e-7, float(self.bernoulli_p)))
            if self.center_bernoulli:
                # Mean-center and scale to unit variance: (B - p)/sqrt(p(1-p))
                mask = torch.rand(param.shape, dtype=torch.float32, device=param.device, generator=gen) < p
                direction = mask.to(torch.float32)
                direction = (direction - p) / math.sqrt(p * (1.0 - p))
                return direction
            # Raw {0,1} Bernoulli(p)
            return (torch.rand(param.shape, dtype=torch.float32, device=param.device, generator=gen) < p).to(torch.float32)
        raise ValueError(f"Unsupported perturbation_distribution: {dist}")

    def estimate_gradients(self, input_batch, target_labels, objective_fn, random_seed):
        """
        Fixed gradient estimation based on MeZO approach
        """
        if len(self.trainable_params) == 0:
            return
        
        # Clear existing gradients
        for param in self.trainable_params:
            if param.grad is not None:
                param.grad.zero_()
        
        # Initialize gradients to zero
        for param in self.trainable_params:
            param.grad = torch.zeros_like(param.data)
        
        total_loss_diff = 0.0
        loss_diffs = []  # monitor |f+ - f-|
        
        for sample_idx in range(self.sample_count):
            # Use deterministic per-parameter seeds to regenerate directions on-the-fly
            base_seed = int(random_seed + sample_idx)
            # Record probe context for external coordination (e.g., coordinated ZOO)
            self._current_context = {
                "sample_index": sample_idx,
                "base_seed": base_seed,
                "mu": float(self.perturbation_scale),
                "estimator": self.estimator_type,
                "num_samples": int(self.sample_count),
            }

            # Forward perturbation: θ <- θ + μ * z
            for idx, param in enumerate(self.trainable_params):
                direction = self._sample_direction_for_param(param, int(base_seed * 1000003 + idx))
                param.data.add_(direction, alpha=self.perturbation_scale)
            # Context for +μ evaluation
            if self._current_context is not None:
                self._current_context.update({"sign": +1, "stage": "plus"})
            try:
                loss_plus = objective_fn(input_batch, target_labels)
            except Exception as e:
                print(f"Error in forward perturbation: {e}")
                loss_plus = torch.tensor(0.0, device=self.compute_device)
            if self._current_context is not None:
                self._current_context["last_loss"] = float(loss_plus.item() if isinstance(loss_plus, torch.Tensor) else loss_plus)

            if self.estimator_type == 'central':
                # Move from θ+μz to θ-μz by applying −2μz, then evaluate
                for idx, param in enumerate(self.trainable_params):
                    direction = self._sample_direction_for_param(param, int(base_seed * 1000003 + idx))
                    param.data.add_(direction, alpha=-2.0 * self.perturbation_scale)
                if self._current_context is not None:
                    self._current_context.update({"sign": -1, "stage": "minus"})
                try:
                    loss_minus = objective_fn(input_batch, target_labels)
                except Exception as e:
                    print(f"Error in backward perturbation: {e}")
                    loss_minus = torch.tensor(0.0, device=self.compute_device)
                if self._current_context is not None:
                    self._current_context["last_loss"] = float(loss_minus.item() if isinstance(loss_minus, torch.Tensor) else loss_minus)

                # Restore back to θ by applying +μz
                for idx, param in enumerate(self.trainable_params):
                    direction = self._sample_direction_for_param(param, int(base_seed * 1000003 + idx))
                    param.data.add_(direction, alpha=self.perturbation_scale)
                denom = (2.0 * self.perturbation_scale)
            else:
                # One-sided estimator: restore to θ and evaluate f(θ)
                for idx, param in enumerate(self.trainable_params):
                    direction = self._sample_direction_for_param(param, int(base_seed * 1000003 + idx))
                    param.data.add_(direction, alpha=-1.0 * self.perturbation_scale)
                if self._current_context is not None:
                    self._current_context.update({"sign": 0, "stage": "base"})
                try:
                    loss_minus = objective_fn(input_batch, target_labels)
                except Exception as e:
                    print(f"Error in base evaluation: {e}")
                    loss_minus = torch.tensor(0.0, device=self.compute_device)
                if self._current_context is not None:
                    self._current_context["last_loss"] = float(loss_minus.item() if isinstance(loss_minus, torch.Tensor) else loss_minus)
                denom = self.perturbation_scale

            # Calculate finite difference
            if isinstance(loss_plus, torch.Tensor):
                loss_plus = loss_plus.item()
            if isinstance(loss_minus, torch.Tensor):
                loss_minus = loss_minus.item()

            fd_num = (loss_plus - loss_minus)
            finite_diff = fd_num / denom
            total_loss_diff += finite_diff
            loss_diffs.append(abs(fd_num))

            # Accumulate gradients: g += FD * z (re-generate z with the same seeds)
            for idx, param in enumerate(self.trainable_params):
                direction = self._sample_direction_for_param(param, int(base_seed * 1000003 + idx))
                param.grad.add_(direction, alpha=finite_diff)
            # Clear context after processing this probe
            self._current_context = None
        
        # Average gradients over samples
        for param in self.trainable_params:
            param.grad.div_(self.sample_count)
            
        print(f"ZOO: Avg loss diff: {total_loss_diff/self.sample_count:.6f}")
        try:
            import math
            if loss_diffs:
                mu_abs = sum(loss_diffs)/len(loss_diffs)
                var = sum((x-mu_abs)**2 for x in loss_diffs)/max(1,len(loss_diffs)-1)
                std = math.sqrt(max(0.0, var))
                print(f"ZOO: |f+-f-| mean: {mu_abs:.6e}, std: {std:.6e}")
        except Exception:
            pass
        
        # Check gradient norms for debugging
        total_grad_norm = 0.0
        for param in self.trainable_params:
            if param.grad is not None:
                total_grad_norm += param.grad.norm().item() ** 2
        total_grad_norm = total_grad_norm ** 0.5
        print(f"ZOO: Gradient norm: {total_grad_norm:.6f}")
        