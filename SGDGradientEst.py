import torch
import numpy as np

class StochasticGradientApproximator:
    def __init__(self, model_params, perturbation_scale=1e-3, sample_count=1,
                 compute_device=None, data_type=torch.float32, estimator_type: str = 'central',
                 perturbation_type: str = 'gaussian', max_grad_value: float = 1.0, debug: bool = False):
        self.trainable_params = [param for param in model_params if param.requires_grad]
        self.param_count = sum([param.numel() for param in self.trainable_params])
        self.perturbation_scale = perturbation_scale
        self.sample_count = sample_count
        self.compute_device = compute_device if compute_device is not None else torch.device('cpu')
        self.data_type = data_type
        self.estimator_type = str(estimator_type).lower()
        self.perturbation_type = str(perturbation_type).lower()
        self.max_grad_value = float(max_grad_value)  # Maximum gradient value for clipping
        self.debug = bool(debug)
        
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

    def _generate_perturbation(self, param_shape, device, dtype, generator):
        """
        Generate perturbation vector based on the specified distribution type.

        Args:
            param_shape: Shape of the parameter tensor
            device: Device to place the tensor on
            dtype: torch dtype to generate with (should match parameter dtype)
            generator: PyTorch random number generator

        Returns:
            Perturbation tensor z with the specified distribution
        """
        if self.perturbation_type == 'rademacher':
            # Rademacher distribution: ±1 with equal probability
            return torch.randint(0, 2, param_shape, dtype=dtype, device=device, generator=generator) * 2.0 - 1.0
        elif self.perturbation_type == 'bernoulli':
            # Bernoulli distribution: 0 or 1, then scaled to ±1
            return (torch.bernoulli(torch.full(param_shape, 0.5, device=device, dtype=dtype), generator=generator) * 2.0 - 1.0)
        elif self.perturbation_type == 'gaussian':
            # Gaussian distribution: N(0,1)
            return torch.randn(param_shape, device=device, dtype=dtype, generator=generator)
        else:
            raise ValueError(f"Unknown perturbation type: {self.perturbation_type}. "
                           "Supported types: 'rademacher', 'bernoulli', 'gaussian'")

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
        avg_loss_diff = 0.0  # Initialize for use outside loop
        
        for sample_idx in range(self.sample_count):
            # Use deterministic per-parameter seeds to regenerate directions on-the-fly
            base_seed = int(random_seed + sample_idx)

            # Forward perturbation: θ <- θ + μ * z
            for idx, param in enumerate(self.trainable_params):
                gen = torch.Generator(device=param.device)
                gen.manual_seed(int(base_seed * 1000003 + idx))
                direction = self._generate_perturbation(param.shape, param.device, param.dtype, gen)

                # Optional debug: save and show first perturbation value for verification
                if self.debug and sample_idx == 0 and idx == 0:
                    self._debug_first_z = direction.clone()
                    try:
                        print(f"DEBUG: Forward z[0,0] = {direction.flatten()[0]:.6f}")
                    except Exception:
                        pass
                param.data.add_(direction, alpha=self.perturbation_scale)
            try:
                loss_plus = objective_fn(input_batch, target_labels)
            except Exception as e:
                print(f"Error in forward perturbation: {e}")
                loss_plus = torch.tensor(0.0, device=self.compute_device)

            if self.estimator_type == 'central':
                # Move from θ+μz to θ-μz by applying −2μz, then evaluate
                for idx, param in enumerate(self.trainable_params):
                    gen = torch.Generator(device=param.device)
                    gen.manual_seed(int(base_seed * 1000003 + idx))
                    direction = self._generate_perturbation(param.shape, param.device, param.dtype, gen)
                    param.data.add_(direction, alpha=-2.0 * self.perturbation_scale)
                try:
                    loss_minus = objective_fn(input_batch, target_labels)
                except Exception as e:
                    print(f"Error in backward perturbation: {e}")
                    loss_minus = torch.tensor(0.0, device=self.compute_device)

                # Restore back to θ by applying +μz
                for idx, param in enumerate(self.trainable_params):
                    gen = torch.Generator(device=param.device)
                    gen.manual_seed(int(base_seed * 1000003 + idx))
                    direction = self._generate_perturbation(param.shape, param.device, param.dtype, gen)
                    param.data.add_(direction, alpha=self.perturbation_scale)
                denom = (2.0 * self.perturbation_scale)
            else:
                # One-sided estimator: restore to θ and evaluate f(θ)
                for idx, param in enumerate(self.trainable_params):
                    gen = torch.Generator(device=param.device)
                    gen.manual_seed(int(base_seed * 1000003 + idx))
                    direction = self._generate_perturbation(param.shape, param.device, param.dtype, gen)
                    param.data.add_(direction, alpha=-1.0 * self.perturbation_scale)
                try:
                    loss_minus = objective_fn(input_batch, target_labels)
                except Exception as e:
                    print(f"Error in base evaluation: {e}")
                    loss_minus = torch.tensor(0.0, device=self.compute_device)
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
            # MeZO gradient estimate: g ≈ (1/d) * Σ[(f(θ+μz) - f(θ-μz)) / (2μ)] * z
            # The finite_diff already includes division by (2μ), so we just multiply by z
            for idx, param in enumerate(self.trainable_params):
                gen = torch.Generator(device=param.device)
                gen.manual_seed(int(base_seed * 1000003 + idx))
                direction = self._generate_perturbation(param.shape, param.device, param.dtype, gen)

                # Optional debug: verify the regenerated z matches the forward phase
                if self.debug and sample_idx == 0 and idx == 0 and hasattr(self, '_debug_first_z'):
                    try:
                        z_match = torch.allclose(direction, self._debug_first_z, atol=1e-6)
                        print(f"DEBUG: Gradient z[0,0] = {direction.flatten()[0]:.6f}")
                        print(f"DEBUG: z matches? {z_match}")
                        if not z_match:
                            diff = (direction - self._debug_first_z).abs().max()
                            print(f"  ERROR: z doesn't match! Diff: {float(diff):.6e}")
                    except Exception:
                        pass
                param.grad.add_(direction, alpha=finite_diff)
        
        # Average gradients over samples AFTER the loop completes
        # MeZO divides by sample_count to average gradients
        # No parameter-count-dependent scaling - MeZO doesn't use this
        if self.sample_count > 0:
            for param in self.trainable_params:
                if param.grad is not None:
                    param.grad.div_(self.sample_count)
        else:
            print(f"WARNING: No samples processed in gradient estimation - sample_count is 0!")
        
        
        # Check gradient norms for debugging (pre-clip)
        total_grad_norm = 0.0
        has_grad = False
        for param in self.trainable_params:
            if param.grad is not None:
                param_norm = param.grad.norm().item()
                if not torch.isfinite(torch.tensor(param_norm)):
                    print(f"WARNING: Non-finite gradient detected for parameter with shape {param.shape}")
                    param.grad.zero_()
                else:
                    total_grad_norm += param_norm ** 2
                    has_grad = True
        total_grad_norm = total_grad_norm ** 0.5 if has_grad else 0.0
        print(f"ZOO: Gradient norm: {total_grad_norm:.6f}")

        # Apply global-norm clipping if configured
        try:
            if float(self.max_grad_value) > 0.0 and has_grad:
                torch.nn.utils.clip_grad_norm_(self.trainable_params, max_norm=float(self.max_grad_value))
                # Recompute norm after clipping for visibility
                clipped_norm_sq = 0.0
                for param in self.trainable_params:
                    if param.grad is not None:
                        clipped_norm_sq += float(param.grad.norm().item()) ** 2
                clipped_norm = clipped_norm_sq ** 0.5
                print(f"ZOO: Gradient norm (clipped): {clipped_norm:.6f} (threshold={float(self.max_grad_value):.6g})")
        except Exception:
            pass
        
        # Warn if gradient norm is suspiciously small or large
        if total_grad_norm < 1e-8:
            print(f"WARNING: Extremely small gradient norm ({total_grad_norm:.6e}). This may indicate:")
            print(f"  - Perturbation scale too small (current: {self.perturbation_scale:.6e})")
            print(f"  - Insufficient perturbations (current: {self.sample_count})")
            print(f"  - Loss landscape is flat or numerical precision issues")
        elif total_grad_norm > 1e3:
            print(f"WARNING: Very large gradient norm ({total_grad_norm:.6e}). This may indicate:")
            print(f"  - Instability in training")
            print(f"  - Loss landscape is very steep")
            print(f"  - Consider reducing learning rate")
        