import torch
import numpy as np

class StochasticGradientApproximator:
    def __init__(self, model_params, perturbation_scale=1e-3, sample_count=1,
                 compute_device=None, data_type=torch.float32, estimator_type: str = 'central'):
        self.trainable_params = [param for param in model_params if param.requires_grad]
        self.param_count = sum([param.numel() for param in self.trainable_params])
        self.perturbation_scale = perturbation_scale
        self.sample_count = sample_count
        self.compute_device = compute_device if compute_device is not None else torch.device('cpu')
        self.data_type = data_type
        self.estimator_type = str(estimator_type).lower()
        
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
            # Generate random direction
            torch.manual_seed(random_seed + sample_idx)
            directions = []
            for param in self.trainable_params:
                direction = torch.randint_like(param, low=0, high=2, dtype=torch.float32) * 2.0 - 1.0
                directions.append(direction)
            
            # Forward perturbation: θ + μ * z
            for param, direction in zip(self.trainable_params, directions):
                param.data.add_(direction, alpha=self.perturbation_scale)
            try:
                loss_plus = objective_fn(input_batch, target_labels)
            except Exception as e:
                print(f"Error in forward perturbation: {e}")
                loss_plus = torch.tensor(0.0, device=self.compute_device)
            
            if self.estimator_type == 'central':
                # Backward perturbation: θ - μ * z
                for param, direction in zip(self.trainable_params, directions):
                    param.data.add_(direction, alpha=-2.0 * self.perturbation_scale)
                try:
                    loss_minus = objective_fn(input_batch, target_labels)
                except Exception as e:
                    print(f"Error in backward perturbation: {e}")
                    loss_minus = torch.tensor(0.0, device=self.compute_device)
            else:
                # Forward (one-sided) estimator: use f(θ+μz) - f(θ)
                # Evaluate f(θ) approximately by restoring params then running objective once
                for param, direction in zip(self.trainable_params, directions):
                    param.data.add_(direction, alpha=-1.0 * self.perturbation_scale)
                try:
                    loss_minus = objective_fn(input_batch, target_labels)
                except Exception as e:
                    print(f"Error in base evaluation: {e}")
                    loss_minus = torch.tensor(0.0, device=self.compute_device)
            
            # Restore original parameters
            for param, direction in zip(self.trainable_params, directions):
                param.data.add_(direction, alpha=self.perturbation_scale)
            
            # Calculate finite difference
            if isinstance(loss_plus, torch.Tensor):
                loss_plus = loss_plus.item()
            if isinstance(loss_minus, torch.Tensor):
                loss_minus = loss_minus.item()
            
            fd_num = (loss_plus - loss_minus)
            denom = (2.0 * self.perturbation_scale) if self.estimator_type == 'central' else (self.perturbation_scale)
            finite_diff = fd_num / denom
            total_loss_diff += finite_diff
            loss_diffs.append(abs(fd_num))
            
            # Accumulate gradients: g += (f(θ+μz) - f(θ-μz)) / (2μ) * z
            for param, direction in zip(self.trainable_params, directions):
                param.grad.add_(direction, alpha=finite_diff)
        
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
