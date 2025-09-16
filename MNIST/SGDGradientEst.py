import torch
import numpy as np

class StochasticGradientApproximator:
    def __init__(self, model_params, perturbation_scale=1e-3, sample_count=1,
                 compute_device=None, data_type=torch.float32):
        self.trainable_params = [param for param in model_params if param.requires_grad]
        self.param_count = sum([param.numel() for param in self.trainable_params])
        self.perturbation_scale = perturbation_scale
        self.sample_count = sample_count
        self.compute_device = compute_device if compute_device is not None else torch.device('cpu')
        self.data_type = data_type
        
        print(f"ZOO Estimator initialized:")
        print(f"  Trainable parameters: {self.param_count}")
        print(f"  Perturbation scale: {self.perturbation_scale}")
        print(f"  Sample count: {self.sample_count}")
        
        # Store original parameters for restoration
        self.original_params = []
        for param in self.trainable_params:
            self.original_params.append(param.data.clone())

    def apply_layerwise_noise(self, noise_generator, alpha):
        """Apply noise perturbations layer by layer (like MNIST version)"""
        for param in self.trainable_params:
            layer_noise = torch.randn(*param.shape, device=self.compute_device, 
                                    dtype=self.data_type, generator=noise_generator)
            param.add_(layer_noise, alpha=alpha)

    def estimate_gradients(self, input_batch, target_labels, objective_fn, random_seed):
        """
        Gradient estimation using finite differences with layerwise noise
        Following the MNIST implementation approach
        """
        if len(self.trainable_params) == 0:
            print("No trainable parameters for ZOO")
            return
        
        # Clear existing gradients
        for param in self.trainable_params:
            if param.grad is not None:
                param.grad.zero_()
            else:
                param.grad = torch.zeros_like(param.data)
        
        directional_samples = []
        
        for sample_idx in range(self.sample_count):
            # Create noise generator with better seeding (like MNIST)
            noise_generator = torch.Generator(device=self.compute_device).manual_seed(
                random_seed * (sample_idx + 17) + sample_idx
            )
            
            # Forward perturbation: θ + μ * z
            self.apply_layerwise_noise(noise_generator, alpha=self.perturbation_scale)
            
            try:
                forward_loss = objective_fn(input_batch, target_labels)
                if isinstance(forward_loss, torch.Tensor):
                    forward_loss = forward_loss.item()
            except Exception as e:
                print(f"Error in forward perturbation: {e}")
                forward_loss = 0.0
            
            # Reset noise generator for same pattern
            noise_generator = torch.Generator(device=self.compute_device).manual_seed(
                random_seed * (sample_idx + 17) + sample_idx
            )
            
            # Backward perturbation: θ - μ * z (by applying -2μ from current state)
            self.apply_layerwise_noise(noise_generator, alpha=-2 * self.perturbation_scale)
            
            try:
                backward_loss = objective_fn(input_batch, target_labels)
                if isinstance(backward_loss, torch.Tensor):
                    backward_loss = backward_loss.item()
            except Exception as e:
                print(f"Error in backward perturbation: {e}")
                backward_loss = 0.0
            
            # Reset noise generator again
            noise_generator = torch.Generator(device=self.compute_device).manual_seed(
                random_seed * (sample_idx + 17) + sample_idx
            )
            
            # Restore original parameters: θ (by applying +μ from current state)
            self.apply_layerwise_noise(noise_generator, alpha=self.perturbation_scale)
            
            # Calculate directional derivative
            directional_derivative = (forward_loss - backward_loss) / (2 * self.perturbation_scale)
            directional_samples.append(directional_derivative)
        
        # Convert to tensor for easier manipulation
        directional_samples = torch.tensor(directional_samples, device=self.compute_device, dtype=self.data_type)
        num_samples = len(directional_samples)
        
        # Accumulate gradients using noise patterns
        for sample_idx, directional_derivative in enumerate(directional_samples):
            # Recreate the same noise pattern
            noise_generator = torch.Generator(device=self.compute_device).manual_seed(
                random_seed * (sample_idx + 17) + sample_idx
            )
            
            for param in self.trainable_params:
                layer_noise = torch.randn(*param.shape, device=self.compute_device, 
                                        dtype=self.data_type, generator=noise_generator)
                
                if sample_idx == 0:
                    # First sample: initialize gradient
                    param.grad = layer_noise.mul_(directional_derivative / num_samples)
                else:
                    # Subsequent samples: accumulate gradient
                    param.grad += layer_noise.mul_(directional_derivative / num_samples)
        
        # Calculate and print gradient statistics for debugging
        total_grad_norm = 0.0
        max_grad = 0.0
        min_grad = float('inf')
        
        for param in self.trainable_params:
            if param.grad is not None:
                grad_norm = param.grad.norm().item()
                total_grad_norm += grad_norm ** 2
                max_grad = max(max_grad, param.grad.abs().max().item())
                min_grad = min(min_grad, param.grad.abs().min().item())
        
        total_grad_norm = total_grad_norm ** 0.5
        avg_loss_diff = directional_samples.mean().item()
        
        print(f"ZOO Stats: Avg loss diff: {avg_loss_diff:.6f}, "
              f"Grad norm: {total_grad_norm:.6f}, "
              f"Max grad: {max_grad:.6f}, Min grad: {min_grad:.9f}")