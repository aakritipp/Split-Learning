import torch

class StochasticGradientApproximator:
    def __init__(self, model_params, perturbation_scale=1e-3, sample_count=1,
                 compute_device=None, data_type=torch.float32):
        
        self.trainable_params = [param for param in model_params if param.requires_grad]
        self.param_count = sum([param.numel() for param in self.trainable_params])
        print(f"Client total trainable parameters: {self.param_count}")

        self.perturbation_scale = perturbation_scale
        self.sample_count = sample_count
        self.compute_device = compute_device
        self.data_type = data_type

    def apply_layerwise_noise(self, noise_generator, alpha):
        """Apply noise perturbations layer by layer"""
        for param in self.trainable_params:
            layer_noise = torch.randn(*param.shape, device=self.compute_device, 
                                    dtype=self.data_type, generator=noise_generator)
            param.add_(layer_noise, alpha=alpha)

    def estimate_gradients(self, input_batch, target_labels, objective_fn, random_seed):
        """Main gradient estimation procedure
        Generate and assign layer-wise gradients"""
        directional_samples = []
            
        for sample_idx in range(self.sample_count):
            noise_generator = torch.Generator(device=self.compute_device).manual_seed(
                random_seed * (sample_idx + 17) + sample_idx
            )
            
            # Forward perturbation
            self.apply_layerwise_noise(noise_generator, alpha=self.perturbation_scale)
            forward_loss = objective_fn(input_batch, target_labels)
            
            # second forward perturbation
            noise_generator = torch.Generator(device=self.compute_device).manual_seed(
                random_seed * (sample_idx + 17) + sample_idx
            )  # Reset for same noise pattern
            self.apply_layerwise_noise(noise_generator, alpha=-2 * self.perturbation_scale)
            second_forward_loss = objective_fn(input_batch, target_labels)
            
            # Restore model
            noise_generator = torch.Generator(device=self.compute_device).manual_seed(
                random_seed * (sample_idx + 17) + sample_idx
            )  # Reset for same noise pattern
            self.apply_layerwise_noise(noise_generator, alpha=self.perturbation_scale)
            
            # Calculate directional derivative
            directional_derivative = (forward_loss - second_forward_loss) / (2 * self.perturbation_scale)
            directional_samples.append(directional_derivative)
            
        directional_samples = torch.tensor(directional_samples, device=self.compute_device)
        num_samples = len(directional_samples)
        
        for sample_idx, directional_derivative in enumerate(directional_samples):
            noise_generator = torch.Generator(device=self.compute_device).manual_seed(
                random_seed * (sample_idx + 17) + sample_idx
            )
            
            for param in self.trainable_params:
                layer_noise = torch.randn(*param.shape, device=self.compute_device, 
                                        dtype=self.data_type, generator=noise_generator)
                
                if sample_idx == 0:
                    param.grad = layer_noise.mul_(directional_derivative / num_samples)
                else:
                    param.grad += layer_noise.mul_(directional_derivative / num_samples)