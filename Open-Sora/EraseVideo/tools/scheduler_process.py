"""
OpenSora Scheduler
2025.10.26 by jnx
"""
import torch
import math
from typing import List, Optional, Union


class OpenSoraFlowMatchScheduler:
    """
    Custom Flow Matching Scheduler for Open-Sora
    """
    def __init__(self):
        with torch.no_grad():
            # create weights for timesteps
            num_timesteps = 1000
            
            self.sigmas = torch.linspace(1.0, 0.0, num_timesteps + 1)
            
            sigma_sqrt_weighing = (self.sigmas[:-1]**-2.0).float()
            sigma_sqrt_weighing = torch.clamp(sigma_sqrt_weighing, max=1e4)
            sigma_sqrt_weighing = sigma_sqrt_weighing / sigma_sqrt_weighing.mean()
            
            timesteps = torch.linspace(1000, 0, num_timesteps, device="cpu")
            
            self.linear_timesteps = timesteps
            self.linear_timesteps_weights = sigma_sqrt_weighing
            self.use_dynamic_shifting = False
            
            self.timesteps = None
    
    def get_weights_for_timesteps(self, timesteps: torch.Tensor) -> torch.Tensor:
        # Get the indices of the timesteps
        step_indices = [(self.timesteps == t).nonzero().item() for t in timesteps]
        
        # Get the weights for the timesteps
        weights = self.linear_timesteps_weights[step_indices].flatten()
        return weights
    
    def get_sigmas(self, timesteps: torch.Tensor, n_dim, dtype, device) -> torch.Tensor:
        sigmas = self.sigmas.to(device=device, dtype=dtype)
        schedule_timesteps = self.timesteps.to(device)
        timesteps = timesteps.to(device)
        step_indices = [(schedule_timesteps == t).nonzero().item() for t in timesteps]
        
        sigma = sigmas[step_indices].flatten()
        while len(sigma.shape) < n_dim:
            sigma = sigma.unsqueeze(-1)
        
        return sigma
    
    def add_noise(
        self,
        original_samples: torch.Tensor,
        noise: torch.Tensor,
        timesteps: torch.Tensor,
    ) -> torch.Tensor:
        ## ref https://github.com/huggingface/diffusers/blob/fbe29c62984c33c6cf9cf7ad120a992fe6d20854/examples/dreambooth/train_dreambooth_sd3.py#L1578
        ## Add noise according to flow matching.
        ## zt = (1 - texp) * x + texp * z1

        # sigmas = get_sigmas(timesteps, n_dim=model_input.ndim, dtype=model_input.dtype)
        # noisy_model_input = (1.0 - sigmas) * model_input + sigmas * noise

        # timestep needs to be in [0, 1], we store them in [0, 1000]
        # noisy_sample = (1 - timestep) * latent + timestep * noise
        t_01 = (timesteps / 1000).to(original_samples.device)
        noisy_model_input = (1 - t_01) * original_samples + t_01 * noise
        
        # n_dim = original_samples.ndim
        # sigmas = self.get_sigmas(timesteps, n_dim, original_samples.dtype, original_samples.device)
        # noisy_model_input = (1.0 - sigmas) * original_samples + sigmas * noise
        return noisy_model_input
    
    def scale_model_input(self, sample: torch.Tensor, timestep: Union[float, torch.Tensor]) -> torch.Tensor:
        return sample
    
    def set_train_timesteps(self, num_timesteps, device, linear=False):
        if linear:
            timesteps = torch.linspace(1000, 0, num_timesteps, device=device)
            self.timesteps = timesteps
            return timesteps
        else:
            # distribute them closer to center. Inference distributes them as a bias toward first
            # Generate values from 0 to 1
            t = torch.sigmoid(torch.randn((num_timesteps,), device=device))
            
            # Scale and reverse the values to go from 1000 to 0
            timesteps = (1 - t) * 1000
            
            # Sort the timesteps in descending order
            timesteps, _ = torch.sort(timesteps, descending=True)
            
            self.timesteps = timesteps.to(device=device)
            
            return timesteps
    
    def step(
        self,
        model_output: torch.Tensor,
        timestep: torch.Tensor,
        sample: torch.Tensor,
        return_dict: bool = True,
    ):
        """
        one step denoising
        """
        # ensure timestep is a scalar
        if timestep.dim() > 0:
            timestep = timestep[0]
        
        # convert to [0, 1] range
        t = timestep / 1000.0
        
        # calculate the step size (assuming uniform step size)
        # in actual flow matching, we need to know the next timestep
        # here we simplify it to a fixed step size
        if self.timesteps is not None and len(self.timesteps) > 1:
            # find the current timestep in the sequence
            current_idx = (self.timesteps == timestep).nonzero()
            if len(current_idx) > 0 and current_idx[0].item() < len(self.timesteps) - 1:
                next_timestep = self.timesteps[current_idx[0].item() + 1]
                dt = (timestep - next_timestep) / 1000.0
            else:
                dt = 0.01  # default step size
        else:
            dt = 0.01
        
        # Euler step: x_{t-dt} = x_t - dt * v_theta(x_t, t)
        # model_output is the predicted velocity v_theta
        prev_sample = sample - dt * model_output
        
        if return_dict:
            return (prev_sample,)
        else:
            return (prev_sample,)
