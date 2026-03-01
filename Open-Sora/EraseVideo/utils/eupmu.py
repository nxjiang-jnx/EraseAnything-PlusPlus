import torch
import torch.nn.functional as F
from typing import Dict, List, Tuple, Union


class EU:
    """
    Efficient Utility-Preserving Machine Unlearning (EUPMU)
    Fast Adaptive Multitask Optimization using Implicit Gradient Surgery.
    """
    def __init__(
        self,
        device: torch.device,
        gamma: float = 0.01,   # the regularization coefficient
        w_lr: float = 0.3,   # the learning rate of the task lambda
        max_norm: float = 1.0, # the maximum gradient norm
        error: float = 0.003, # the error term
        log_loss: bool = False, # whether to log the loss (set to False for better numerical stability)

    ):
        self.min_losses = torch.zeros(2).to(device)
        self.w = torch.tensor([0.], device=device, requires_grad=True)
        self.w_opt = torch.optim.Adam([self.w], lr=w_lr, weight_decay=gamma)
        self.max_norm = max_norm
        self.n_tasks = 2
        self.device = device
        self.error = error
        self.log_loss = log_loss
        self.prev_ret_loss = None


    def set_min_losses(self, losses):
        """Set minimum losses for normalization."""
        self.min_losses = losses

    def get_weighted_loss(self, ret_loss, fgt_loss):
        """
        Compute weighted loss using EUPMU method.
        
        Args:
            ret_loss: retain loss
            fgt_loss: forget loss
        
        Returns:
            weighted_loss: dynamically weighted combination of the two losses
        """
        losses = torch.stack([ret_loss, fgt_loss]).to(self.device)
        self.prev_ret_loss = ret_loss.detach().clone()
        
        # Normalize losses
        D = losses - self.min_losses + 1e-12
        if self.log_loss:
            D = D.log()
        
        D_copy = D.clone()
        
        # Apply dynamic weighting based on learnable weight w
        if self.w < 0:
            D_copy[0] = D[0] * 0
        else:
            D_copy[0] = D[0] * (self.w/(1 + self.w))
            D_copy[1] = D[1] * (1/(1 + self.w))
        
        loss = D_copy.sum()
        return loss

    def update(self, curr_ret_loss, curr_lr=None):
        """
        Update the task weighting using implicit gradient surgery.
        
        Args:
            curr_ret_loss: the current retain loss after gradient update
            curr_lr: the learning rate of the model (optional, for adaptive update)
        """
        if self.prev_ret_loss is None:
            # First update, skip
            return
            
        if curr_lr is not None:
            if self.log_loss:
                delta = ((self.prev_ret_loss - self.min_losses[0] + 1e-12).log() - 
                        (curr_ret_loss - self.min_losses[0] + 1e-12).log())/(curr_lr + 1e-12) - self.error
            else:
                delta = ((self.prev_ret_loss) - (curr_ret_loss))/(curr_lr + 1e-12) - self.error
        else:
            if self.log_loss:
                delta = (self.prev_ret_loss - self.min_losses[0] + 1e-12).log() - \
                        (curr_ret_loss - self.min_losses[0] + 1e-12).log() - self.error
            else:
                delta = (self.prev_ret_loss) - (curr_ret_loss) - self.error
        
        d = delta.unsqueeze(0)
        self.w_opt.zero_grad()
        self.w.grad = d
        self.w_opt.step()

    def apply_gradient_surgery(self, ret_loss, fgt_loss, shared_parameters):
        """
        Apply explicit gradient surgery: remove the component of retain gradient 
        that is in the opposite direction of forget gradient.
        
        This implements the core idea: "沿着擦除反方向的保留分量裁剪掉"
        (Remove the retain component in the opposite direction of forget)
        
        This is the PCGrad-style explicit gradient surgery mentioned in EUPMU paper.
        Corresponds to the gradient projection operation in multi-task learning.
        
        Args:
            ret_loss: retain loss (retain任务的loss)
            fgt_loss: forget loss (forget任务的loss)
            shared_parameters: List of shared parameters between tasks
        """
        if shared_parameters is None or len(shared_parameters) == 0:
            return
            
        # Step 1: Compute gradients separately for retain and forget tasks
        # First, zero out existing gradients
        for param in shared_parameters:
            if param.grad is not None:
                param.grad.zero_()
        
        # Compute retain gradient
        retain_grads = []
        if ret_loss.requires_grad:
            ret_loss.backward(retain_graph=True)
            for param in shared_parameters:
                if param.grad is not None:
                    retain_grads.append(param.grad.clone())
                    param.grad.zero_()
        
        # Compute forget gradient
        forget_grads = []
        if fgt_loss.requires_grad:
            fgt_loss.backward(retain_graph=True)
            for param in shared_parameters:
                if param.grad is not None:
                    forget_grads.append(param.grad.clone())
                    param.grad.zero_()
        
        if len(retain_grads) == 0 or len(forget_grads) == 0:
            # If no gradients, use weighted loss
            weighted_loss = self.get_weighted_loss(ret_loss, fgt_loss)
            weighted_loss.backward()
            return
        
        # Step 2: Flatten gradients for computation
        retain_grad_flat = torch.cat([g.flatten() for g in retain_grads])
        forget_grad_flat = torch.cat([g.flatten() for g in forget_grads])
        
        # Step 3: Check if gradients conflict (negative inner product)
        grad_inner_product = torch.dot(retain_grad_flat, forget_grad_flat)
        
        # Step 4: Apply gradient surgery if gradients conflict
        if grad_inner_product < 0:  # Gradients conflict
            # Project retain gradient to remove component opposite to forget gradient
            # This is the PCGrad-style gradient surgery
            # Formula: retain_grad_projected = retain_grad - (retain_grad · forget_grad) / ||forget_grad||^2 * forget_grad
            forget_grad_norm_sq = torch.dot(forget_grad_flat, forget_grad_flat)
            if forget_grad_norm_sq > 1e-8:
                proj_coeff = grad_inner_product / forget_grad_norm_sq
                retain_grad_projected = retain_grad_flat - proj_coeff * forget_grad_flat
                
                # Step 5: Update gradients in parameters
                # Apply dynamic weighting from EUPMU
                w_retain = self.w / (1 + self.w) if self.w >= 0 else 0.0
                w_forget = 1.0 / (1 + self.w)
                
                idx = 0
                for i, param in enumerate(shared_parameters):
                    if param.grad is None:
                        param.grad = torch.zeros_like(param)
                    param_size = param.numel()
                    # Combine projected retain gradient with forget gradient using EUPMU weights
                    retain_part = w_retain * retain_grad_projected[idx:idx+param_size].view(param.shape)
                    forget_part = w_forget * forget_grads[i]
                    param.grad = retain_part + forget_part
                    idx += param_size
        else:
            # No conflict, use weighted combination
            w_retain = self.w / (1 + self.w) if self.w >= 0 else 0.0
            w_forget = 1.0 / (1 + self.w)
            
            for i, param in enumerate(shared_parameters):
                if param.grad is None:
                    param.grad = torch.zeros_like(param)
                param.grad = w_retain * retain_grads[i] + w_forget * forget_grads[i]

    def backward(
        self,
        losses: torch.Tensor,
        shared_parameters: Union[
            List[torch.nn.parameter.Parameter], torch.Tensor
        ] = None,
        not_backward: bool = False,
    ) -> Union[torch.Tensor, None]:
        """
        Modified to work with Accelerator by ensuring gradients are properly detached.
        """
        loss = self.get_weighted_loss(losses[0], losses[1])
        if self.max_norm > 0 and shared_parameters is not None:
            torch.nn.utils.clip_grad_norm_(shared_parameters, self.max_norm)
        if not_backward:
            return loss
        loss.backward()
        return loss

