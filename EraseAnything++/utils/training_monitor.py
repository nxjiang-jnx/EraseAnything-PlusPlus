#!/usr/bin/env python
# coding=utf-8
"""
    @date:  2025.10
    @func:  Training Monitor for Open-Sora LoRA
            Handles metrics tracking, plotting, and visualization
"""
import os
import json
import logging
import numpy as np
import matplotlib.pyplot as plt

logger = logging.getLogger(__name__)


class TrainingMonitor:
    """Training monitor for tracking metrics and generating plots"""
    
    def __init__(self, output_dir, enable_monitoring=True):
        """
        Args:
            output_dir: Directory to save monitoring outputs
            enable_monitoring: Whether to enable monitoring (can be disabled via config)
        """
        self.output_dir = output_dir
        self.enable_monitoring = enable_monitoring
        
        if self.enable_monitoring:
            os.makedirs(output_dir, exist_ok=True)
            
        # Training metrics tracking
        self.metrics = {
            'steps': [],
            'esd_loss': [],
            'attn_loss': [],
            'lora_loss': [],
            'infonce_loss': [],
            'total_loss': [],
            'learning_rate': []
        }
    
    def log_step(self, step, esd_loss, attn_loss, lora_loss, infonce_loss, total_loss, lr):
        """Log metrics for a single training step"""
        if not self.enable_monitoring:
            return
            
        self.metrics['steps'].append(step)
        self.metrics['esd_loss'].append(esd_loss)
        self.metrics['attn_loss'].append(attn_loss)
        self.metrics['lora_loss'].append(lora_loss)
        self.metrics['infonce_loss'].append(infonce_loss)
        self.metrics['total_loss'].append(total_loss)
        self.metrics['learning_rate'].append(lr)
    
    def save_metrics(self):
        """Save training metrics to JSON file"""
        if not self.enable_monitoring:
            return
            
        metrics_path = os.path.join(self.output_dir, "training_metrics.json")
        with open(metrics_path, 'w') as f:
            json.dump(self.metrics, f, indent=2)
        logger.info(f"Training metrics saved to {metrics_path}")
    
    def generate_plots(self):
        """Generate and save training loss plots"""
        if not self.enable_monitoring:
            return
            
        if not self.metrics['steps']:
            logger.warning("No metrics to plot")
            return
            
        # Generate main training curves
        self._generate_main_plots()
        
        # Generate combined loss plot
        self._generate_combined_plot()
        
        # Print training summary
        self._print_training_summary()
    
    def _generate_main_plots(self):
        """Generate 6-subplot training curves - Bi-level Optimization Structure"""
        plt.style.use('default')
        fig, axes = plt.subplots(2, 3, figsize=(18, 10))
        fig.suptitle('Bi-level Optimization Training Progress', fontsize=16)
        
        steps = self.metrics['steps']
        
        # Calculate combined losses
        upper_loss = np.array(self.metrics['esd_loss']) + np.array(self.metrics['attn_loss'])
        lower_loss = np.array(self.metrics['lora_loss']) + np.array(self.metrics['infonce_loss'])
        
        # 1. ESD Loss
        axes[0, 0].plot(steps, self.metrics['esd_loss'], 'r-', linewidth=2, label='ESD Loss')
        axes[0, 0].set_title('ESD Loss')
        axes[0, 0].set_xlabel('Steps')
        axes[0, 0].set_ylabel('Loss')
        axes[0, 0].grid(True, alpha=0.3)
        axes[0, 0].legend()
        
        # 2. Attention Loss
        axes[0, 1].plot(steps, self.metrics['attn_loss'], 'g-', linewidth=2, label='Attention Loss')
        axes[0, 1].set_title('Attention Loss')
        axes[0, 1].set_xlabel('Steps')
        axes[0, 1].set_ylabel('Loss')
        axes[0, 1].grid(True, alpha=0.3)
        axes[0, 1].legend()
        
        # 3. LoRA Loss
        axes[0, 2].plot(steps, self.metrics['lora_loss'], 'b-', linewidth=2, label='LoRA Loss')
        axes[0, 2].set_title('LoRA Loss')
        axes[0, 2].set_xlabel('Steps')
        axes[0, 2].set_ylabel('Loss')
        axes[0, 2].grid(True, alpha=0.3)
        axes[0, 2].legend()
        
        # 4. InfoNCE Loss
        axes[1, 0].plot(steps, self.metrics['infonce_loss'], 'm-', linewidth=2, label='InfoNCE Loss')
        axes[1, 0].set_title('InfoNCE Loss')
        axes[1, 0].set_xlabel('Steps')
        axes[1, 0].set_ylabel('Loss')
        axes[1, 0].grid(True, alpha=0.3)
        axes[1, 0].legend()
        
        # 5. Upper Loss (ESD + Attention)
        axes[1, 1].plot(steps, upper_loss, 'orange', linewidth=2, label='Upper Loss (ESD + Attn)')
        axes[1, 1].set_title('Upper Loss (ESD + Attention)')
        axes[1, 1].set_xlabel('Steps')
        axes[1, 1].set_ylabel('Loss')
        axes[1, 1].grid(True, alpha=0.3)
        axes[1, 1].legend()
        
        # 6. Lower Loss (LoRA + InfoNCE)
        axes[1, 2].plot(steps, lower_loss, 'purple', linewidth=2, label='Lower Loss (LoRA + InfoNCE)')
        axes[1, 2].set_title('Lower Loss (LoRA + InfoNCE)')
        axes[1, 2].set_xlabel('Steps')
        axes[1, 2].set_ylabel('Loss')
        axes[1, 2].grid(True, alpha=0.3)
        axes[1, 2].legend()
        
        plt.tight_layout()
        
        # Save plot
        plot_path = os.path.join(self.output_dir, "training_curves.png")
        plt.savefig(plot_path, dpi=300, bbox_inches='tight')
        logger.info(f"Training curves saved to {plot_path}")
        plt.close()
    
    def _generate_combined_plot(self):
        """Generate combined loss plot - Bi-level Optimization Structure"""
        steps = self.metrics['steps']
        
        # Calculate combined losses
        upper_loss = np.array(self.metrics['esd_loss']) + np.array(self.metrics['attn_loss'])
        lower_loss = np.array(self.metrics['lora_loss']) + np.array(self.metrics['infonce_loss'])
        
        plt.figure(figsize=(12, 6))
        
        # Upper Loss components
        plt.plot(steps, self.metrics['esd_loss'], 'r--', linewidth=1.5, label='ESD Loss', alpha=0.7)
        plt.plot(steps, self.metrics['attn_loss'], 'g--', linewidth=1.5, label='Attention Loss', alpha=0.7)
        plt.plot(steps, upper_loss, 'r-', linewidth=2, label='Upper Loss (ESD + Attn)')
        
        # Lower Loss components  
        plt.plot(steps, self.metrics['lora_loss'], 'b--', linewidth=1.5, label='LoRA Loss', alpha=0.7)
        plt.plot(steps, self.metrics['infonce_loss'], 'm--', linewidth=1.5, label='InfoNCE Loss', alpha=0.7)
        plt.plot(steps, lower_loss, 'b-', linewidth=2, label='Lower Loss (LoRA + InfoNCE)')
        
        plt.title('Bi-level Optimization Losses Over Time')
        plt.xlabel('Steps')
        plt.ylabel('Loss')
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        
        combined_path = os.path.join(self.output_dir, "loss_curves_combined.png")
        plt.savefig(combined_path, dpi=300, bbox_inches='tight')
        logger.info(f"Combined loss curves saved to {combined_path}")
        plt.close()
    
    def _print_training_summary(self):
        """Print training summary statistics - Bi-level Optimization Structure"""
        logger.info("=" * 60)
        logger.info("BI-LEVEL OPTIMIZATION TRAINING SUMMARY")
        logger.info("=" * 60)
        
        total_steps = len(self.metrics['steps'])
        
        # Calculate combined losses
        upper_loss = np.array(self.metrics['esd_loss']) + np.array(self.metrics['attn_loss'])
        lower_loss = np.array(self.metrics['lora_loss']) + np.array(self.metrics['infonce_loss'])
        
        # Final values
        final_upper_loss = upper_loss[-1]
        final_lower_loss = lower_loss[-1]
        final_esd_loss = self.metrics['esd_loss'][-1]
        final_attn_loss = self.metrics['attn_loss'][-1]
        final_lora_loss = self.metrics['lora_loss'][-1]
        final_infonce_loss = self.metrics['infonce_loss'][-1]
        
        # Average values
        avg_upper_loss = np.mean(upper_loss)
        avg_lower_loss = np.mean(lower_loss)
        avg_esd_loss = np.mean(self.metrics['esd_loss'])
        avg_attn_loss = np.mean(self.metrics['attn_loss'])
        avg_lora_loss = np.mean(self.metrics['lora_loss'])
        avg_infonce_loss = np.mean(self.metrics['infonce_loss'])
        
        # Calculate trends (last 20% vs first 20%)
        n_20_percent = max(1, total_steps // 5)
        first_20_upper = np.mean(upper_loss[:n_20_percent])
        last_20_upper = np.mean(upper_loss[-n_20_percent:])
        first_20_lower = np.mean(lower_loss[:n_20_percent])
        last_20_lower = np.mean(lower_loss[-n_20_percent:])
        
        upper_improvement = ((first_20_upper - last_20_upper) / first_20_upper) * 100
        lower_improvement = ((first_20_lower - last_20_lower) / first_20_lower) * 100
        
        logger.info(f"Total Steps: {total_steps}")
        logger.info("")
        logger.info("=== UPPER LOSS (ESD + Attention) ===")
        logger.info(f"Final Upper Loss: {final_upper_loss:.6f}")
        logger.info(f"  - Final ESD Loss: {final_esd_loss:.6f}")
        logger.info(f"  - Final Attention Loss: {final_attn_loss:.6f}")
        logger.info(f"Average Upper Loss: {avg_upper_loss:.6f}")
        logger.info(f"  - Average ESD Loss: {avg_esd_loss:.6f}")
        logger.info(f"  - Average Attention Loss: {avg_attn_loss:.6f}")
        logger.info(f"Upper Loss Improvement: {upper_improvement:.2f}%")
        logger.info("")
        logger.info("=== LOWER LOSS (LoRA + InfoNCE) ===")
        logger.info(f"Final Lower Loss: {final_lower_loss:.6f}")
        logger.info(f"  - Final LoRA Loss: {final_lora_loss:.6f}")
        logger.info(f"  - Final InfoNCE Loss: {final_infonce_loss:.6f}")
        logger.info(f"Average Lower Loss: {avg_lower_loss:.6f}")
        logger.info(f"  - Average LoRA Loss: {avg_lora_loss:.6f}")
        logger.info(f"  - Average InfoNCE Loss: {avg_infonce_loss:.6f}")
        logger.info(f"Lower Loss Improvement: {lower_improvement:.2f}%")
        logger.info("=" * 60)
    
    def finish_training(self):
        """Call this at the end of training to save metrics and generate plots"""
        if not self.enable_monitoring:
            return
            
        self.save_metrics()
        self.generate_plots()
