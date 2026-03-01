#!/usr/bin/env python
# coding=utf-8
"""
    @date:  2025.10
    @func:  Training Open-Sora LoRA for Video Concept Erasure
            Adapted from EraseAnything for MMDiT architecture
            Using ESD + Attention Deactivation + InfoNCE losses
"""
import os
import argparse
import yaml
import time
import copy
import math
import logging
import random
from pathlib import Path
from contextlib import nullcontext

import torch
import torch.nn as nn
import torch.utils.checkpoint
from peft import LoraConfig, get_peft_model, get_peft_model_state_dict
from tqdm.auto import tqdm

# Open-Sora imports
from opensora.registry import MODELS, build_module
from opensora.utils.config import parse_configs
from opensora.utils.misc import to_torch_dtype
from opensora.acceleration.checkpoint import set_grad_checkpoint

# Custom modules
from video_lora_dataset import VideoLoraDataset, collate_video_fn
from utils.calc_video_loss import (
    calculate_upper_loss,
    calculate_lower_loss
)
from utils.eupmu import EU
from utils.training_monitor import TrainingMonitor
from tools.ir_concept import UniversalModelCaller, MoE
from tools.scheduler_process import OpenSoraFlowMatchScheduler
from opensora.utils.sampling import time_shift, get_res_lin_function

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def finetune(args):
    device = torch.device(f"cuda:{args.device}" if torch.cuda.is_available() else "cpu")
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Data type
    weight_dtype = torch.float32
    if args.mixed_precision == "fp16":
        weight_dtype = torch.float16
    elif args.mixed_precision == "bf16":
        weight_dtype = torch.bfloat16
    
    logger.info("=" * 50)
    logger.info("Loading Open-Sora MMDiT Model...")
    logger.info("=" * 50)
    
    # Build Open-Sora model
    class SimpleConfig:
        def __init__(self, model_config):
            self.model = model_config
            
        def get(self, key, default=None):
            return getattr(self, key, default)
    
    model_config = {
        "type": "flux",
        "from_pretrained": args.pretrained_model_path,
        "guidance_embed": False,
        "fused_qkv": False,
        "use_liger_rope": True,
        "in_channels": 64,
        "vec_in_dim": 768,
        "context_in_dim": 4096,
        "hidden_size": 3072,
        "mlp_ratio": 4.0,
        "num_heads": 24,
        "depth": 19,
        "depth_single_blocks": 38,
        "axes_dim": [16, 56, 56],
        "theta": 10_000,
        "qkv_bias": True,
        "cond_embed": True,
    }
    
    cfg = SimpleConfig(model_config)
    transformer = build_module(model_config, MODELS, device_map=device, torch_dtype=weight_dtype)
    
    # Freeze base model
    transformer.requires_grad_(False)
    
    # Load VAE
    logger.info("Loading VAE...")
    vae_config = {
        "type": "hunyuan_vae",
        "from_pretrained": args.vae_path,
        "in_channels": 3,
        "out_channels": 3,
        "layers_per_block": 2,
        "latent_channels": 16,
        "use_spatial_tiling": True,
        "use_temporal_tiling": False,
    }
    vae = build_module(vae_config, MODELS, device_map=device, torch_dtype=weight_dtype)
    vae.requires_grad_(False)
    vae.eval()
    
    # Load Text Encoders
    logger.info("Loading Text Encoders...")
    t5_config = {
        "type": "text_embedder",
        "from_pretrained": args.t5_path,
        "max_length": 512,
        "shardformer": False,
    }
    clip_config = {
        "type": "text_embedder",
        "from_pretrained": args.clip_path,
        "max_length": 77,
    }
    
    model_t5 = build_module(t5_config, MODELS, device_map=device, torch_dtype=weight_dtype)
    model_clip = build_module(clip_config, MODELS, device_map=device, torch_dtype=weight_dtype)
    model_t5.requires_grad_(False)
    model_clip.requires_grad_(False)
    model_t5.eval()
    model_clip.eval()
    tokenizer_t5 = model_t5.tokenizer
    
    # Configure LoRA
    logger.info("Adding LoRA adapters...")
    if args.lora_layers is not None:
        target_modules = [layer.strip() for layer in args.lora_layers.split(",")]
    else:
        # Target modules for Open-Sora MMDiT
        target_modules = []
        
        # DoubleStreamBlock
        for i in range(19):
            target_modules.extend([
                # f"double_blocks.{i}.img_attn.q_proj",
                # f"double_blocks.{i}.img_attn.k_proj", 
                # f"double_blocks.{i}.img_attn.v_proj",
                # f"double_blocks.{i}.img_attn.proj",
                f"double_blocks.{i}.txt_attn.q_proj",
                f"double_blocks.{i}.txt_attn.k_proj",
            ])
    
    lora_config = LoraConfig(
        r=args.rank,
        lora_alpha=args.rank,
        init_lora_weights="gaussian",
        target_modules=target_modules,
    )
    
    # Use get_peft_model to wrap the transformer with LoRA
    transformer = get_peft_model(transformer, lora_config)
    transformer.train()
    
    # Print trainable parameters
    trainable_params = 0
    all_params = 0
    for name, param in transformer.named_parameters():
        all_params += param.numel()
        if param.requires_grad:
            trainable_params += param.numel()

    # Enable gradient checkpointing
    if args.gradient_checkpointing:
        set_grad_checkpoint(transformer)
    
    # Get trainable parameters
    lora_parameters = list(filter(lambda p: p.requires_grad, transformer.parameters()))
    logger.info(f"Number of trainable LoRA parameters: {sum(p.numel() for p in lora_parameters):,}")
    
    # Optimizer
    if args.use_8bit_adam:
        try:
            import bitsandbytes as bnb
            optimizer_class = bnb.optim.AdamW8bit
        except ImportError:
            raise ImportError("To use 8-bit Adam, install bitsandbytes: pip install bitsandbytes")
    else:
        optimizer_class = torch.optim.AdamW
    
    optimizer = optimizer_class(
        lora_parameters,
        lr=float(args.learning_rate),
        betas=(args.adam_beta1, args.adam_beta2),
        weight_decay=float(args.adam_weight_decay),
        eps=float(args.adam_epsilon),
    )
    
    # Loss function
    criteria = nn.MSELoss()
    
    # Load scheduler
    noise_scheduler = OpenSoraFlowMatchScheduler()
    noise_scheduler_copy = copy.deepcopy(noise_scheduler)
    
    # Dataset
    logger.info("Building dataset...")
    train_dataset = VideoLoraDataset(
        video_data_root=args.video_data_dir,
        instance_prompt=args.instance_prompt,
        key_word=args.key_word,
        tokenizer_t5=tokenizer_t5,
        resolution=args.resolution,
        num_frames=args.num_frames,
        repeats=args.repeats,
    )
    
    train_dataloader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=args.train_batch_size,
        shuffle=True,
        collate_fn=collate_video_fn,
        num_workers=args.dataloader_num_workers,
    )
    
    # Compute text embeddings helper
    def compute_text_embeddings(prompt):
        with torch.no_grad():
            # Simplified version - you may need to implement full encode_prompt
            t5_out = model_t5(prompt)
            clip_out = model_clip(prompt)
            return t5_out, clip_out
        
    # We need to recalculate our total training steps as the size of the training dataloader may have changed.
    num_update_steps_per_epoch = math.ceil(len(train_dataloader) / args.gradient_accumulation_steps)
    # Afterwards we recalculate our number of training epochs
    args.num_train_epochs = math.ceil(args.max_train_steps / num_update_steps_per_epoch)

    # Train!
    total_batch_size = args.train_batch_size * args.gradient_accumulation_steps
        
    logger.info("***** Running training *****")
    logger.info(f"  Num examples = {len(train_dataset)}")
    logger.info(f"  Num batches each epoch = {len(train_dataloader)}")
    logger.info(f"  Num Epochs = {args.num_train_epochs}")
    logger.info(f"  Instantaneous batch size per device = {args.train_batch_size}")
    logger.info(f"  Total train batch size (w. parallel, distributed & accumulation) = {total_batch_size}")
    logger.info(f"  Gradient Accumulation steps = {args.gradient_accumulation_steps}")
    logger.info(f"  Total optimization steps = {args.max_train_steps}")
    
    global_step = 0
    progress_bar = tqdm(range(0, args.max_train_steps), desc="Steps")
    
    # Initialize training monitor 
    monitor = TrainingMonitor(
        output_dir=args.output_dir,
        enable_monitoring=getattr(args, 'enable_monitoring', True)
    )
    
    # Prepare for irrelevant concepts (InfoNCE)
    # model_caller = UniversalModelCaller(args.api_keys) if hasattr(args, 'api_keys') else None
    # irrelevant_concepts = MoE(model_caller, args.key_word, 3, model_list=["gpt"])
    irrelevant_concepts = ["computer", "clothing", "beautiful"]
    
    print("[20251014] negative prompt!", str(args.prompt_b))
    
    use_eupmu = getattr(args, 'use_eupmu', False)
    eupmu = None
    if use_eupmu:
        eupmu = EU(
            device=device,
            gamma=getattr(args, 'eupmu_gamma', 0.01),
            w_lr=getattr(args, 'eupmu_w_lr', 0.3),
            error=getattr(args, 'eupmu_error', 0.003),
            log_loss=getattr(args, 'eupmu_log_loss', False),
        )
        logger.info(f"[EUPMU] Enabled with gamma={eupmu.error}, w_lr={eupmu.w_opt.param_groups[0]['lr']}, error={eupmu.error}")
    else:
        logger.info("[Training] Using alternating training mode (no EUPMU)")
    
    for epoch in range(args.num_train_epochs):
        transformer.train()
        for step, batch in enumerate(train_dataloader):
            # Move batch to device
            batch["videos"] = batch["videos"].to(device, dtype=weight_dtype)
            
            if use_eupmu:
                # (A) Upper Loss (forget task: ESD + Attn)
                upper_loss, t_enc = calculate_upper_loss(
                    transformer=transformer,
                    vae=vae,
                    model_t5=model_t5,
                    model_clip=model_clip,
                    batch=batch,
                    noise_scheduler=noise_scheduler_copy,
                    tokenizer_t5=tokenizer_t5,
                    criteria=criteria,
                    weight_dtype=weight_dtype,
                    device=device,
                    args=args,
                )
                forget_loss = float(args.lamb1) * upper_loss[0] + float(args.lamb2) * upper_loss[1]
                
                # (B) Lower Loss (retain task: LoRA + InfoNCE)
                lower_loss = calculate_lower_loss(
                    transformer=transformer,
                    vae=vae,
                    model_t5=model_t5,
                    model_clip=model_clip,
                    batch=batch,
                    noise_scheduler=noise_scheduler_copy,
                    tokenizer_t5=tokenizer_t5,
                    criteria=criteria,
                    weight_dtype=weight_dtype,
                    device=device,
                    t_enc=t_enc,
                    ir_concept_lst=irrelevant_concepts,
                    args=args,
                )
                retain_loss = float(args.lamb3) * lower_loss[0] + float(args.lamb4) * lower_loss[1]
                
                # Use EUPMU to dynamically balance losses
                loss = eupmu.get_weighted_loss(retain_loss, forget_loss)
                
                # Record metrics
                esd_loss_val = float(args.lamb1) * upper_loss[0].detach().item()
                attn_loss_val = float(args.lamb2) * upper_loss[1].detach().item()
                lora_loss_val = float(args.lamb3) * lower_loss[0].detach().item()
                infonce_loss_val = float(args.lamb4) * lower_loss[1].detach().item()
                total_loss_val = loss.detach().item()
                lr_val = optimizer.param_groups[0]["lr"]
                
                logs = {
                    "esd": esd_loss_val,
                    "attn": attn_loss_val,
                    "lora": lora_loss_val,
                    "infonce": infonce_loss_val,
                    "total": total_loss_val,
                    "eupmu_w": eupmu.w.detach().item(),
                    "prompt": batch["prompts"][0][:50] + "..." if len(batch["prompts"][0]) > 50 else batch["prompts"][0],
                    "lr": lr_val,
                }
                
            else:
                # Alternating mode: train forget and retain tasks separately
                if global_step % 2 == 0 or (float(args.lamb3) + float(args.lamb4)) == 0.0:
                    # Upper Loss (ESD + Attn)
                    upper_loss, t_enc = calculate_upper_loss(
                        transformer=transformer,
                        vae=vae,
                        model_t5=model_t5,
                        model_clip=model_clip,
                        batch=batch,
                        noise_scheduler=noise_scheduler_copy,
                        tokenizer_t5=tokenizer_t5,
                        criteria=criteria,
                        weight_dtype=weight_dtype,
                        device=device,
                        args=args,
                    )
                    loss = float(args.lamb1) * upper_loss[0] + float(args.lamb2) * upper_loss[1]
                    
                    # Record metrics
                    esd_loss_val = float(args.lamb1) * upper_loss[0].detach().item()
                    attn_loss_val = float(args.lamb2) * upper_loss[1].detach().item()
                    lora_loss_val = 0.0
                    infonce_loss_val = 0.0
                    total_loss_val = loss.detach().item()
                    lr_val = optimizer.param_groups[0]["lr"]
                    
                    logs = {
                        "esd": esd_loss_val,
                        "attn": attn_loss_val,
                        "lora": lora_loss_val,
                        "infonce": infonce_loss_val,
                        "total": total_loss_val,
                        "prompt": batch["prompts"][0][:50] + "..." if len(batch["prompts"][0]) > 50 else batch["prompts"][0],
                        "lr": lr_val,
                    }
                    
                else:
                    # Lower Loss (LoRA + InfoNCE)                
                    lower_loss = calculate_lower_loss(
                        transformer=transformer,
                        vae=vae,
                        model_t5=model_t5,
                        model_clip=model_clip,
                        batch=batch,
                        noise_scheduler=noise_scheduler_copy,
                        tokenizer_t5=tokenizer_t5,
                        criteria=criteria,
                        weight_dtype=weight_dtype,
                        device=device,
                        t_enc=t_enc,
                        ir_concept_lst=irrelevant_concepts,
                        args=args,
                    )
                    loss = float(args.lamb3) * lower_loss[0] + float(args.lamb4) * lower_loss[1]
                    
                    # Record metrics
                    esd_loss_val = 0.0
                    attn_loss_val = 0.0
                    lora_loss_val = float(args.lamb3) * lower_loss[0].detach().item()
                    infonce_loss_val = float(args.lamb4) * lower_loss[1].detach().item()
                    total_loss_val = loss.detach().item()
                    lr_val = optimizer.param_groups[0]["lr"]
                    
                    logs = {
                        "esd": esd_loss_val,
                        "attn": attn_loss_val,
                        "lora": lora_loss_val,
                        "infonce": infonce_loss_val,
                        "total": total_loss_val,
                        "prompt": batch["prompts"][0][:50] + "..." if len(batch["prompts"][0]) > 50 else batch["prompts"][0],
                        "lr": lr_val,
                    }
            
            # Log metrics to monitor
            monitor.log_step(global_step, esd_loss_val, attn_loss_val, lora_loss_val, infonce_loss_val, total_loss_val, lr_val)
            
            progress_bar.set_postfix(**logs)
            
            # Log detailed metrics
            if global_step % 11 == 0:
                logger.info(f"Step {global_step}: ESD={esd_loss_val:.6f}, Attn={attn_loss_val:.6f}, LoRA={lora_loss_val:.6f}, InfoNCE={infonce_loss_val:.6f}, Total={total_loss_val:.6f}")
                print("\n")
            
            # Backward
            loss.backward()
            
            if (step + 1) % args.gradient_accumulation_steps == 0:
                optimizer.step()
                optimizer.zero_grad()
                
                # Update EUPMU weights after gradient update
                if use_eupmu:
                    eupmu_update_freq = getattr(args, 'eupmu_update_freq', 1)
                    if global_step % eupmu_update_freq == 0:
                        with torch.no_grad():
                            # Recalculate retain loss after update
                            lower_loss_updated = calculate_lower_loss(
                                transformer=transformer,
                                vae=vae,
                                model_t5=model_t5,
                                model_clip=model_clip,
                                batch=batch,
                                noise_scheduler=noise_scheduler_copy,
                                tokenizer_t5=tokenizer_t5,
                                criteria=criteria,
                                weight_dtype=weight_dtype,
                                device=device,
                                t_enc=t_enc,
                                ir_concept_lst=irrelevant_concepts,
                                args=args,
                            )
                            retain_loss_updated = float(args.lamb3) * lower_loss_updated[0] + float(args.lamb4) * lower_loss_updated[1]
                            eupmu.update(retain_loss_updated, curr_lr=lr_val)
            
            progress_bar.update(1)
            global_step += 1
            
            # Save checkpoint
            if global_step % args.checkpointing_steps == 0:
                save_path = os.path.join(args.output_dir, f"checkpoint-{global_step}")
                os.makedirs(save_path, exist_ok=True)
                
                # Save LoRA weights using PEFT's save method
                transformer.save_pretrained(save_path)
                logger.info(f"Saved checkpoint to {save_path}")
        
        if global_step >= args.max_train_steps:
            break
    
    # Final save
    final_path = os.path.join(args.output_dir, "final")
    os.makedirs(final_path, exist_ok=True)
    transformer.save_pretrained(final_path)
    logger.info(f"Training finished! Final weights saved to {final_path}")
    
    # Finish monitoring (save metrics and generate plots)
    monitor.finish_training()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train Open-Sora LoRA for video concept erasure")
    parser.add_argument('--config', type=str, required=True, help='Path to config file')
    args = parser.parse_args()
    
    def read_config(config_path):
        with open(config_path, 'r') as f:
            return yaml.safe_load(f)
    
    config = read_config(args.config)
    for key, value in config.items():
        setattr(args, key, value)
    
    finetune(args)

