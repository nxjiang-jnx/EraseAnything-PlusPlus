"""
Loss calculation for Open-Sora video concept erasure
Implements: ESD Loss, Attention Deactivation Loss, InfoNCE Loss
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from opensora.utils.sampling import pack, prepare_ids, time_shift, get_res_lin_function
from .find_token import search_sequence_numpy, get_word_index
from .opensora_utils import latent_sample, predict_noise, prepare_position_ids, opensora_pack_latents
import math
import random
import numpy as np

def encode_video(vae, video, weight_dtype):
    """Encode video to latent space"""
    with torch.no_grad():
        video = video.to(vae.device, dtype=weight_dtype)
        latents = vae.encode(video)  # Returns latent distribution
        if hasattr(latents, 'sample'):
            latents = latents.sample()
        elif hasattr(latents, 'latent_dist'):
            latents = latents.latent_dist.sample()
    return latents


def encode_text(model_t5, model_clip, prompts, device, weight_dtype):
    """Encode text prompts"""
    with torch.no_grad():
        # T5 encoding
        t5_out = model_t5(prompts)
        if isinstance(t5_out, tuple):
            t5_embedding = t5_out[0]
        else:
            t5_embedding = t5_out
        
        # CLIP encoding
        clip_out = model_clip(prompts)
        if isinstance(clip_out, tuple):
            clip_embedding = clip_out[0]
        else:
            clip_embedding = clip_out
        
        t5_embedding = t5_embedding.to(device, dtype=weight_dtype)
        clip_embedding = clip_embedding.to(device, dtype=weight_dtype)
    
    return t5_embedding, clip_embedding


def calculate_upper_loss(
    transformer,
    vae,
    model_t5,
    model_clip,
    batch,
    noise_scheduler,
    tokenizer_t5,
    criteria,
    weight_dtype,
    device,
    args,
):
    """
    2025.10.17 by jnx
    Calculate Upper Loss (ESD Loss + Attn Loss)
    Following EraseAnything/utils/calc_loss.py::calculate_upper_loss line 321-445
    """
    
    videos = batch["videos"]
    prompts = batch["prompts"]
    key_word = getattr(args, 'key_word', None)
    negative_guidance = float(getattr(args, 'negative_guidance', 1.0))
    start_guidance = float(getattr(args, 'guidance', 3.0))
    ddim_steps = getattr(args, 'ddim_steps', 28)
    
    batch_size = videos.shape[0]
    
    # Encode video to latents
    latents = encode_video(vae, videos, weight_dtype)  # (bs, c, t, h, w)
    latents_packed = opensora_pack_latents(latents, patch_size=2)  # (bs, seq_len, c)
    
    # Encode prompts
    t5_0, clip_0 = encode_text(model_t5, model_clip, [""] * batch_size, device, weight_dtype)
    t5_p, clip_p = encode_text(model_t5, model_clip, prompts, device, weight_dtype)
    
    # Sample timestep
    t_enc = torch.randint(ddim_steps, (1,), device=device)
    # time step from 1000 to 0 (0 being good)
    og_num = round((int(t_enc)/ddim_steps)*1000)
    og_num_lim = round((int(t_enc+1)/ddim_steps)*1000)
    t_enc_ddpm = torch.randint(og_num, og_num_lim, (1,), device=device)
    
    # Create position IDs
    img_ids, txt_ids = prepare_position_ids(latents, t5_p, device, weight_dtype)
    
    # Prepare guidance
    start_guidance = 3
    start_guidance = torch.tensor([start_guidance], device=device)
    start_guidance = start_guidance.expand(latents_packed.shape[0])
    
    # Generate an image/video with the concept from ESD model
    with torch.no_grad():
        # generatea a video with latent_sample
        z, latent_image_ids = latent_sample(transformer,
                                            noise_scheduler,
                                            batch_size,
                                            latents_packed,
                                            t5_p,
                                            clip_p,
                                            img_ids,
                                            txt_ids,
                                            start_guidance,
                                            int(ddim_steps),
                                            device,
                                            weight_dtype,
                                            return_attn=False
                                        )
        
        # e_0 & e_p
        e_0 = predict_noise(transformer, z, t5_0, clip_0, txt_ids, img_ids, start_guidance, t_enc_ddpm, device, weight_dtype)
        e_p = predict_noise(transformer, z, t5_p, clip_p, txt_ids, img_ids, start_guidance, t_enc_ddpm, device, weight_dtype)
    
    # get conditional score from ESD model
    e_n = predict_noise(transformer, z, t5_p, clip_p, txt_ids, img_ids, start_guidance, t_enc_ddpm, device, weight_dtype)
    e_0.requires_grad = False
    e_p.requires_grad = False
    
    total_loss = []
    
    # Calculate ESD loss
    loss_esd = criteria(e_n.to(device), e_0.to(device) - (negative_guidance*(e_p.to(device) - e_0.to(device))))
    
    total_loss.append(loss_esd)
    
    # Calculate Attn Loss    
    # add noise and get attention map
    noise = torch.randn_like(latents_packed)
    bsz = latents_packed.shape[0]
    
    # add noise with scheduler
    noisy_model_input = noise_scheduler.add_noise(latents_packed, noise, t_enc_ddpm)
    
    # prepare condition
    cond = torch.zeros(batch_size, noisy_model_input.shape[1], 68, device=device, dtype=weight_dtype)
    
    # get remove_indices
    remove_indices = batch['remove_indices'][0]
    
    # forward pass to get attention map
    model_pred, attn_maps = transformer(
        img=noisy_model_input.to(dtype=weight_dtype, device=device),
        img_ids=img_ids.to(dtype=weight_dtype, device=device),
        txt=t5_p.to(dtype=weight_dtype, device=device),
        txt_ids=txt_ids.to(dtype=weight_dtype, device=device),
        timesteps=(t_enc_ddpm / 1000).to(weight_dtype),
        y_vec=clip_p.to(dtype=weight_dtype, device=device),
        guidance=start_guidance.to(dtype=weight_dtype, device=device),
        cond=cond,
        return_attn_weights=True,
    )
    
    # calculate attention loss
    attn_map_mask = torch.ones_like(attn_maps).to(device)
    attn_map_mask[..., remove_indices] = 0
    attn_map_mask = 1 - attn_map_mask
        
    # Compute regular loss.
    loss_attn = sum(torch.norm(attn_map_mask*attn_maps, dim=(0, 1))).sum()
    
    total_loss.append(loss_attn)
    
    return total_loss, t_enc_ddpm


def calculate_lower_loss(
    transformer,
    vae,
    model_t5,
    model_clip,
    batch,
    noise_scheduler,
    tokenizer_t5,
    criteria,
    weight_dtype,
    device,
    t_enc,
    ir_concept_lst,
    args,
):
    """
    2025.10.23 by jnx    
    Returns:
        total_loss: [loss_lora, loss_infonce]
    """
    
    videos = batch["videos"]
    prompts = batch["prompts"]
    batch_size = videos.shape[0]
    
    # Convert videos to latent space
    latents = encode_video(vae, videos, weight_dtype)  # (bs, c, t, h, w)
    model_input = opensora_pack_latents(latents, patch_size=2)  # (bs, seq_len, c)
    
    # Sample noise that we'll add to the latents
    noise = torch.randn_like(model_input)
    bsz = model_input.shape[0]
    
    # Encode text
    t5_p, clip_p = encode_text(model_t5, model_clip, prompts, device, weight_dtype)
    
    # Create position IDs
    img_ids, txt_ids = prepare_position_ids(latents, t5_p, device, weight_dtype)
    
    # Add noise with scheduler
    ddim_steps = getattr(args, 'ddim_steps', 28)
    noisy_model_input = noise_scheduler.add_noise(model_input, noise, t_enc)
    
    # Prepare guidance
    start_guidance_scale = float(getattr(args, 'guidance_scale', 3.0))
    guidance = torch.tensor([start_guidance_scale], device=device)
    guidance = guidance.expand(model_input.shape[0])
    
    # Prepare condition
    cond = torch.zeros(batch_size, noisy_model_input.shape[1], 68, device=device, dtype=weight_dtype)
    
    # forward pass to get model_pred and attn_maps
    model_pred, attn_maps = transformer(
        img=noisy_model_input.to(dtype=weight_dtype, device=device),
        img_ids=img_ids.to(dtype=weight_dtype, device=device),
        txt=t5_p.to(dtype=weight_dtype, device=device),
        txt_ids=txt_ids.to(dtype=weight_dtype, device=device),
        timesteps=(t_enc / 1000).to(weight_dtype),
        y_vec=clip_p.to(dtype=weight_dtype, device=device),
        guidance=guidance.to(dtype=weight_dtype, device=device),
        cond=cond,
        return_attn_weights=True,
    )
    
    # flow matching loss
    target = noise - model_input
    
    loss_lora = torch.mean(
        ((model_pred.float() - target.float()) ** 2).reshape(target.shape[0], -1),
        1,
    )[0]
    
    total_loss = []
    total_loss.append(loss_lora)
    
    # one negtive sample (synonym) + K positive sample (irrelevant)
    start_code = torch.randn_like(model_input)
    start_guidance = 3
    start_guidance = torch.tensor([start_guidance], device=device)
    start_guidance = start_guidance.expand(model_input.shape[0])
    
    # negtive sample: emb_neg
    synonym_words = batch.get("synonym_words", [batch.get("original_keyword", [""])[0]] * batch_size)
    t5_neg, clip_neg = encode_text(model_t5, model_clip, synonym_words, device, weight_dtype)
    
    # one negtive sample + K positive sample
    with torch.no_grad():
        _, _, attn_map_lst_neg = latent_sample(transformer,
                                               noise_scheduler,
                                               batch_size,
                                               start_code,
                                               t5_neg,
                                               clip_neg,
                                               img_ids,
                                               txt_ids,
                                               start_guidance,
                                               int(ddim_steps),
                                               device,
                                               weight_dtype,
                                               return_attn=True)
    
    # irrelevant sample: emb_pos
    K = len(ir_concept_lst)
    if len(ir_concept_lst) != K:
        raise Exception("Please check ir_concept_lst")
    
    # random index
    attn_map_rand_idx = random.randint(0, int(ddim_steps)-1)
    
    pos_lst = []
    for idx in range(K):
        t5_ir, clip_ir = encode_text(model_t5, model_clip, [ir_concept_lst[idx]], device, weight_dtype)
        
        # forward pass to get attention map
        _, _, attn_map_lst_pos_sub = latent_sample(transformer,
                                                   noise_scheduler,
                                                   batch_size,
                                                   start_code,
                                                   t5_ir,
                                                   clip_ir,
                                                   img_ids,
                                                   txt_ids,
                                                   start_guidance,
                                                   int(ddim_steps),
                                                   device,
                                                   weight_dtype,
                                                   return_attn=True)
        
        # get attention map
        tmp_attn_pos = attn_map_lst_pos_sub[attn_map_rand_idx]
        pos_lst.append(tmp_attn_pos)
    
    attn_map_neg = attn_map_lst_neg[attn_map_rand_idx]
    attn_map_pos = pos_lst
    
    info_neg = attn_map_neg[..., batch['remove_indices'][0]][:, 0, ...].permute(0, 2, 1)
    info_pos_lst = []
    
    for idx in range(K):
        info_pos = pos_lst[idx][..., batch['remove_indices'][0]][:, 0, ...].permute(0, 2, 1)
        info_pos_lst.append(info_pos)
    
    info_center = attn_maps[..., batch['remove_indices'][0]][:, 0, ...].permute(0, 2, 1)
    
    from .infoNCE import calculate_steer_loss
    loss_contrastive = calculate_steer_loss(info_center,
                                            info_neg,
                                            info_pos_lst,
                                            temperature=0.07)
    
    total_loss.append(loss_contrastive)
    return total_loss