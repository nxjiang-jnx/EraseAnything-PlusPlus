import torch
from typing import Any, Callable, Dict, List, Optional, Union
from opensora.utils.sampling import pack, unpack


def opensora_pack_latents(latents, patch_size=2):
    return pack(latents, patch_size=patch_size)


def opensora_unpack_latents(latents_packed, height, width, num_frames, patch_size=2):
    return unpack(latents_packed, height, width, num_frames, patch_size=patch_size)


def prepare_position_ids(latents_unpacked, t5_embedding, device, dtype):
    bs, c, t, h, w = latents_unpacked.shape
    txt_seq_len = t5_embedding.shape[1]
    
    # Image position IDs
    img_ids = torch.zeros(bs, t * h * w // 4, 3, device=device, dtype=dtype)  # packed size
    idx = 0
    for ti in range(t):
        for hi in range(0, h, 2):  # patch_size=2
            for wi in range(0, w, 2):
                img_ids[:, idx, 0] = ti
                img_ids[:, idx, 1] = hi // 2
                img_ids[:, idx, 2] = wi // 2
                idx += 1
    
    # Text position IDs
    txt_ids = torch.zeros(bs, txt_seq_len, 3, device=device, dtype=dtype)
    for i in range(txt_seq_len):
        txt_ids[:, i, 0] = 0
        txt_ids[:, i, 1] = i
        txt_ids[:, i, 2] = 0
    
    return img_ids, txt_ids


@torch.no_grad()
def latent_sample(
    transformer,
    scheduler,
    batch_size,
    latents_packed,
    t5_emb,
    clip_emb,
    img_ids,
    txt_ids,
    guidance,
    ddim_steps,
    device,
    weight_dtype,
    return_attn=False
):
    scheduler.set_train_timesteps(ddim_steps, device=device)
    timesteps = scheduler.timesteps
    
    latents = latents_packed.clone().to(device, dtype=weight_dtype)
    t5_emb = t5_emb.to(device, dtype=weight_dtype)
    clip_emb = clip_emb.to(device, dtype=weight_dtype)
    txt_ids = txt_ids.to(device, dtype=weight_dtype)
    img_ids = img_ids.to(device, dtype=weight_dtype)
    
    attn_map_lst = []
    
    for i, t in enumerate(timesteps):
        # broadcast to batch dimension in a way that's compatible with ONNX/Core ML
        timestep = t.expand(latents.shape[0]).to(weight_dtype)
        
        # prepare zero condition for T2V
        cond = torch.zeros(batch_size, latents.shape[1], 68, device=device, dtype=weight_dtype)
        
        # forward pass
        output = transformer(
            img=latents,
            img_ids=img_ids,
            txt=t5_emb,
            txt_ids=txt_ids,
            timesteps=timestep.to(weight_dtype),
            y_vec=clip_emb,
            guidance=guidance,
            cond=cond,
            return_attn_weights=return_attn,
        )
        
        if return_attn:
            if isinstance(output, tuple) and len(output) >= 2:
                noise_pred, attn_maps = output[0], output[1]
            else:
                noise_pred = output
                attn_maps = torch.zeros_like(latents)
        else:
            noise_pred = output
            attn_maps = None
        
        # compute previous noise sample x_t -> x_{t-1}
        latents = scheduler.step(noise_pred, t, latents, return_dict=False)[0]
        
        # save attention map
        if return_attn:
            attn_map_lst.append(attn_maps)
    
    if return_attn:
        return latents, img_ids, attn_map_lst
    else:
        return latents, img_ids


def predict_noise(
    transformer,
    latent_code,
    t5_emb,
    clip_emb,
    txt_ids,
    img_ids,
    guidance,
    timesteps,
    device,
    weight_dtype
):
    batch_size = latent_code.shape[0]
    
    cond = torch.zeros(batch_size, latent_code.shape[1], 68, device=device, dtype=weight_dtype)
    
    # forward pass
    model_pred = transformer(
        img=latent_code.to(device),
        img_ids=img_ids.to(device),
        txt=t5_emb.to(device),
        txt_ids=txt_ids.to(device),
        timesteps=timesteps.to(device, dtype=weight_dtype),
        y_vec=clip_emb.to(device),
        guidance=guidance.to(device),
        cond=cond,
        return_attn_weights=False,
    )
    
    if isinstance(model_pred, tuple):
        model_pred = model_pred[0]
    
    return model_pred

