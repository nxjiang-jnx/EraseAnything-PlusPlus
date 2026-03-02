# coding: UTF-8
"""
    @date: 2025.10.17
    @author: jnx
    @description: Token indexing utilities for concept erasure
    @reference: EraseAnything/utils/find_token.py
"""
import numpy as np
import torch


def search_sequence_numpy(arr, seq):
    non_zero_indices = np.where(seq != 0)
    result_indices = []
    
    for row, col in zip(*non_zero_indices):
        match_indices = np.where(arr[row] == seq[row, col])[0]
        
        if match_indices.size > 0:
            result_indices.extend([col_idx for col_idx in match_indices])
    
    return result_indices


def get_word_index(prompt, keyword, tokenizer_t5, max_length=256):
    # by jnx 2025.10.17
    prompt_encoding = tokenizer_t5(
        prompt,
        truncation=True,
        max_length=max_length,
        return_length=False,
        return_overflowing_tokens=False,
        padding="max_length",
        return_tensors="pt",  # Use PT instead of NP for consistency
    )
    prompt_text_ids = prompt_encoding["input_ids"].numpy()
    
    # Tokenize keyword with same parameters
    keyword_encoding = tokenizer_t5(
        keyword,
        truncation=True,
        max_length=max_length,
        return_length=False,
        return_overflowing_tokens=False,
        padding="max_length",
        return_tensors="pt",
    )
    word_ids = keyword_encoding["input_ids"].numpy()
    
    # Find matching indices
    idxs = search_sequence_numpy(prompt_text_ids, word_ids)
    
    # Remove last padding index
    return idxs[:-1] if len(idxs) > 0 else idxs



if __name__ == "__main__":
    from opensora.registry import MODELS, build_module
    from opensora.utils.misc import to_torch_dtype
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    weight_dtype = to_torch_dtype("bf16")
    
    # jnx: this is opensora t5 tokenizer
    t5_config = {
        "type": "text_embedder",
        # replace it with your own t5 path
        "from_pretrained": "/shared_disk/users/jeff/code/zhaoxin.fan/Open-Sora/ckpts/google/t5-v1_1-xxl",
        "max_length": 256,
        "shardformer": False,
    }
    
    model_t5 = build_module(t5_config, MODELS, device_map=device, torch_dtype=weight_dtype)
    tokenizer_t5 = model_t5.tokenizer
    
    model_t5.requires_grad_(False)
    model_t5.eval()
        
    answer = get_word_index("a nude girl with beautiful hair and her breast open to see", "breast", tokenizer_t5)
    print(answer)
    
    # prompt = "a nude girl with beautiful hair and her breast open to see"
    # keyword = "breast"
    # prompt_tokens = tokenizer_t5.tokenize(prompt)
    # keyword_tokens = tokenizer_t5.tokenize(keyword)
    # print(f"[debug] prompt tokens: {prompt_tokens}")
    # print(f"[debug] keyword tokens: {keyword_tokens}")