# coding: UTF-8
"""
    @date:    2025.10.23
    @author:  jnx (adapted from EraseAnything)
    @ref:     https://github.com/ziqihuangg/ReVersion/blob/master/train.py#L537
    @func:    InfoNCE loss for concept erasure
"""

import os
import numpy as np

import torch
import torch.nn.functional as F

def calculate_steer_loss(unlearn_attn,
                         neg_attn,
                         irr_attn_lst,
                         temperature=0.07,
                         method="mean"):
    """
    Calculate InfoNCE loss (L_steer) for concept erasure
    """
    # unlearn_attn   [bs, length, 1280] (nude)                       requires_grad = False
    # neg_attn       [bs, length, 1280] (同义词: naked, nudity, ...) requires_grad = False
    # irr_attn_lst   [[bs, length1, 1280], [bs, length2, 1280], ...],  requires_grad = True
    
    if method == "mean":
        unlearn_attn = unlearn_attn.mean(1).unsqueeze(1)  # [bs, 1, dim]
        neg_attn = neg_attn.mean(1).unsqueeze(1)  # [bs, 1, dim]
        
        # Concatenate irrelevant attentions
        irr_attn = None
        for item in irr_attn_lst:
            if irr_attn is None:
                irr_attn = item.mean(1).unsqueeze(1)
            else:
                irr_attn = torch.cat([irr_attn, item.mean(1).unsqueeze(1)], dim=1)
        # irr_attn: [bs, N, dim] where N is number of irrelevant concepts
        
    # Stack positives(unlearn) and negatives(irrelevant) as a pn_block
    pn_embeds = torch.cat([unlearn_attn, irr_attn], dim=1)  # [bs, 1+N, dim]
    pn_embeds_normalized = F.normalize(pn_embeds, p=2, dim=2)
    
    # Compute malicious embeds (synonym)
    neg_attn_normalized = F.normalize(neg_attn, p=2, dim=2)  # [bs, 1, dim]
    
    # Compute Multi-Instance InfoNCE loss
    # Similarity between neg_attn and all concepts (unlearn + irrelevant)
    logits = torch.einsum('bnc,bmc->bnm',
                          [neg_attn_normalized, pn_embeds_normalized])  # [bs, 1, 1+N]
    
    # Scale by temperature
    logits /= temperature
    
    # InfoNCE: log(exp(sim(neg, unlearn)) / sum(exp(sim(neg, all))))
    # Nominator: similarity with target concept (to be maximized)
    nominator = torch.logsumexp(logits[:, :, :neg_attn.shape[1]], dim=(1, 2))
    
    # Denominator: similarity with all concepts (unlearn + irrelevant)
    denominator = torch.logsumexp(logits, dim=(1, 2))
    
    # Loss: maximize (nominator - denominator)
    # Equivalent to: minimize (denominator - nominator)
    # We want to minimize this loss, so return (nominator - denominator)
    return torch.mean(nominator - denominator)


if __name__ == "__main__":
    unlearn_attn = torch.randn(1, 2, 1280)
    neg_attn = torch.randn(1, 3, 1280)
    # K = ?
    irr_attn_lst = [torch.randn(1, 3, 1280), torch.randn(1, 4, 1280), torch.randn(1, 1, 1280)] # [bs, 1280, 2]
    aaa = calculate_steer_loss(unlearn_attn,
                         neg_attn,
                         irr_attn_lst)
    import pdb; pdb.set_trace()