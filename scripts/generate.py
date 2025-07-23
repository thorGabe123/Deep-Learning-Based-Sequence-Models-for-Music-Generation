import sys
sys.path.append('..')
import processing
import configs.common as cc
import train
from train import new_model
import torch
import configs.paths as paths
import argparse
import os
from collections import Counter

def generate(model, context_len, token_ids, meta_ids, num_tokens=1000, device='cuda'):
    model.eval()
    batch_size = token_ids.size(0)
    generated = token_ids.detach().cpu().tolist()  # list of lists, len=batch_size

    token_ids = token_ids.to(device)
    meta_ids = meta_ids.to(device)

    for _ in range(num_tokens):
        logits = model(token_ids, meta_ids)    # (batch, seq_len, vocab_size)
        filtered_logits = train.filtered_logit(token_ids, logits)
        logits_last = filtered_logits[:, -1, :]   # (batch, vocab_size)

        # Repeat penalty: apply per-sample
        for i in range(batch_size):
            recent = generated[i][-200:]
            counts = Counter(recent)
            for token, count in counts.items():
                if cc.start_idx['tempo'] <= token:
                    continue
                elif cc.start_idx['time'] <= token:
                    continue
                elif cc.start_idx['length'] <= token:
                    penalty = min(1.01 ** count, 1.2)
                elif cc.start_idx['dyn'] <= token:
                    penalty = min(1.03 ** count, 1.2)
                else:
                    penalty = min(1.01 ** count, 1.1)
                if count > 0:
                    logits_last[i, token] /= penalty

        topk = torch.tensor([1] * batch_size)
        topk_probs, topk_indices = torch.topk(logits_last, 1, dim=-1)

        topk_probs = topk_probs / topk_probs.sum(dim=-1, keepdim=True)
        next_token_idx = torch.multinomial(topk_probs, num_samples=1)  # (batch, 1)
        next_token = topk_indices.gather(1, next_token_idx)            # (batch, 1)

        for i in range(batch_size):
            generated[i].append(next_token[i,0].item())

        token_ids = torch.cat([token_ids, next_token], dim=1)

        if token_ids.size(1) > context_len:
            token_ids = token_ids[:, -context_len:]

    return generated