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
import random

def generate(
    model, context_len, token_ids, meta_ids, num_tokens=1000, device='cuda'
):
    model.eval()
    batch_size, cur_len = token_ids.size()
    # Preallocate generated for the max possible length
    generated = torch.cat([token_ids.to("cpu"), torch.zeros(batch_size, num_tokens, device="cpu", dtype=token_ids.dtype)], dim=1)
    generated_lens = [cur_len] * batch_size  # Keeps track of per-batch sequence length

    token_ids = token_ids.to(device)
    meta_ids = meta_ids.to(device)

    for step in range(num_tokens):
        if token_ids.size(1) > context_len:
            token_ids = token_ids[:, -context_len:]
        logits = model(token_ids, meta_ids)      # (batch, seq_len, vocab_size)
        filtered_logits = train.filtered_logit(token_ids, logits)
        logits_last = filtered_logits[:, -1, :]  # (batch, vocab_size)

        # Only the recent tokens matter for penalty calculation
        recent_tokens = []
        ks = torch.ones(batch_size, dtype=torch.long)
        for i in range(batch_size):
            cur_gen = generated[i, :generated_lens[i]].tolist()
            val, idx = 0, 0
            for j, token in enumerate(reversed(cur_gen)):
                if cc.start_idx['time'] <= token < cc.start_idx['tempo']:
                    val += token - cc.start_idx['time']
                if val >= 64*16:
                    break
            recent_idx = j
            recent = cur_gen[-recent_idx:]

            if cc.start_idx['tempo'] <= cur_gen[-1]:
                ks[i] = random.choice([1,1,1,2,2])
            elif cc.start_idx['time'] <= cur_gen[-1]:
                pass
            elif cc.start_idx['length'] <= cur_gen[-1]:
                pass
            elif cc.start_idx['dyn'] <= cur_gen[-1]:
                ks[i] = random.choice([1,3])
            else:
                ks[i] = random.choice([1,2])

            counts = Counter(recent)
            for token, count in counts.items():
                if cc.start_idx['tempo'] <= token:
                    continue
                elif cc.start_idx['time'] <= token:
                    continue
                elif cc.start_idx['length'] <= token:
                    continue
                elif cc.start_idx['dyn'] <= token:
                    penalty = min(1.02 ** count, 1.2)
                else:
                    penalty = min(1.01 ** count, 1.2)
                if count > 0:
                    logits_last[i, token] /= penalty

        # Torch batch mult sampling (for k=1, this is just multinomial)
        # When ks>1, you have to branch out to sample top-k: this is slower for k>1 but rare.
        next_tokens = []
        for i in range(batch_size):
            topk = ks[i]
            topk_probs, topk_indices = torch.topk(logits_last[i], topk)
            topk_probs = topk_probs / topk_probs.sum()
            next_token_idx = torch.multinomial(topk_probs, 1)
            token_val = topk_indices[next_token_idx].item()
            next_tokens.append(token_val)
            # Append token to generated
            generated[i, generated_lens[i]] = token_val
            generated_lens[i] += 1

        # Update token_ids for next step
        new_tokens_tensor = torch.tensor(next_tokens, device=device).unsqueeze(1)
        token_ids = torch.cat([token_ids, new_tokens_tensor], dim=1)

    # Slice generated tensor lists to lengths
    out = []
    for i, l in enumerate(generated_lens):
        out.append(generated[i, :l].tolist())
    return out