import sys
sys.path.append('..')
import processing
import configs.common as cc
import train
from train import new_model
import torch
import configs.paths as paths
import argparse


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generation")
    parser.add_argument("--length", type=int, help="Number of generated tokens")
    parser.add_argument("--band", type=str, help="Band to generate from")
    parser.add_argument("--model", type=str, help="Model to generate from")
    args = parser.parse_args()

    loader = processing.DatasetLoader(f'/home/s203861/dataset/{args.band}')
    train_dataloader, test_dataloader = loader.get_dataloaders()
    for src, trg, meta in train_dataloader:
        break
    type = args.model
    if type == 'xlstm':
        model_name = 'loss_1.38_time_2025-07-02-09-19-13.pth'
    elif type == 'mamba':
        model_name = 'loss_1.37_time_2025-07-02-11-55-29.pth'
    elif type == 'transformer':
        model_name = 'loss_0.47_time_2025-07-02-11-08-21.pth'
    model = new_model(type)
    model.load_state_dict(torch.load(f'../pretrained/{type}/{model_name}'))
    model.to('cuda')

    def generate(model, context_len, token_ids, meta_ids, num_tokens=1000, device='cpu'):
        model.eval()
        generated = token_ids.detach().cpu().numpy().tolist()[0]

        with torch.no_grad():
                for _ in range(num_tokens):
                    print(_, end="\r")
                    logits = model(token_ids, meta_ids)  # (1, seq_len, vocab_size)
                    filtered_logits = train.filtered_logit(token_ids, logits)
                    logits_last = filtered_logits[:, -1, :]       # (1, vocab_size)

                    # Get top-k
                    topk_probs, topk_indices = torch.topk(logits_last, 5, dim=-1)  # Each: shape (1, 5)
                    # Normalize to sum to 1
                    topk_probs = topk_probs / topk_probs.sum(dim=-1, keepdim=True)

                    # Sample
                    next_token_idx = torch.multinomial(topk_probs, num_samples=1)  # (1, 1), value in [0..4]
                    next_token = topk_indices.gather(1, next_token_idx)

                    generated.append(next_token.item())

                    token_ids = torch.cat([token_ids, next_token], dim=1)
                    token_ids = token_ids[:, -context_len:]

                # If you want metadata, sample or set as zeros:
                # e.g., meta_ids = torch.cat([meta_ids, new_meta], dim=1)[:, -context_len:]

        return generated

    new_seq = generate(model, cc.config.values.block_len, src[1].unsqueeze(0), meta[1].unsqueeze(0), num_tokens=args.length, device='cuda')

    decoded_notes_old = processing.decode(src[1])
    processing.note_to_midi(decoded_notes_old, f"comparison_{type}.mid")

    decoded_notes_new = processing.decode(new_seq)
    processing.note_to_midi(decoded_notes_new, f"generated_{type}.mid")