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

def generate(model, context_len, token_ids, meta_ids, num_tokens=1000, device='cpu'):
    model.eval()
    generated = token_ids.detach().cpu().numpy().tolist()[0]

    with torch.no_grad():
            for _ in range(num_tokens):
                print(_, end="\r")
                logits = model(token_ids, meta_ids)  # (1, seq_len, vocab_size)
                filtered_logits = train.filtered_logit(token_ids, logits)
                logits_last = filtered_logits[:, -1, :]       # (1, vocab_size)

                if len(generated) > 0:
                    recent = generated[-100:]
                    counts = Counter(recent)
                    for token, count in counts.items():
                        # Penalize by dividing by repeat_penalty ** count
                        if cc.start_idx['tempo'] <= token:
                            continue
                        elif cc.start_idx['time'] <= token:
                            if count >= 10:
                                penalty = 1.1 * count
                            else:
                                penalty = 1
                        elif cc.start_idx['length'] <= token:
                            penalty = min(1.015 ** count, 1.08)
                        elif cc.start_idx['dyn'] <= token:
                            continue
                        else:
                            penalty = min(1.04 ** count, 1.25)
                        if count > 0:
                            logits_last[0, token] /= penalty

                
                next_token = logits_last.argmax(-1).unsqueeze(0)

                generated.append(next_token.item())

                token_ids = torch.cat([token_ids, next_token], dim=1)
                token_ids = token_ids[:, -context_len:]

            # If you want metadata, sample or set as zeros:
            # e.g., meta_ids = torch.cat([meta_ids, new_meta], dim=1)[:, -context_len:]

    return generated


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generation")
    parser.add_argument("--length", type=int, help="Number of generated tokens")
    parser.add_argument("--model", type=str, help="Model to generate from")
    parser.add_argument("--name", type=str)
    parser.add_argument("--gen", type=int, default=1, help="Number of generations per model")
    args = parser.parse_args()

    data_root = "/home/s203861/midi-classical-music/np_data/beety"
    band_folders = [d for d in os.listdir(data_root) if os.path.isdir(os.path.join(data_root, d))]

    type = args.model
    model_name = args.name
    model = new_model(type)
    pretrained_path = paths.config.paths.pretrained
    model.load_state_dict(torch.load(f'{pretrained_path}/{type}/{model_name}'))
    model.to('cuda')

    for band in band_folders:
        model_folder = os.path.join(data_root, band)
        # Count only files (not directories) in the folder
        num_files = sum([os.path.isfile(os.path.join(model_folder, f)) for f in os.listdir(model_folder)])
        if num_files < 2:
            print(f"Skipping {band} (not enough files: {num_files})")
            continue
        
        output_dir = os.path.join("/scratch/s203861/output", band)
        os.makedirs(output_dir, exist_ok=True)
    
        # Check if outputs already exist for this band and skip if so
        generated_file = os.path.join(output_dir, f"generated_{band}_{type}_9.mid")
        if os.path.exists(generated_file):
            print(f"Skipping {band} (already has output)")
            continue
        print(os.path.join(data_root, band))
        print(f"Processing band: {band}")
        loader = processing.DatasetLoader(os.path.join(data_root, band))
        train_dataloader, test_dataloader = loader.get_dataloaders()
        
        for i in range(args.gen):
            for src, trg, meta in train_dataloader:
                break
            new_seq = generate(
                model, cc.config.values.block_len,
                src[0].unsqueeze(0), meta[0].unsqueeze(0),
                num_tokens=args.length, device='cuda'
            )
            # decoded_notes_new = processing.decode(new_seq[-args.length:])
            decoded_notes_new = processing.decode(new_seq)
            processing.note_to_midi(
                decoded_notes_new,
                os.path.join(output_dir, f"generated_{band}_{type}_{i}.mid")
            )
