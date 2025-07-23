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
from generate import generate


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generation")
    parser.add_argument("--length", type=int, help="Number of generated tokens")
    parser.add_argument("--xlstm", type=bool, default=False)
    parser.add_argument("--mamba", type=bool, default=False)
    parser.add_argument("--transformer", type=bool, default=False)
    parser.add_argument("--retain", type=bool, default=False)
    parser.add_argument("--reverse", type=bool, default=False)
    parser.add_argument("--no_metadata", type=bool, default=False)
    parser.add_argument("--data_root", type=str, default="/home/s203861/midi-classical-music/np_data/data")
    parser.add_argument("--output_path", type=str, default="/scratch/s203861/output")
    parser.add_argument("--combined_path", type=bool, default=False)
    args = parser.parse_args()

    data_root = args.data_root
    output_path = args.output_path
    band_folders = [d for d in os.listdir(data_root) if os.path.isdir(os.path.join(data_root, d))]
    band_folders = sorted(band_folders)
    if args.reverse:
        band_folders = sorted(band_folders, reverse=True)

    pretrained_path = paths.config.paths.pretrained
    cc.config.values.block_len = 1024
    if args.mamba:
        mamba_model = new_model("mamba")
        mamba_model.load_state_dict(torch.load(cc.config.models.mamba))
        mamba_model.to('cuda')
        mamba_model.eval()
    if args.xlstm:
        xlstm_model = new_model("xlstm")
        xlstm_model.load_state_dict(torch.load(cc.config.models.xlstm))
        xlstm_model.to('cuda')
        xlstm_model.eval()
    if args.transformer:
        transformer_model = new_model("transformer")
        transformer_model.load_state_dict(torch.load(cc.config.models.transformer))
        transformer_model.to('cuda')
        transformer_model.eval()
    cc.config.values.block_len = 2048
    for band in band_folders:
        model_folder = os.path.join(data_root, band)
        # Count only files (not directories) in the folder
        num_files = sum([os.path.isfile(os.path.join(model_folder, f)) for f in os.listdir(model_folder)])
        if num_files < 2:
            print(f"Skipping {band} (not enough files: {num_files})")
            continue

        if args.no_metadata:
            mamba_output_dir = os.path.join(f"{output_path}/mamba_no_meta", band)
            xlstm_output_dir = os.path.join(f"{output_path}/xlstm_no_meta", band)
            transformer_output_dir = os.path.join(f"{output_path}/transformer_no_meta", band)
        else:
            mamba_output_dir = os.path.join(f"{output_path}/mamba", band)
            xlstm_output_dir = os.path.join(f"{output_path}/xlstm", band)
            transformer_output_dir = os.path.join(f"{output_path}/transformer", band)
            combined_output_dir = os.path.join(f"{output_path}/combined", band)
        if not args.combined_path:
            os.makedirs(mamba_output_dir, exist_ok=True)
            os.makedirs(xlstm_output_dir, exist_ok=True)
            os.makedirs(transformer_output_dir, exist_ok=True)
    
        cc.config.values.block_len = 2048
        loader = processing.DatasetLoader(os.path.join(data_root, band))
        full_dataloader = loader.get_dataloader_full()
        
        for src, trg, meta in full_dataloader:
            break
        B, _ = src.shape
        cc.config.values.block_len = 1024
        # Check if outputs already exist for this band and skip if so
        if args.combined_path:
            generated_file = os.path.join(combined_output_dir, f"generated_{band}_mamba_{B - 1}.mid")
        elif args.mamba:
            generated_file = os.path.join(mamba_output_dir, f"generated_{band}_mamba_{B - 1}.mid")
        elif args.xlstm:
            generated_file = os.path.join(xlstm_output_dir, f"generated_{band}_xlstm_{B - 1}.mid")
        elif args.transformer:
            generated_file = os.path.join(transformer_output_dir, f"generated_{band}_transformer_{B - 1}.mid")
        if os.path.exists(generated_file):
            print(f"Skipping {band} (already has output)")
            continue
        print(os.path.join(data_root, band))
        print(f"Processing band: {band}")
        if args.no_metadata:
            meta = torch.zeros_like(meta, device="cuda")
        if args.mamba:
            mamba_seq = generate(
                mamba_model, cc.config.values.block_len,
                src[:,:cc.config.values.block_len], meta,
                num_tokens=args.length, device='cuda'
            )
        if args.xlstm:
            xlstm_seq = generate(
                xlstm_model, cc.config.values.block_len,
                src[:,:cc.config.values.block_len], meta,
                num_tokens=args.length, device='cuda'
            )
        if args.transformer:
            transformer_seq = generate(
                transformer_model, cc.config.values.block_len,
                src[:,:cc.config.values.block_len], meta,
                num_tokens=args.length, device='cuda'
            )
        for i in range(B):
            if args.retain:
                if args.mamba:
                    decoded_notes_mamba = processing.decode(mamba_seq[i])
                if args.xlstm:
                    decoded_notes_xlstm = processing.decode(xlstm_seq[i])
                if args.transformer:
                    decoded_notes_transformer = processing.decode(transformer_seq[i])
                if args.combined_path:
                    decoded_notes_data = processing.decode(src[i])
            else:
                if args.mamba:
                    decoded_notes_mamba = processing.decode(mamba_seq[i][800:])
                if args.xlstm:
                    decoded_notes_xlstm = processing.decode(xlstm_seq[i][800:])
                if args.transformer:
                    decoded_notes_transformer = processing.decode(transformer_seq[i][800:])
                if args.combined_path:
                    decoded_notes_data = processing.decode(src[i][800:])
            if args.combined_path:
                os.makedirs(combined_output_dir, exist_ok=True)
                processing.note_to_midi(
                    decoded_notes_data,
                    os.path.join(combined_output_dir, f"generated_{band}_data_{i}.mid")
                )
                processing.note_to_midi(
                    decoded_notes_mamba,
                    os.path.join(combined_output_dir, f"generated_{band}_mamba_{i}.mid")
                )
                processing.note_to_midi(
                    decoded_notes_xlstm,
                    os.path.join(combined_output_dir, f"generated_{band}_xlstm_{i}.mid")
                )
                processing.note_to_midi(
                    decoded_notes_transformer,
                    os.path.join(combined_output_dir, f"generated_{band}_transformer_{i}.mid")
                )
            else:
                if args.mamba:
                    processing.note_to_midi(
                        decoded_notes_mamba,
                        os.path.join(mamba_output_dir, f"generated_{band}_mamba_{i}.mid")
                    )
                if args.xlstm:
                    processing.note_to_midi(
                        decoded_notes_xlstm,
                        os.path.join(xlstm_output_dir, f"generated_{band}_xlstm_{i}.mid")
                    )
                if args.transformer:
                    processing.note_to_midi(
                        decoded_notes_transformer,
                        os.path.join(transformer_output_dir, f"generated_{band}_transformer_{i}.mid")
                    )