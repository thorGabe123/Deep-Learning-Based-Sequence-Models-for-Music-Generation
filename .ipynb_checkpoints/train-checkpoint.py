import argparse
import torch
import configs.mamba as cm
import configs.xlstm as cx
import configs.transformer as ct
import configs.common as cc
import models
import math
import processing
from types import SimpleNamespace
import os
from datetime import datetime
import configs.paths as paths
import torch.nn.functional as F
import torch
import json

length_tensor = torch.linspace(1, 3, steps=499).to("cuda")

def get_actual_vocab_size(type):
    config = cm.config.model_values
    new_vocab_size = config.vocab_size
    if type == 'mamba':
        if config.vocab_size % config.pad_vocab_size_multiple != 0:
            new_vocab_size += (config.pad_vocab_size_multiple - config.vocab_size % config.pad_vocab_size_multiple)
    return new_vocab_size

def get_mamba_dict():
    config = cm.config.model_values
    config.vocab_size = cc.vocab_size
    config.d_inner = int(config.expand * config.d_model)
    config.dt_rank = math.ceil(config.d_model / 16)
    config.vocab_size = get_actual_vocab_size('mamba')
    config.metadata_vocab_size = cc.metadata_vocab_size
    config = SimpleNamespace(**vars(config), **vars(cc.config.values))
    return config

def get_xlstm_dict():
    config = cx.config.model_values
    config.vocab_size = cc.vocab_size
    config.metadata_vocab_size = cc.metadata_vocab_size
    config = SimpleNamespace(**vars(config), **vars(cc.config.values))
    return config

def get_transformer_dict():
    config = ct.config.model_values
    config.vocab_size = cc.vocab_size
    config.metadata_vocab_size = cc.metadata_vocab_size
    config = SimpleNamespace(**vars(config), **vars(cc.config.values))
    return config

def new_model(type):
    if type == "mamba":
        model = models.mamba.Mamba()
    elif type == "xlstm":
        from models.xlstm import xLSTM
        model = xLSTM()
    elif type == "transformer":
        transformer_dict = get_transformer_dict()
        model = models.transformer.Transformer(transformer_dict)
    return model

def load_model(type, name):
    model = new_model(type)
    pretrained = paths.config.paths.pretrained
    model.load_state_dict(torch.load(f'{pretrained}/{type}/{name}'))
    return model

def save_model(model, loss):
    now = datetime.now().strftime("%Y-%m-%d-%H-%M-%S")
    pretrained_path = paths.config.paths.pretrained

    save_path = f'{pretrained_path}/{args.model}/loss_{loss:.2f}_time_{now}.pth'

    if not os.path.exists(os.path.dirname(save_path)):
        os.makedirs(os.path.dirname(save_path))
    torch.save(model.state_dict(), save_path)

def make_distributions():
    # Prepare output tensor
    block_len = cc.config.values.block_len
    device = cc.config.values.device
    vocab_size = cc.vocab_size
    distributions = torch.zeros(5, vocab_size, device=device)

    start = [cc.start_idx["pitch"],
             cc.start_idx["dyn"],
             cc.start_idx["length"],
             cc.start_idx["time"],
             cc.start_idx["tempo"]]
    end   = [cc.start_idx["dyn"] - 1,
             cc.start_idx["length"] - 1,
             cc.start_idx["time"] - 1,
             cc.start_idx["tempo"] - 1,
             cc.vocab_size]   # block_len implies : to the end

    for token in range(5):
        distributions[token - 1, start[token]:end[token]] = 1
    distributions[2, start[4]:end[4]] = 1

    if length_tensor is not None:
        length_start = cc.start_idx["length"]
        length_end = cc.start_idx["time"] - 1
        expected_length = length_end - length_start
        if length_tensor.shape[0] != expected_length:
            raise ValueError(f"length_tensor has shape {length_tensor.shape}, expected length {expected_length}")
        distributions[1, length_start:length_end] *= length_tensor

    distributions[4, cc.start_idx["pitch"]:cc.start_idx["dyn"] - 1] *= 10

    return distributions


def pick_distributions_by_prev_token(
        input_tokens: torch.Tensor
    ) -> torch.Tensor:
    boundaries = [
                cc.start_idx['dyn'] - 1,
                cc.start_idx['length'] - 1,
                cc.start_idx['time'] - 1,
                cc.start_idx['tempo'] - 1]
        
    bins = torch.tensor(boundaries, device=input_tokens.device)
        # Each token is assigned to a bucket 0..len(boundaries)
        # For example, with bins=[10,20,30]: 
        #   token <10 → 0, 10<=token<20 → 1, 20<=token<30 → 2, 30<=token → 3
    buckets = torch.bucketize(input_tokens, bins, right=False)
    distributions = make_distributions()

    # Ensure buckets is long (int64) and on the same device as distributions
    buckets = buckets.long().to(distributions.device)

    # Now use advanced indexing to get values
    output = distributions[buckets]  # shape: [6, 963]

    return output

def filtered_logit(input, output):
    weights = pick_distributions_by_prev_token(input)
    temperature = 1.5
    log_probs = F.log_softmax(output, dim=1)
    loss = -log_probs * weights
    return loss

def train(model, type_name):
    model.to(cc.config.values.device)
    dataset_path = paths.config.paths.np_dataset
    loader = processing.DatasetLoader(dataset_path)
    train_dataloader, test_dataloader = loader.get_dataloaders()
    criterion = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=cc.config.values.learning_rate)

    # Logging setup
    log_data = []
    log_file_path = f'/scratch/s203861/logs/training_log_{type_name}.json'

    # Training loop
    num_epochs = cc.config.values.epochs
    print('Training started!')
    log_data.append({'timestamp': str(datetime.now()), 'message': 'Training started!'})
    try:
        for epoch in range(num_epochs):
            model.train()
            total_loss = 0
    
            for batch_idx, (src, trg, meta) in enumerate(train_dataloader):
                output = model(src, meta)
                filtered_output = filtered_logit(src, output)
                filtered_output = filtered_output.reshape(-1, cc.vocab_size)
                trg = trg.view(-1)
    
                loss = criterion(filtered_output, trg)
    
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
    
                total_loss += loss.item()
    
                if (batch_idx + 1) % cc.config.values.eval_interval == 0:
                    msg = f'{loss.item():.4f}'
                    log_data.append({'Step': len(train_dataloader) * epoch + batch_idx + 1, 'Loss': msg})
                    print(f'Step: {len(train_dataloader)*epoch+batch_idx+1}, Loss: {msg}')
    
            avg_loss = total_loss / len(train_dataloader)
            msg = f'Epoch [{epoch+1}/{num_epochs}], Average Loss: {avg_loss:.4f}'
            print(msg)
    
            model.eval()
            val_loss = 0
            with torch.no_grad():
                for src, trg, meta in test_dataloader:
                    output = model(src, meta)
                    filtered_output = filtered_logit(src, output)
                    filtered_output = filtered_output.reshape(-1, cc.vocab_size)
                    trg = trg.view(-1)
                    val_loss += criterion(filtered_output, trg)
    
            avg_val_loss = val_loss / len(test_dataloader)
            msg = f'Epoch [{epoch+1}/{num_epochs}], Validation Loss: {avg_val_loss:.4f}'
            print(msg)
            log_data.append({'timestamp': str(datetime.now()), 'message': msg})
    
            if (epoch + 1) % cc.config.values.save_interval == 0:
                save_model(model, avg_val_loss)
                with open(log_file_path, 'w') as f:
                    json.dump(log_data, f, indent=2)
    except KeyboardInterrupt:
        print("Interrupted!")
    finally:
        print("Saving model before exit...")
        save_model(model, avg_val_loss if avg_val_loss is not None else 0.0)
        # Save logs 
        with open(log_file_path, 'w') as f:
            json.dump(log_data, f, indent=2)
        print("Model saved. Exiting.")

    print("Training complete!")
    log_data.append({'timestamp': str(datetime.now()), 'message': 'Training complete!'})

    save_model(model, avg_val_loss)

    # Final log save
    with open(log_file_path, 'w') as f:
        json.dump(log_data, f, indent=2)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Training Script")

    # Add command-line arguments
    parser.add_argument("--model",
                    type=str,
                    default="mamba",
                    choices=["mamba", "xlstm", "transformer"],  # List of allowed choices
                    help="Model name (choices: mamba, xlstm, transformer)")
    parser.add_argument("--name", type=str, help="Name of the model to train, e.g. mamba1000.pth")

    # Parse arguments
    args = parser.parse_args()
    
    if args.name is None:
        model = new_model(args.model)
    else:
        model = load_model(args.model, args.name)
    model.to(cc.config.values.device)

    train(model, args.model)
