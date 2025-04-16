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
        mamba_dict = get_mamba_dict()
        model = models.mamba.Mamba(mamba_dict)
    elif type == "xlstm":
        xlstm_dict = get_xlstm_dict()
        model = models.xlstm.xLSTM(xlstm_dict)
    elif type == "transformer":
        transformer_dict = get_transformer_dict()
        model = models.transformer.Transformer(transformer_dict)
    return model

def load_model(type, name):
    model = new_model(type)
    model.load_state_dict(torch.load(f'pretrained/{type}/{name}'))
    return model

def save_model(model, loss):
    now = datetime.now().strftime("%Y-%m-%d-%H-%M-%S")

    save_path = f'pretrained/{args.model}/loss_{loss:.2f}_time_{now}.pth'

    if not os.path.exists(os.path.dirname(save_path)):
        os.makedirs(os.path.dirname(save_path))
    torch.save(model.state_dict(), save_path)

def train(model):
    model.to(cc.config.values.device)
    loader = processing.DatasetLoader('..\\dataset\\np_dataset')
    train_dataloader, test_dataloader = loader.get_dataloaders()
    criterion = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=cc.config.values.learning_rate)

    # Training loop
    num_epochs = cc.config.values.epochs
    print('Training started!')
    for epoch in range(num_epochs):
        model.train()  # Set the model to training mode
        total_loss = 0

        for batch_idx, (src, trg, metadata) in enumerate(train_dataloader):
            output = model(src, metadata)
            output = output.reshape(-1, model.vocab_size)  # Flatten the output to [batch_size * seq_len, vocab_size]
            trg = trg.view(-1)  # Flatten the target to [batch_size * seq_len]

            loss = criterion(output, trg)
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()

            if (batch_idx + 1) % 100 == 0:
                print(f'Epoch [{epoch+1}/{num_epochs}], Step [{batch_idx+1}/{len(train_dataloader)}], Loss: {loss.item():.4f}')

        avg_loss = total_loss / len(train_dataloader)
        print(f'Epoch [{epoch+1}/{num_epochs}], Average Loss: {avg_loss:.4f}')

        model.eval()  # Set the model to evaluation mode
        val_loss = 0
        with torch.no_grad():
            for src, trg, metadata in test_dataloader:
                output = model(src, metadata)
                output = output.reshape(-1, model.vocab_size)
                trg = trg.view(-1)
                val_loss += criterion(output, trg).item()
        
        avg_val_loss = val_loss / len(test_dataloader)
        print(f'Epoch [{epoch+1}/{num_epochs}], Validation Loss: {avg_val_loss:.4f}')
        if (epoch + 1) % cc.config.values.save_interval == 0:
            save_model(model, avg_val_loss)

    print("Training complete!")
    save_model(model, avg_val_loss)

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

    train(model)