import argparse
import torch
import torch.nn as nn
import configs.common as cc
import torch.nn.functional as F
import processing
import configs.paths as paths
from datetime import datetime
import os
import json
from models.classifier.model import Classifier
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data.distributed import DistributedSampler

def get_set(tensor):
    return [torch.unique(row) for row in tensor]

def make_meta_target(tensor):
    target = torch.zeros(cc.metadata_vocab_size)
    target[tensor] = 1
    return target

def get_all_targets(tensor):
    meta_unique = get_set(tensor)
    return torch.stack([make_meta_target(m) for m in meta_unique])

def save_model(model, loss):
    now = datetime.now().strftime("%Y-%m-%d-%H-%M-%S")
    pretrained_path = paths.config.paths.pretrained

    save_path = f'{pretrained_path}/classifier/loss_{loss:.2f}_time_{now}.pth'

    if not os.path.exists(os.path.dirname(save_path)):
        os.makedirs(os.path.dirname(save_path))
    torch.save(model.state_dict(), save_path)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Training Script")

    # Add command-line arguments
    parser.add_argument("--name", type=str)
    args = parser.parse_args()
    
    torch.cuda.set_device(int(os.environ["LOCAL_RANK"]))
    dist.init_process_group("nccl")
    rank = dist.get_rank()
    print(f"Start running basic DDP example on rank {rank}.")
    # create model and move it to GPU with id rank
    device_id = rank % torch.cuda.device_count()
    
    with open('/scratch/s203861/data/data_distribution.json', 'r') as f:
        data = json.load(f)
    test_dataset = processing.DatasetLoader(data["test"])
    train_dataset = processing.DatasetLoader(data["train"])
    test_dataloader = test_dataset.get_dataloader_full()
    train_dataloader = train_dataset.get_dataloader_full()

    model = Classifier().to("cuda")
    if args.name:
        pretrained = paths.config.paths.pretrained
        model.load_state_dict(torch.load(f'{pretrained}/classifier/{args.name}'))
    model.to(device_id)
    ddp_model = DDP(model, device_ids=[device_id])

    criterion = torch.nn.BCEWithLogitsLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=cc.config.values.learning_rate)
    num_classes = cc.metadata_vocab_size  # Set to your number of possible labels

    # Logging setup
    log_data = []
    log_file_path = f'/scratch/s203861/logs/training_log_classifier.json'

    # Training loop
    num_epochs = cc.config.values.epochs
    print('Training started!')
    log_data.append({'timestamp': str(datetime.now()), 'message': 'Training started!'})

    try:
        for epoch in range(num_epochs):
            model.train()
            total_loss = 0
    
            for batch_idx, (src, trg, meta) in enumerate(train_dataloader):
                output = model(src).to('cuda')
                trg = get_all_targets(meta).to('cuda')
    
                loss = criterion(output, trg)
    
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
    
                total_loss += loss.item()
    
                if (batch_idx + 1) % cc.config.values.eval_interval == 0 and dist.get_rank() == 0:
                    msg = f'Epoch [{epoch+1}/{num_epochs}], Step [{batch_idx+1}/{len(train_dataloader)}], Loss: {loss.item():.4f}'
                    print(msg)
                    log_data.append({'timestamp': str(datetime.now()), 'message': msg})
    
            avg_loss = total_loss / len(train_dataloader)
            if dist.get_rank() == 0:
                msg = f'Epoch [{epoch+1}/{num_epochs}], Average Loss: {avg_loss:.4f}'
                print(msg)
            log_data.append({'timestamp': str(datetime.now()), 'message': msg})
    
            model.eval()
            val_loss = 0
            with torch.no_grad():
                for src, trg, meta in test_dataloader:
                    output = model(src).to('cuda')
                    trg = get_all_targets(meta).to('cuda')
                    val_loss += criterion(output, trg).item()
    
            avg_val_loss = val_loss / len(test_dataloader)
            if dist.get_rank() == 0:
                msg = f'Epoch [{epoch+1}/{num_epochs}], Validation Loss: {avg_val_loss:.4f}'
                print(msg)
                log_data.append({'timestamp': str(datetime.now()), 'message': msg})
    
            if (epoch + 1) % cc.config.values.save_interval == 0 and dist.get_rank() == 0:
                save_model(model, avg_val_loss)
                with open(log_file_path, 'w') as f:
                    json.dump(log_data, f, indent=2)

    except KeyboardInterrupt:
        print("Interrupted!")
    finally:
        if dist.get_rank() == 0:
            print("Saving model before exit...")
            save_model(model, avg_val_loss if avg_val_loss is not None else 0.0)
            # Save logs 
            with open(log_file_path, 'w') as f:
                json.dump(log_data, f, indent=2)
            print("Model saved. Exiting.")

    if dist.get_rank() == 0:
        print("Training complete!")
        log_data.append({'timestamp': str(datetime.now()), 'message': 'Training complete!'})
    
        save_model(model, avg_val_loss)
    
        # Final log save
        with open(log_file_path, 'w') as f:
            json.dump(log_data, f, indent=2)
