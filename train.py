import argparse
import torch
import configs.mamba as cm
import configs.xlstm as cx
import configs.transformer as ct
import configs.common as cc
import models

cm.config_dict
cc.config_dict

def train(type, name):
    if type == "mamba":
        model = models.mamba.Mamba(nn.Module)
        pass
    elif type == "xlstm":
        model = models.xlstm.xLSTM(nn.Module)
        pass
    elif type == "transformer":
        model = models.transformer.Transformer(nn.Module)
        pass
    loaded_model = torch.load(f'models/{model}/{name}.pth')
    model.load_state_dict(state_dict)




if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Training Script")

    # Add command-line arguments
    parser.add_argument("--model",
                    type=str,
                    default="mamba",
                    choices=["mamba", "xlstm", "transformer"],  # List of allowed choices
                    help="Model name (choices: mamba, xlstm, transformer)")
    parser.add_argument("--name", type=str, help="Name of the model to train, e.g. mamba1000.pth")
    # parser.add_argument("--batch_size", type=int, default=32, help="Batch size")
    # parser.add_argument("--learning_rate", type=float, default=0.001, help="Learning rate")

    # Parse arguments
    args = parser.parse_args()

    # Call training function
    train(args.model, args.name)


