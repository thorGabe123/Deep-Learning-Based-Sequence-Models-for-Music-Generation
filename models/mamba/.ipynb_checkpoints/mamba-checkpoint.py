import torch.nn as nn
import configs.common as cc
import torch
from mamba_ssm import Mamba2
from mamba_ssm.utils.generation import InferenceParams


class Mamba(nn.Module):
    def __init__(self, d_model=1024, n_layers=10):
        super().__init__()

        self.token_embedding = nn.Embedding(cc.vocab_size, d_model)
        self.metadata_embedding = nn.Embedding(cc.metadata_vocab_size, d_model)
        self.output_layer = nn.Linear(d_model, cc.vocab_size)

        self.layers = nn.ModuleList([
            Mamba2(
                d_model=d_model, 
                d_state=64,  # SSM state expansion factor, typically 64 or 128
                d_conv=4,    # Local convolution width
                expand=2,    # Block expansion factor
                layer_idx=i  # Required for recurrent generation
            ).to("cuda") for i in range(n_layers)
        ])
        self.norm = nn.LayerNorm(d_model)

    def forward(self, tokens, meta):
        
        x = self.token_embedding(tokens)
        x = torch.cat((self.metadata_embedding(meta), x), dim=-2)

        for layer in self.layers:
            x = layer(x)
            # x = layer(x, inference_params=inference_params)
        x = self.norm(x)
        return self.output_layer(x)[:,6:]