import torch.nn as nn
import configs.common as cc
import torch
from mamba_ssm import Mamba2

class Mamba(nn.Module):
    def __init__(self, d_model=512, n_layers=12):
        super().__init__()

        self.token_embedding = nn.Embedding(cc.vocab_size, d_model)
        self.metadata_embedding = nn.Embedding(cc.metadata_vocab_size, d_model)
        self.output_layer = nn.Linear(d_model, cc.vocab_size)

        self.layers = nn.ModuleList([
            Mamba2(
                # This module uses roughly 3 * expand * d_model^2 parameters
                d_model=d_model, # Model dimension d_model
                d_state=64,  # SSM state expansion factor, typically 64 or 128
                d_conv=4,    # Local convolution width
                expand=2,    # Block expansion factor
            ).to("cuda") for _ in range(n_layers)
        ])
        self.norm = nn.LayerNorm(d_model)

    def forward(self, tokens, meta):
        x = self.token_embedding(tokens)
        x = torch.cat((self.metadata_embedding(meta), x), dim=-2)
        for layer in self.layers:
            x = layer(x)
        x = self.norm(x)
        return self.output_layer(x)[:,6:]