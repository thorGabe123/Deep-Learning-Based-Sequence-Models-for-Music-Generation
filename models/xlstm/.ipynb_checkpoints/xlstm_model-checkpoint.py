import torch
import torch.nn as nn
import configs.common as cc

import xlstm

from xlstm import (
    xLSTMBlockStack,
    xLSTMBlockStackConfig,
    mLSTMBlockConfig,
    mLSTMLayerConfig,
    sLSTMBlockConfig,
    sLSTMLayerConfig,
    FeedForwardConfig,
)

cfg = xlstm.xLSTMBlockStackConfig(
    mlstm_block=mLSTMBlockConfig(
        mlstm=mLSTMLayerConfig(
            conv1d_kernel_size=4,
            qkv_proj_blocksize=4,
            num_heads=4,
        )
    ),
    slstm_block=sLSTMBlockConfig(
        slstm=sLSTMLayerConfig(
            backend="cuda",
            num_heads=4,
            conv1d_kernel_size=4,
            bias_init="powerlaw_blockdependent",
        ),
        feedforward=FeedForwardConfig(
            proj_factor=1.3,
            act_fn="gelu",
        ),
    ),
    # context_length=cc.config.values.block_len,
    context_length=cc.config.values.block_len+6,
    embedding_dim=1024,
    num_blocks=11,
    slstm_at=[1, 4, 7, 10],
)

class xLSTM(nn.Module):
    def __init__(self, d_model=1024, n_layers=12):
        super().__init__()

        self.token_embedding = nn.Embedding(cc.vocab_size, d_model)
        self.metadata_embedding = nn.Embedding(cc.metadata_vocab_size, d_model)
        self.output_layer = nn.Linear(d_model, cc.vocab_size)
        self.layers = xLSTMBlockStack(cfg)
        # self.norm = nn.LayerNorm(d_model)

    def forward(self, tokens, meta):
        x = self.token_embedding(tokens)
        x = torch.cat((self.metadata_embedding(meta), x), dim=-2)
        x = self.layers(x)
        # x = self.norm(x)
        return self.output_layer(x)[:,6:]
