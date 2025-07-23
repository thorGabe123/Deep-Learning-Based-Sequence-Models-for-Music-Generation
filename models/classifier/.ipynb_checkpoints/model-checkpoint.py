import torch
import torch.nn as nn
import configs.common as cc
import torch.nn.functional as F

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
    context_length=cc.config.values.block_len,
    embedding_dim=512,
    num_blocks=11,
    slstm_at=[1, 4, 7, 10],
)

class Classifier(nn.Module):
    def __init__(self, d_model=512, n_layers=12):
        super().__init__()

        self.token_embedding = nn.Embedding(cc.vocab_size, d_model)
        self.output_layer = nn.Linear(d_model, cc.vocab_size)
        self.layers = xLSTMBlockStack(cfg)
        self.fc = nn.Linear(d_model, cc.metadata_vocab_size)

    def forward(self, tokens):
        x = self.token_embedding(tokens)
        x = self.layers(x)
        last_hidden = x[:, -1, :]
        output = self.fc(last_hidden)
        return output

