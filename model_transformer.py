import torch
import torch.nn as nn
import math
from config import *

class TransformerNLP(nn.Module):
    def __init__(self, vocab_size, embed_size, num_heads, num_encoder_layers, num_decoder_layers, feedforward_dim, dropout=0.1):
        super(TransformerNLP, self).__init__()

        # Embedding layers for input and output vocab
        self.input_embedding = nn.Embedding(vocab_size, embed_size)
        self.output_embedding = nn.Embedding(vocab_size, embed_size)

        # Positional encoding
        self.positional_encoding = PositionalEncoding(embed_size, dropout)

        # Transformer model
        self.transformer = nn.Transformer(
            d_model=embed_size,
            nhead=num_heads,
            num_encoder_layers=num_encoder_layers,
            num_decoder_layers=num_decoder_layers,
            dim_feedforward=feedforward_dim,
            dropout=dropout,
        )

        # Final linear layer to project to output vocabulary size
        self.fc_out = nn.Linear(embed_size, vocab_size)

    def forward(self, src, tgt, src_mask=None, tgt_mask=None, src_padding_mask=None, tgt_padding_mask=None, memory_key_padding_mask=None):
        # Embed and apply positional encoding
        src = self.input_embedding(src) * math.sqrt(self.input_embedding.embedding_dim)
        src = self.positional_encoding(src)

        tgt = self.output_embedding(tgt) * math.sqrt(self.output_embedding.embedding_dim)
        tgt = self.positional_encoding(tgt)

        # Ensure batch-first format for transformer
        src = src.permute(1, 0, 2)  # [batch_size, seq_len, embed_size] -> [seq_len, batch_size, embed_size]
        tgt = tgt.permute(1, 0, 2)

        # Pass through transformer
        output = self.transformer(
            src,
            tgt,
            src_mask=src_mask,
            tgt_mask=tgt_mask,
            src_key_padding_mask=src_padding_mask,
            tgt_key_padding_mask=tgt_padding_mask,
            memory_key_padding_mask=memory_key_padding_mask,
        )

        # Project to output vocabulary size
        output = output.permute(1, 0, 2)  # [seq_len, batch_size, embed_size] -> [batch_size, seq_len, embed_size]
        output = self.fc_out(output)
        return output

class PositionalEncoding(nn.Module):
    def __init__(self, embed_size, dropout, max_len=5000):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)

        # Compute positional encodings
        pe = torch.zeros(max_len, embed_size)
        position = torch.arange(0, max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, embed_size, 2) * -(math.log(10000.0) / embed_size))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)
        self.register_buffer('pe', pe)

    def forward(self, x):
        x = x + self.pe[:, :x.size(1), :]
        return self.dropout(x)