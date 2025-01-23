import torch
import torch.nn as nn
import math
from config import *

import torch
import torch.nn as nn
import torch.nn.functional as F

class MultiHeadAttention(nn.Module):
    def __init__(self, d_model, num_heads):
        super(MultiHeadAttention, self).__init__()
        self.d_model = d_model
        self.num_heads = num_heads
        self.head_dim = d_model // num_heads
        
        assert self.head_dim * num_heads == d_model, "d_model must be divisible by num_heads"
        
        self.query = nn.Linear(d_model, d_model)
        self.key = nn.Linear(d_model, d_model)
        self.value = nn.Linear(d_model, d_model)
        self.fc_out = nn.Linear(d_model, d_model)
    
    def forward(self, query, key, value, mask=None):
        batch_size = query.size(0)
        
        # Linear transformations
        Q = self.query(query)
        K = self.key(key)
        V = self.value(value)
        
        # Split into multiple heads
        Q = Q.view(batch_size, -1, self.num_heads, self.head_dim).transpose(1, 2)
        K = K.view(batch_size, -1, self.num_heads, self.head_dim).transpose(1, 2)
        V = V.view(batch_size, -1, self.num_heads, self.head_dim).transpose(1, 2)
        
        # Scaled Dot-Product Attention
        scores = torch.matmul(Q, K.transpose(-2, -1)) / torch.sqrt(torch.tensor(self.head_dim, dtype=torch.float32))
        
        if mask is not None:
            scores = scores.masked_fill(mask == 0, float('-inf'))
        
        attention = F.softmax(scores, dim=-1)
        out = torch.matmul(attention, V)
        
        # Concatenate heads
        out = out.transpose(1, 2).contiguous().view(batch_size, -1, self.d_model)
        
        # Final linear layer
        out = self.fc_out(out)
        
        return out

class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=5000):
        super(PositionalEncoding, self).__init__()
        self.encoding = torch.zeros(max_len, d_model).to(DEVICE)  # Move encoding to DEVICE
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1).to(DEVICE)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-torch.log(torch.tensor(10000.0)) / d_model)).to(DEVICE)
        self.encoding[:, 0::2] = torch.sin(position * div_term)
        self.encoding[:, 1::2] = torch.cos(position * div_term)
        self.encoding = self.encoding.unsqueeze(0)
    
    def forward(self, x):
        return x + self.encoding[:, :x.size(1)].detach()

class TransformerBlock(nn.Module):
    def __init__(self, d_model, num_heads, d_ff, dropout=0.1):
        super(TransformerBlock, self).__init__()
        self.attention = MultiHeadAttention(d_model, num_heads)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.feed_forward = nn.Sequential(
            nn.Linear(d_model, d_ff),
            nn.ReLU(),
            nn.Linear(d_ff, d_model)
        )
        self.dropout = nn.Dropout(dropout)
    
    def forward(self, x, mask=None):
        attention_output = self.attention(x, x, x, mask)
        x = self.norm1(x + self.dropout(attention_output))
        ff_output = self.feed_forward(x)
        x = self.norm2(x + self.dropout(ff_output))
        return x
    
def generate_causal_mask(seq_len):
    """
    Generate a causal mask for a sequence of length `seq_len`.
    """
    mask = torch.triu(torch.ones(seq_len, seq_len), diagonal=1)  # Upper triangular matrix with diagonal=1
    mask = mask.masked_fill(mask == 1, float('-inf'))  # Replace 1s with -inf
    return mask

class SimpleTransformer(nn.Module):
    def __init__(self, src_vocab_size, d_model, num_heads, num_layers, d_ff, max_len, dropout=0.1):
        super(SimpleTransformer, self).__init__()
        self.embedding = nn.Embedding(src_vocab_size, d_model).to(DEVICE)
        self.positional_encoding = PositionalEncoding(d_model, max_len).to(DEVICE)
        self.layers = nn.ModuleList([TransformerBlock(d_model, num_heads, d_ff, dropout).to(DEVICE) for _ in range(num_layers)])  # Move each layer to DEVICE
        self.fc_out = nn.Linear(d_model, src_vocab_size).to(DEVICE)
        self.dropout = nn.Dropout(dropout).to(DEVICE)
        self.max_len = max_len  # Store as an integer, not a tensor

    def forward(self, src, mask=None):
        # Move input tensor to DEVICE
        src = src.to(DEVICE)
        
        x = self.embedding(src)  # No need for .to(DEVICE) here, as embedding is already on DEVICE
        x = self.positional_encoding(x)  # No need for .to(DEVICE) here, as positional_encoding is already on DEVICE
        x = self.dropout(x)  # No need for .to(DEVICE) here, as dropout is already on DEVICE

        for layer in self.layers:
            x = layer(x, mask)  # No need for .to(DEVICE) here, as layers are already on DEVICE

        out = self.fc_out(x)  # No need for .to(DEVICE) here, as fc_out is already on DEVICE
        return out

    def generate(self, src, gen_len):
        # Move input tensor to DEVICE
        src = src.to(DEVICE)
        
        src_len = src.size(1)  # Get the sequence length from the input tensor
        full_seq = src.detach().clone()
        
        for _ in range(gen_len - src_len):
            # Ensure the input to the model is on the correct device
            trg = self.forward(full_seq[:, -self.max_len:].to(DEVICE))
            
            # Get the last predicted token
            last_pred = trg.argmax(2)[:, -1].unsqueeze(1)  # Shape: [batch_size, 1]
            
            # Append the predicted token to the input sequence
            full_seq = torch.cat([full_seq, last_pred], dim=1).to(DEVICE)
        
        return full_seq