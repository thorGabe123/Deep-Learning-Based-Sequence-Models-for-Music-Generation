import torch
import torch.nn as nn

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
    def __init__(self, d_model, max_len=5000, device="cuda"):
        super(PositionalEncoding, self).__init__()
        self.encoding = torch.zeros(max_len, d_model).to(device)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1).to(device)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-torch.log(torch.tensor(10000.0)) / d_model)).to(device)
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
    mask = torch.triu(torch.ones(seq_len, seq_len), diagonal=1)
    mask = mask.masked_fill(mask == 1, float('-inf'))
    return mask

class SimpleTransformer(nn.Module):
    def __init__(self, src_vocab_size, metadata_vocab_size, d_model, num_heads, num_layers, d_ff, max_len, dropout=0.1, device="cuda"):
        super(SimpleTransformer, self).__init__()
        self.embedding = nn.Embedding(src_vocab_size, d_model).to(device)
        self.metadata_embedding = nn.Embedding(metadata_vocab_size, d_model).to(device)  # Metadata embedding layer
        self.positional_encoding = PositionalEncoding(d_model, max_len, device)
        self.layers = nn.ModuleList([TransformerBlock(d_model * 2, num_heads, d_ff, dropout).to(device) for _ in range(num_layers)])  # Adjust for concatenated embeddings
        self.fc_out = nn.Linear(d_model * 2, src_vocab_size).to(device)  # Adjust for concatenated embeddings
        self.dropout = nn.Dropout(dropout).to(device)
        self.max_len = max_len

    def forward(self, src, metadata, mask=None, device="cuda"):
        src = src.to(device)
        metadata = metadata.to(device)
        metadata_emb = self.metadata_embedding(metadata)
        src_emb = self.embedding(src)
        src_emb = self.positional_encoding(src_emb)
        
        x = torch.cat([src_emb, metadata_emb], dim=-1) 
        x = self.dropout(x)
        
        for layer in self.layers:
            x = layer(x, mask)
        
        out = self.fc_out(x)
        return out

    def generate(self, src, metadata, gen_len, device="cuda"):
        src = src.to(device)
        metadata = metadata.to(device)
        
        src_len = src.size(1)
        full_seq = src.detach().clone()
        full_metadata = metadata.detach().clone()
        
        for _ in range(gen_len - src_len):
            trg = self.forward(full_seq[:, -self.max_len:], full_metadata[:, -self.max_len:], device=device)
            
            last_pred = trg.argmax(2)[:, -1].unsqueeze(1) 
            
            full_seq = torch.cat([full_seq, last_pred], dim=1).to(device)
            
            full_metadata = torch.cat([full_metadata, full_metadata[:, -1:]], dim=1).to(device)
        
        return full_seq