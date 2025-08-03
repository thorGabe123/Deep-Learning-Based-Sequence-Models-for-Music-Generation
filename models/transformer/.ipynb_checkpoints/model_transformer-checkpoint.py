import torch
import torch.nn as nn
import configs.common as cc
import torch.nn.functional as F
import math


def generate_matrix(n: int, x: int) -> torch.Tensor:
    matrix = torch.zeros((n, n), dtype=torch.float32, device=cc.config.values.device)
    
    for i in range(n):
        matrix[i, : ((i // x) + 1) * x] = 1.
        # Set the first 6 columns of each row to 1
        matrix[i, :6] = 1.0
    
    return matrix
class Head(nn.Module):
    """ one head of self-attention """
    
    def __init__(self, n_embd, head_size, block_size):
        super().__init__()
        self.key = nn.Linear(n_embd, head_size, bias=False)
        self.query = nn.Linear(n_embd, head_size, bias=False)
        self.value = nn.Linear(n_embd, head_size, bias=False)
        self.register_buffer('tril', generate_matrix(block_size, 6))
    
        self.dropout = nn.Dropout(cc.config.values.dropout)
    
    def forward(self, x):
        B,T,C = x.shape
        k = self.key(x)   # (B,T,C)
        q = self.query(x) # (B,T,C)
        # compute attention scores ("affinities")
        wei = q @ k.transpose(-2,-1) * C**-0.5 # (B, T, C) @ (B, C, T) -> (B, T, T)
        wei = wei.masked_fill(self.tril[:T, :T] == 0, float('-inf')) # (B, T, T)
        wei = F.softmax(wei, dim=-1) # (B, T, T)
        wei = self.dropout(wei)
        # perform the weighted aggregation of the values
        v = self.value(x) # (B,T,C)
        out = wei @ v # (B, T, T) @ (B, T, C) -> (B, T, C)
        return out

class MultiHeadAttention(nn.Module):
    """ multiple heads of self-attention in parallel """

    def __init__(self, n_embd, n_heads, head_size, block_size, dropout):
        super().__init__()
        # self.heads = nn.ModuleList([Head(n_embd, head_size, block_size) for _ in range(n_heads)])
        self.heads = nn.ModuleList([HeadRelPos(n_embd, head_size, block_size) for _ in range(n_heads)])
        self.proj = nn.Linear(n_embd, n_embd)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        out = torch.cat([h(x) for h in self.heads], dim=-1)
        out = self.dropout(self.proj(out))
        return out

class HeadRelPos(nn.Module):
    """ one head of self-attention with relative positional encoding (Transformer-XL style) """

    def __init__(self, n_embd, head_size, block_size):
        super().__init__()
        self.key = nn.Linear(n_embd, head_size, bias=False)
        self.query = nn.Linear(n_embd, head_size, bias=False)
        self.value = nn.Linear(n_embd, head_size, bias=False)
        self.rel_pos_emb = nn.Parameter(torch.randn(block_size, head_size))  # learnable relative position embedding
        self.register_buffer('tril', generate_matrix(block_size, 1))
        self.dropout = nn.Dropout(cc.config.values.dropout)

    def forward(self, x):
        B, T, C = x.shape
        k = self.key(x)   # (B,T,head_size)
        q = self.query(x) # (B,T,head_size)
        v = self.value(x) # (B,T,head_size)

        # Content-based attention
        AC = torch.einsum('bth,bsh->bts', q, k)  # (B, T, T)

        # Relative positional attention
        # rel_pos_emb: (block_size, head_size)
        rel_pos_emb = self.rel_pos_emb[:T, :]  # (T, head_size)
        # q: (B, T, head_size), rel_pos_emb: (T, head_size)
        BD = torch.einsum('bth,sh->bts', q, rel_pos_emb)  # (B, T, T)

        # Shift BD so that each position i attends to j-i
        BD = self._rel_shift(BD)

        # Combine
        attn = (AC + BD) * (C ** -0.5)
        attn = attn.masked_fill(self.tril[:T, :T] == 0, float('-inf'))
        attn = F.softmax(attn, dim=-1)
        attn = self.dropout(attn)
        out = attn @ v
        return out

    @staticmethod
    def _rel_shift(x):
        # x: (B, T, T)
        # Shift the matrix for relative positions: see Transformer-XL Appendix A.4
        B, T, _ = x.size()
        # Add a dummy zero column on the left
        zero_pad = torch.zeros((B, T, 1), device=x.device, dtype=x.dtype)
        x_padded = torch.cat([zero_pad, x], dim=2)  # (B, T, T+1)
        x_shifted = x_padded.view(B, T + 1, T)[:, 1:, :]  # (B, T, T)
        return x_shifted

class FeedFoward(nn.Module):
    """ a simple linear layer followed by a non-linearity """

    def __init__(self, n_embd, dropout):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(n_embd, 4 * n_embd),
            nn.ReLU(),
            nn.Linear(4 * n_embd, n_embd),
            nn.Dropout(dropout),
        )

    def forward(self, x):
        return self.net(x)

class Block(nn.Module):
    """ Transformer block: communication followed by computation """

    def __init__(self, n_embd, n_heads, block_size, dropout):
        # n_embd: embedding dimension, n_head: the number of heads we'd like
        super().__init__()
        head_size = n_embd // n_heads
        self.sa = MultiHeadAttention(n_embd, n_heads, head_size, block_size, dropout)
        self.ffwd = FeedFoward(n_embd, dropout)
        self.ln1 = nn.LayerNorm(n_embd)
        self.ln2 = nn.LayerNorm(n_embd)

    def forward(self, x):
        x = x + self.sa(self.ln1(x))
        x = x + self.ffwd(self.ln2(x))
        return x
    
class PositionalEncoding(nn.Module):
    def __init__(self, n_embd, max_len=5000, device="cuda"):
        super(PositionalEncoding, self).__init__()
        self.encoding = torch.zeros([max_len, n_embd]).to(device)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1).to(device)
        div_term = torch.exp(torch.arange(0, n_embd, 2).float() * (-torch.log(torch.tensor(10000.0)) / n_embd)).to(device)
        self.encoding[:, 0::2] = torch.sin(position // 6 * div_term)
        self.encoding[:, 1::2] = torch.cos(position // 6 * div_term)
        self.encoding = self.encoding.unsqueeze(0)
    
    def forward(self, x):
        return x + self.encoding[:, :x.size(1)].detach()

class Transformer(nn.Module):
    def __init__(self, params):
        super().__init__()
        self.vocab_size = params.vocab_size
        self.metadata_vocab_size = params.metadata_vocab_size
        self.token_embedding_table = nn.Embedding(params.vocab_size, params.n_embd)
        self.metadata_embedding_table = nn.Embedding(params.metadata_vocab_size, params.n_embd)
        # self.positional_encoding = PositionalEncoding(params.n_embd, params.block_len, params.device)

        self.blocks = nn.Sequential(*[Block(params.n_embd, params.n_heads, params.block_len + 6, params.dropout) for _ in range(params.n_layer)])
        self.ln_f = nn.LayerNorm(params.n_embd)  # final layer norm
        self.lm_head = nn.Linear(params.n_embd, params.vocab_size)

    def forward(self, idx, metadata_idx, targets=None):
        B, T = idx.shape

        x = self.token_embedding_table(idx)  # Shape: (B, T, C)
        # x = self.positional_encoding(x)
        metadata_embedding = self.metadata_embedding_table(metadata_idx)
        x = torch.cat((metadata_embedding, x), dim=-2)
        
        B1, T1, C1 = x.shape

        # Transformer blocks
        x = self.blocks(x)  # Shape: (B, T, C)
        x = self.ln_f(x)    # Shape: (B, T, C)
        logits = self.lm_head(x)  # Shape: (B, T, vocab_size)

        logits = logits.view(B1, T1, -1)
        logits = logits[:, -T:, :]
        return logits
    
    def get_name(self):
        return 'Transformer'
