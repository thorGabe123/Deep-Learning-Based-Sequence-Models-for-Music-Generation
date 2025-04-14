import torch
import torch.nn as nn
import configs.common as cc
import torch.nn.functional as F

def generate_matrix(n: int, x: int) -> torch.Tensor:
    matrix = torch.zeros((n, n), dtype=torch.float32, device=cc.config.values.device)
    
    for i in range(n):
        matrix[i, : ((i // x) + 1) * x] = 1.
        # Set the last 6 columns of each row to 1
        matrix[i, -6:] = 1.0
    
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
        self.heads = nn.ModuleList([Head(n_embd, head_size, block_size) for _ in range(n_heads)])
        self.proj = nn.Linear(n_embd, n_embd)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        out = torch.cat([h(x) for h in self.heads], dim=-1)
        out = self.dropout(self.proj(out))
        return out

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
        # Embedding layers for tokens and positions
        self.token_embedding_table = nn.Embedding(params.vocab_size, params.n_embd)
        self.metadata_embedding_table = nn.Embedding(params.metadata_vocab_size, params.n_embd)
        self.positional_encoding = PositionalEncoding(params.n_embd, params.block_len, params.device)

        # Transformer blocks
        self.blocks = nn.Sequential(*[Block(params.n_embd, params.n_heads, params.block_len + 6, params.dropout) for _ in range(params.n_layer)])
        self.ln_f = nn.LayerNorm(params.n_embd)  # final layer norm
        self.lm_head = nn.Linear(params.n_embd, params.vocab_size)

    def forward(self, idx, metadata_idx, targets=None):
        B, T = idx.shape

        # Embedding lookups for tokens and positions
        x = self.token_embedding_table(idx)  # Shape: (B, T, C)
        x = self.positional_encoding(x)
        metadata_embedding = self.metadata_embedding_table(metadata_idx)
        x = torch.cat((x, metadata_embedding), dim=-2)

        B1, T1, C1 = x.shape

        # Transformer blocks
        x = self.blocks(x)  # Shape: (B, T, C)
        x = self.ln_f(x)    # Shape: (B, T, C)
        logits = self.lm_head(x)  # Shape: (B, T, vocab_size)

        # logits = logits.view(B, T, -1)
        logits = logits.view(B1, T1, -1)
        logits = logits[:, :T, :]
        return logits
    
    def get_name(self):
        return 'Transformer'