import torch
import torch.nn as nn
from torch.nn import functional as F
from config import *

class Head(nn.Module):
    """ one head of self-attention """

    def __init__(self, head_size):
        super().__init__()
        self.key = nn.Linear(N_EMBD, head_size, bias=False)
        self.query = nn.Linear(N_EMBD, head_size, bias=False)
        self.value = nn.Linear(N_EMBD, head_size, bias=False)
        self.register_buffer('tril', torch.tril(torch.ones(BLOCK_SIZE, BLOCK_SIZE)))

        self.dropout = nn.Dropout(DROPOUT)

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

    def __init__(self, num_heads, head_size):
        super().__init__()
        self.heads = nn.ModuleList([Head(head_size) for _ in range(num_heads)])
        self.proj = nn.Linear(N_EMBD, N_EMBD)
        self.dropout = nn.Dropout(DROPOUT)

    def forward(self, x):
        out = torch.cat([h(x) for h in self.heads], dim=-1)
        out = self.dropout(self.proj(out))
        return out

class FeedFoward(nn.Module):
    """ a simple linear layer followed by a non-linearity """

    def __init__(self, N_EMBD):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(N_EMBD, 4 * N_EMBD),
            nn.ReLU(),
            nn.Linear(4 * N_EMBD, N_EMBD),
            nn.Dropout(DROPOUT),
        )

    def forward(self, x):
        return self.net(x)

class Block(nn.Module):
    """ Transformer block: communication followed by computation """

    def __init__(self, N_EMBD, n_head):
        # N_EMBD: embedding dimension, n_head: the number of heads we'd like
        super().__init__()
        head_size = N_EMBD // n_head
        self.sa = MultiHeadAttention(n_head, head_size)
        self.ffwd = FeedFoward(N_EMBD)
        self.ln1 = nn.LayerNorm(N_EMBD)
        self.ln2 = nn.LayerNorm(N_EMBD)

    def forward(self, x):
        x = x + self.sa(self.ln1(x))
        x = x + self.ffwd(self.ln2(x))
        return x

# super simple bigram model
class MusicalTransformer(nn.Module):
    def __init__(self):
        super().__init__()
        # Embedding layers for tokens and positions
        self.token_embedding_table = nn.Embedding(VOCAB_SIZE, N_EMBD)
        self.position_embedding_table = nn.Embedding(BLOCK_SIZE, N_EMBD)

        # Embedding layers for metadata (assuming METADATA_DIMS is a dict with metadata types and dimensions)
        self.metadata_embeddings = nn.ModuleDict({
            key: nn.Embedding(num_classes, N_EMBD)
            for key, num_classes in METADATA_DIMS.items()
        })

        # Transformer blocks
        self.blocks = nn.Sequential(*[Block(N_EMBD, n_head=N_HEAD) for _ in range(N_LAYER)])
        self.ln_f = nn.LayerNorm(N_EMBD)  # final layer norm
        self.lm_head = nn.Linear(N_EMBD, VOCAB_SIZE)

    def forward(self, idx, metadata, targets=None):
        B, T = idx.shape

        # Embedding lookups for tokens and positions
        tok_emb = self.token_embedding_table(idx)  # Shape: (B, T, C)
        pos_emb = self.position_embedding_table(torch.arange(T, device=idx.device))  # Shape: (T, C)
        x = torch.concatenate((tok_emb, pos_emb))  # Shape: (B, T, C)

        # Embedding lookups for metadata and combining them
        metadata_emb = sum(
            emb_layer(metadata[key])
            for key, emb_layer in self.metadata_embeddings.items()
        ).unsqueeze(1)  # Shape: (B, 1, C)

        # Add or concatenate metadata embeddings to sequence embeddings
        x = torch.concatenate((x, metadata_emb))  # Broadcasting metadata to each position (B, T, C)

        # Transformer blocks
        x = self.blocks(x)  # Shape: (B, T, C)
        x = self.ln_f(x)    # Shape: (B, T, C)
        logits = self.lm_head(x)  # Shape: (B, T, VOCAB_SIZE)

        # Calculate loss if targets are provided
        if targets is None:
            loss = None
        else:
            logits = logits.view(B * T, -1)
            targets = targets.view(B * T)
            loss = F.cross_entropy(logits, targets)

        logits = logits.view(B, T, -1)
        return logits, loss

    def generate(self, idx, metadata, max_new_tokens):
        # idx is (B, T) array of indices in the current context
        for _ in range(max_new_tokens):
            # Crop idx to the last BLOCK_SIZE tokens
            idx_cond = idx[:, -BLOCK_SIZE:].long()
            # Get predictions conditioned on metadata
            logits, _ = self(idx_cond, metadata)
            # Focus only on the last time step
            logits = logits[:, -1, :]  # (B, VOCAB_SIZE)
            # Apply softmax to get probabilities
            probs = F.softmax(logits, dim=-1)  # (B, VOCAB_SIZE)
            # Sample from the distribution
            idx_next = torch.multinomial(probs, num_samples=1)  # (B, 1)
            # Append sampled index to the running sequence
            idx = torch.cat((idx, idx_next), dim=1)  # (B, T+1)
        return idx