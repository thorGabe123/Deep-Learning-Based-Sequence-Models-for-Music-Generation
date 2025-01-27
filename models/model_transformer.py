import torch
import torch.nn as nn
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
        self.layers = nn.ModuleList([TransformerBlock(d_model, num_heads, d_ff, dropout).to(device) for _ in range(num_layers)])  # Adjust for concatenated embeddings
        self.fc_out = nn.Linear(d_model, src_vocab_size).to(device)  # Adjust for concatenated embeddings
        self.dropout = nn.Dropout(dropout).to(device)
        self.max_len = max_len
        self.M = torch.stack((torch.cat((torch.zeros(START_IDX['DYN_RES']), torch.ones(DYN_RES), torch.zeros(VOCAB_SIZE - START_IDX['LENGTH_RES']))),
            torch.cat((torch.zeros(START_IDX['PITCH_RES']), torch.ones(PITCH_RES), torch.zeros(VOCAB_SIZE - START_IDX['DYN_RES']))),
            torch.cat((torch.zeros(START_IDX['LENGTH_RES']), torch.ones(LENGTH_RES), torch.zeros(VOCAB_SIZE - START_IDX['TIME_RES']))),
            torch.cat((torch.zeros(START_IDX['TIME_RES']), torch.ones(TIME_RES), torch.zeros(VOCAB_SIZE - START_IDX['CHANNEL_RES']))),
            torch.cat((torch.zeros(START_IDX['CHANNEL_RES']), torch.ones(CHANNEL_RES), torch.zeros(VOCAB_SIZE - START_IDX['TEMPO_RES']))),
            torch.cat((torch.zeros(START_IDX['TEMPO_RES']), torch.ones(TEMPO_RES), torch.zeros(VOCAB_SIZE - START_IDX['TEMPO_RES'] - TEMPO_RES))))).to(device)

    def forward(self, src, metadata, mask=None, device="cuda"):
        src = src.to(device)
        # start_idx = [self.get_idx(x) for x in src[:, 0]]
        metadata = metadata.to(device)
        # Embed the input sequence
        src_emb = self.embedding(src)  # Shape: [batch_size, seq_len, d_model]
        src_emb = self.positional_encoding(src_emb)  # Shape: [batch_size, seq_len, d_model]
        metadata_emb = self.metadata_embedding(metadata)  # Shape: [batch_size, 6, d_model]
        
        x = torch.cat([metadata_emb, src_emb], dim=-2)  # Shape: [batch_size, seq_len + 6, d_model]
        x = self.dropout(x)
        
        # Pass through transformer layers
        for layer in self.layers:
            x = layer(x, mask)
        
        # Final output layer
        out = self.fc_out(x)  # Shape: [batch_size, seq_len, src_vocab_size]
        out = out[:, -src_emb.size(1):]
        
        # # Restrict output based on M and start_idx
        # batch_size, seq_len, vocab_size = out.shape
        # restricted_out = torch.zeros_like(out).to(device)  # Create a zero tensor with the same shape
        
        # for i in range(batch_size):
        #     for j in range(seq_len):
        #         idx = start_idx[i]
        #         restricted_out[i, j, :] = out[i, j, :] * self.M[idx, :]

        return out
    
    def get_idx(self, num):
        if START_IDX['PITCH_RES'] <= num < START_IDX['DYN_RES']:
            return 1
        elif START_IDX['DYN_RES'] <= num < START_IDX['LENGTH_RES']:
            return 0
        elif START_IDX['LENGTH_RES'] <= num < START_IDX['TIME_RES']:
            return 2
        elif START_IDX['TIME_RES'] <= num < START_IDX['CHANNEL_RES']:
            return 3
        elif START_IDX['CHANNEL_RES'] <= num < START_IDX['TEMPO_RES']:
            return 4
        elif START_IDX['TEMPO_RES'] <= num < VOCAB_SIZE:
            return 5

    def generate(self, src, metadata, gen_len, device="cuda"):
        src = src.to(device)
        metadata = metadata.to(device)
        
        src_len = src.size(1)
        full_seq = src.detach().clone()
        full_metadata = metadata.detach().clone()
        
        for _ in range(gen_len - src_len):
            trg = self.forward(full_seq[:, -self.max_len:], full_metadata, device=device)
            
            last_pred = trg.argmax(2)[:, -1].unsqueeze(1) 
            
            full_seq = torch.cat([full_seq, last_pred], dim=1).to(device)
            
            # Repeat the last metadata for the new token
            full_metadata = {
                'decade': full_metadata['decade'],
                'genres': full_metadata['genres'],
                'band': full_metadata['band']
            }
        
        return full_seq