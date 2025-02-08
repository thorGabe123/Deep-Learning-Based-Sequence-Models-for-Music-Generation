import torch
import torch.nn as nn
import torch.nn.functional as F

class MambaBlock(nn.Module):
    def __init__(self, d_model, d_state, d_ff):
        super(MambaBlock, self).__init__()
        self.d_model = d_model
        self.d_state = d_state
        self.d_ff = d_ff

        # State space model parameters
        self.A = nn.Parameter(torch.randn(d_state, d_state))
        self.B = nn.Parameter(torch.randn(d_model, d_state))
        self.C = nn.Parameter(torch.randn(d_model, d_state))
        self.D = nn.Parameter(torch.randn(d_model))

        # Feed-forward network
        self.ffn = nn.Sequential(
            nn.Linear(d_model, d_ff),
            nn.ReLU(),
            nn.Linear(d_ff, d_model)
        )

    def forward(self, x):
        batch_size, seq_len, _ = x.shape

        # Initialize the state
        state = torch.zeros(batch_size, self.d_state, device=x.device)

        # Apply the state space model
        outputs = []
        for t in range(seq_len):
            state = torch.matmul(state, self.A) + torch.matmul(x[:, t, :], self.B)
            output = torch.matmul(state, self.C.T) + self.D
            outputs.append(output)

        # Stack outputs along the sequence dimension
        output = torch.stack(outputs, dim=1)

        # Apply the feed-forward network
        output = self.ffn(output)

        return output

class MambaModel(nn.Module):
    def __init__(self, d_model, d_state, d_ff, num_layers, vocab_size):
        super(MambaModel, self).__init__()
        self.embedding = nn.Embedding(vocab_size, d_model)
        self.layers = nn.ModuleList([MambaBlock(d_model, d_state, d_ff) for _ in range(num_layers)])
        self.fc_out = nn.Linear(d_model, vocab_size)

    def forward(self, x):
        x = self.embedding(x)
        for layer in self.layers:
            x = layer(x)
        x = self.fc_out(x)
        return x