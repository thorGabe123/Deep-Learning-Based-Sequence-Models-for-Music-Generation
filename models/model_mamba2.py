import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange  # You might need to install this: pip install einops
import math

class MambaBlock(nn.Module):
    def __init__(self, dim, d_state=16, d_conv=4, expand=2, dt_rank="auto", d_inner=None,
                 conv_bias=True, bias=False, use_fast_path=True):
        """
        Initializes a Mamba block.

        Args:
            dim: Dimension of the input and output.
            d_state: Dimension of the state.
            d_conv: Dimension of the convolution kernel.
            expand: Expansion factor for the hidden dimension in the block.
            dt_rank: Rank of the delta projection.
            d_inner: Inner dimension, defaults to expand * dim.
            conv_bias: Whether to use bias in the convolution.
            bias: Whether to use bias in the linear layers.
            use_fast_path: Whether to use the fast path for inference.
        """
        super().__init__()
        self.dim = dim
        self.d_state = d_state
        self.d_inner = d_inner or int(expand * dim)
        self.d_conv = d_conv
        self.expand = expand
        self.use_fast_path = use_fast_path

        self.in_proj = nn.Linear(dim, self.d_inner * 2, bias=bias)
        self.conv1d = nn.Conv1d(
            in_channels=self.d_inner,
            out_channels=self.d_inner,
            bias=conv_bias,
            kernel_size=d_conv,
            groups=self.d_inner,
            padding=(d_conv - 1) // 2,
        )

        # Learnable state transition parameters
        if dt_rank == "auto":
            dt_rank = math.ceil(self.dim / 16)
        self.dt_proj = nn.Linear(self.d_inner, dt_rank, bias=bias)
        self.A_proj = nn.Parameter(torch.randn(self.d_inner, self.d_state))
        self.B_proj = nn.Parameter(torch.randn(self.d_inner, self.d_state))
        self.C_proj = nn.Parameter(torch.randn(self.d_inner, self.d_state))

        self.out_proj = nn.Linear(self.d_inner, dim, bias=bias)

    def forward(self, x, *args, **kwargs):
        """
        Forward pass of the Mamba block.

        Args:
            x: Input tensor of shape [batch, seq_len, dim]

        Returns:
            Output tensor of shape [batch, seq_len, dim]
        """
        # In-projection to get x and z (split along the last dimension)
        xz = self.in_proj(x)
        x, z = xz.chunk(2, dim=-1)  # (B L D) -> (B L d_inner)

        # Convolution
        x = rearrange(x, 'b l d -> b d l')
        x = self.conv1d(x)[:, :, : x.size(2)]
        x = rearrange(x, 'b d l -> b l d')

        # Gating with silu
        x = F.silu(x)

        # Selective SSM
        A = -torch.exp(self.A_proj)  # Ensure real negative A
        B = self.B_proj.float()
        C = self.C_proj.float()
        
        # Adding config option for using the fast path during inference.
        if self.use_fast_path and x.size(1) == 1:
            delta = self.dt_proj(x)  # (B L d_inner) -> (B L dt_rank)
            delta = F.softplus(delta)
            
            ssm_args = kwargs.get('ssm_args', None)
            if ssm_args is None or kwargs.get('mode', None) == 'train':
                # Ssm_args are only carried over when in eval mode and a sequence size of 1 is passed.
                x_dbl, x_d = self.selective_scan(x, delta, A, B, C)
                y = x_dbl * z * x_d
            else:
                # Retrieve the previous ssm state.
                ssm_state = ssm_args['ssm_state']
                ssm_state.copy_(delta * ssm_state + B.unsqueeze(0) * x)
                y = z * (ssm_state * C.unsqueeze(0)).sum(dim=-1)
                kwargs['ssm_args'] = {'ssm_state': ssm_state}
        else:
            # Discretize A and B
            delta = self.dt_proj(x)  # (B L d_inner) -> (B L dt_rank)
            delta = F.softplus(delta)
            x_dbl, x_d = self.selective_scan(x, delta, A, B, C)
            y = x_dbl * z * x_d

        # Out-projection
        y = self.out_proj(y)
        return y, kwargs

    def selective_scan(self, u, delta, A, B, C):
        """
        Does the selective scan operation.

        Args:
            u: Input tensor of shape [batch, seq_len, dim]
            delta: Discretization delta of shape [batch, seq_len, dt_rank]
            A: State transition matrix of shape [dim, state]
            B: Input-to-state projection of shape [dim, state]
            C: State-to-output projection of shape [dim, state]

        Returns:
            Tuple of output tensor of shape [batch, seq_len, dim] and state tensor of shape [batch, seq_len, dim]
        """
        delta_A = torch.exp(torch.einsum('b l r, d r -> b l d', delta, A))
        delta_B_u = torch.einsum('b l r, d r, b l d -> b l d', delta, B, u)

        N = A.size(0)
        x = torch.zeros_like(u).to(u)
        d = torch.ones_like(u).to(u)
        
        for i in range(u.size(1)):
            x[:, i, :] = delta_A[:, i, :] * x[:, i - 1, :] + delta_B_u[:, i, :]
            d[:, i, :] = d[:, i - 1, :] + u[:, i, :] * x[:, i, :]
            
        x_c = torch.einsum('b l d, d r -> b l r', x, C)

        return x_c, d

class Mamba(nn.Module):
    def __init__(self, d_model, n_layer, vocab_size):
        """
        Initializes a Mamba model.

        Args:
            d_model: Dimension of the model.
            n_layer: Number of Mamba blocks.
            vocab_size: Size of the vocabulary.
        """
        super().__init__()
        self.d_model = d_model
        self.n_layer = n_layer
        self.vocab_size = vocab_size

        # Embedding layer
        self.embedding = nn.Embedding(vocab_size, d_model)

        # Mamba blocks
        self.layers = nn.ModuleList([MambaBlock(d_model) for _ in range(n_layer)])

        # Final layer normalization
        self.norm_f = nn.LayerNorm(d_model)

        # LM head
        self.lm_head = nn.Linear(d_model, vocab_size)

    def forward(self, input_ids, *args, **kwargs):
        """
        Forward pass of the Mamba model.

        Args:
            input_ids: Input token IDs of shape [batch, seq_len]

        Returns:
            Output logits of shape [batch, seq_len, vocab_size]
        """
        # Embedding
        x = self.embedding(input_ids)  # (B, L) -> (B, L, d_model)

        # Passing through Mamba blocks
        for layer in self.layers:
            x, kwargs = layer(x, *args, **kwargs)

        # Final layer normalization
        x = self.norm_f(x)

        # LM head
        logits = self.lm_head(x)

        return logits, kwargs

    def load_config_state_dict(self, config, state_dict):
        """Loads the state dict, possibly also a config in the future."""
        self.load_state_dict(state_dict)

    @torch.no_grad()
    def generate(self, input_ids, max_new_tokens=64, temperature=1.0, top_k=None, top_p=None):
        """
        Generates text using the Mamba model.

        Args:
            input_ids: Input token IDs of shape [batch, seq_len]
            max_new_tokens: Maximum number of new tokens to generate
            temperature: Temperature for sampling
            top_k: Top-k sampling
            top_p: Top-p sampling

        Returns:
            Generated token IDs of shape [batch, seq_len + max_new_tokens]
        """
        # Ensure we are in evaluation mode
        self.eval()

        # Add padding to input if necessary
        batch, seq_len = input_ids.shape
        ssm_args = {'ssm_state': torch.zeros(batch, self.d_model, 16).to(input_ids.device)}

        # Generate new tokens
        for _ in range(max_new_tokens):
            # Forward pass to get logits
            logits, ssm_args = self(input_ids[:, -1:], mode='eval', ssm_args=ssm_args)  # (B, 1, vocab_size)
            logits = logits[:, -1, :]  # (B, vocab_size)

            # Apply temperature
            if temperature > 0:
                logits = logits / temperature

            # Apply top-k and top-p filtering
            if top_k is not None:
                v, _ = torch.topk(logits, top_k)
                logits[logits < v[:, [-1]]] = -float('Inf')
            if top_p is not None:
                sorted_logits, sorted_indices = torch.sort(logits, descending=True)
                cumulative_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)
                sorted_indices_to_remove = cumulative_probs > top_p
                sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
                sorted_indices_to_remove[..., 0] = 0
                indices_to_remove = sorted_indices_to_remove.scatter(-1, sorted_indices, sorted_indices_to_remove)
                logits[indices_to_remove] = -float('Inf')

            # Sample next token
            probs = F.softmax(logits, dim=-1)
            next_token = torch.multinomial(probs, num_samples=1)  # (B, 1)

            # Append to input
            input_ids = torch.cat([input_ids, next_token], dim=1)

        return input_ids