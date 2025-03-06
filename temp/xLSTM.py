import torch
import torch.nn as nn
import torch.nn.functional as F
from temp.mLSTMblock import mLSTMblock
from temp.sLSTMblock import sLSTMblock

class xLSTM(nn.Module):
    def __init__(self, layers, x_example, depth=4, factor=2):
        super(xLSTM, self).__init__()

        self.layers = nn.ModuleList()
        for layer_type in layers:
            if layer_type == 's':
                layer = sLSTMblock(x_example, depth)
            elif layer_type == 'm':
                layer = mLSTMblock(x_example, factor, depth)
            else:
                raise ValueError(f"Invalid layer type: {layer_type}. Choose 's' for sLSTM or 'm' for mLSTM.")
            self.layers.append(layer)
    
    def init_states(self, x):
        [l.init_states(x) for l in self.layers]
        
    def forward(self, x):
        # Optionally check and ensure the input has a sequence dimension
        if x.dim() == 2:  # Shape: [batch_size, num_embd]
            x = x.unsqueeze(1)  # Add sequence length dimension to get shape [batch_size, 1, num_embd]
        elif x.dim() != 3:
            raise ValueError("Input must have either shape [batch_size, num_embd] or [batch_size, sequence_length, num_embd]")

        x_original = x.clone()
        for l in self.layers:
            x = l(x) + x_original

        return x
