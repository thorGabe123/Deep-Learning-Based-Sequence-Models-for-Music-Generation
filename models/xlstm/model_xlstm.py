import torch
import torch.nn as nn
import torch.nn.functional as F
from models.xlstm.mLSTMblock import mLSTMblock
from models.xlstm.sLSTMblock import sLSTMblock

class xLSTM(nn.Module):
    def __init__(self, params):
        super(xLSTM, self).__init__()

        self.vocab_size = params.vocab_size

        self.layers = nn.ModuleList()
        for layer_type in params.layers:
            if layer_type == 's':
                layer = sLSTMblock(params)
            elif layer_type == 'm':
                layer = mLSTMblock(params)
            else:
                raise ValueError(f"Invalid layer type: {layer_type}. Choose 's' for sLSTM or 'm' for mLSTM.")
            self.layers.append(layer)

        self.output_layer = nn.Linear(1, self.vocab_size)

    def init_states(self, x):
        [l.init_states(x) for l in self.layers]
    
    def forward(self, x):
        x = x.unsqueeze(1)
        x_original = x.clone()
        for l in self.layers:
             x = l(x) + x_original

        # Apply the linear transformation
        x = x.permute(0, 2, 1)
        x = self.output_layer(x)

        # # Apply softmax to get probabilities
        probabilities = F.log_softmax(x, dim=-1)

        return probabilities
    
    def get_name(self):
        return 'xLSTM'