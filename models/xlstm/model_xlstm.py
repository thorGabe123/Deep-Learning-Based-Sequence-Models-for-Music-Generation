import torch
import torch.nn as nn
import torch.nn.functional as F
from models.xlstm.mLSTMblock import mLSTMblock
from models.xlstm.sLSTMblock import sLSTMblock

class xLSTM(nn.Module):
    def __init__(self, params):
        super(xLSTM, self).__init__()

        self.vocab_size = params.vocab_size
        self.token_embedding_table = nn.Embedding(params.vocab_size, params.n_embd)

        self.layers = nn.ModuleList()
        for layer_type in params.layers:
            if layer_type == 's':
                layer = sLSTMblock(params)
            elif layer_type == 'm':
                layer = mLSTMblock(params)
            else:
                raise ValueError(f"Invalid layer type: {layer_type}. Choose 's' for sLSTM or 'm' for mLSTM.")
            self.layers.append(layer)

        self.output_layer = nn.Linear(params.n_embd, self.vocab_size)

    def init_states(self, x):
        [l.init_states(x) for l in self.layers]
    
    def forward(self, token_ids, meta_ids):
        print(token_ids.shape)
        x = self.token_embedding_table(token_ids)
        # print(x.shape)
        # x = x.unsqueeze(1)
        print(x.shape)
        x_original = x.clone()
        for l in self.layers:
             x = l(x) + x_original

        # Go from n_embd to vocab_size and then normalize

        return x
    
    def get_name(self):
        return 'xLSTM'