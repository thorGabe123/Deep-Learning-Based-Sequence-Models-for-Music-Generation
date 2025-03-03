import torch
import torch.nn as nn
import torch.nn.functional as F
from models.xlstm.utils import BlockDiagonal, CausalConv1D

class mLSTMblock(nn.Module):
    def __init__(self, params):
        super().__init__()
        self.n_embd = params.n_embd
        self.hidden_size = int(self.n_embd*params.factor)
        
        self.ln = nn.LayerNorm(self.n_embd)
        
        self.left = nn.Linear(self.n_embd, self.hidden_size)
        self.right = nn.Linear(self.n_embd, self.hidden_size)
        
        self.conv = CausalConv1D(self.hidden_size, self.hidden_size, int(self.n_embd/10)) 
        self.drop = nn.Dropout(params.dropout+0.1)
        
        self.lskip = nn.Linear(self.hidden_size, self.hidden_size)
        
        self.wq = BlockDiagonal(self.hidden_size, self.hidden_size, params.depth)
        self.wk = BlockDiagonal(self.hidden_size, self.hidden_size, params.depth)
        self.wv = BlockDiagonal(self.hidden_size, self.hidden_size, params.depth)
        self.dropq = nn.Dropout(params.dropout/2)
        self.dropk = nn.Dropout(params.dropout/2)
        self.dropv = nn.Dropout(params.dropout/2)
        
        self.i_gate = nn.Linear(self.hidden_size, self.hidden_size)
        self.f_gate = nn.Linear(self.hidden_size, self.hidden_size)
        self.o_gate = nn.Linear(self.hidden_size, self.hidden_size)

        self.ln_c = nn.LayerNorm(self.hidden_size)
        self.ln_n = nn.LayerNorm(self.hidden_size)
        
        self.lnf = nn.LayerNorm(self.hidden_size)
        self.lno = nn.LayerNorm(self.hidden_size)
        self.lni = nn.LayerNorm(self.hidden_size)
        
        self.GN = nn.LayerNorm(self.hidden_size)
        self.ln_out = nn.LayerNorm(self.hidden_size)

        self.drop2 = nn.Dropout(params.dropout)
        
        self.proj = nn.Linear(self.hidden_size, self.n_embd)
        self.ln_proj = nn.LayerNorm(self.n_embd)
        
        self.init_states(params)
    
    def init_states(self, params):
        self.ct_1 = torch.zeros([1, 1, self.hidden_size], device=params.device)
        self.nt_1 = torch.zeros([1, 1, self.hidden_size], device=params.device)
    
    def forward(self, x):
        assert x.ndim == 3
        
        x = self.ln(x) # layer norm on x
        
        left = self.left(x) # part left 
        right = F.silu(self.right(x)) # part right with just swish (silu) function

        left_left = left.transpose(1, 2)
        left_left = F.silu( self.drop( self.conv( left_left ).transpose(1, 2) ) )
        l_skip = self.lskip(left_left)

        # start mLSTM
        q = self.dropq(self.wq(left_left))
        k = self.dropk(self.wk(left_left))
        v = self.dropv(self.wv(left))
        
        i = torch.exp(self.lni(self.i_gate(left_left)))
        f = torch.exp(self.lnf(self.f_gate(left_left)))
        o = torch.sigmoid(self.lno(self.o_gate(left_left)))

        ct_1 = self.ct_1
        ct = f*ct_1 + i*v*k
        ct = torch.mean(self.ln_c(ct), [0, 1], keepdim=True)
        self.ct_1 = ct.detach()
        
        nt_1 = self.nt_1
        nt = f*nt_1 + i*k
        nt =torch.mean( self.ln_n(nt), [0, 1], keepdim=True)
        self.nt_1 = nt.detach()
        
        ht = o * ((ct*q) / torch.max(nt*q))
        # end mLSTM
        ht = ht
        
        left = self.drop2(self.GN(ht + l_skip))
        
        out = self.ln_out(left * right)
        out = self.ln_proj(self.proj(out))
        
        return out