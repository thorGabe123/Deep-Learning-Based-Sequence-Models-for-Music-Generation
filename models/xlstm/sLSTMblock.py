import torch
import torch.nn as nn
import torch.nn.functional as F
from models.xlstm.utils import BlockDiagonal, CausalConv1D

class sLSTMblock(nn.Module):
    def __init__(self, params):
        super().__init__()
        
        self.ln = nn.LayerNorm(params.n_embd)
        
        self.conv = CausalConv1D(params.n_embd, params.n_embd, int(params.n_embd/8))
        self.drop = nn.Dropout(params.dropout)
        
        self.i_gate = BlockDiagonal(params.n_embd, params.n_embd, params.depth)
        self.f_gate = BlockDiagonal(params.n_embd, params.n_embd, params.depth)
        self.o_gate = BlockDiagonal(params.n_embd, params.n_embd, params.depth)
        self.z_gate = BlockDiagonal(params.n_embd, params.n_embd, params.depth)
        
        self.ri_gate = BlockDiagonal(params.n_embd, params.n_embd, params.depth, bias=False)
        self.rf_gate = BlockDiagonal(params.n_embd, params.n_embd, params.depth, bias=False)
        self.ro_gate = BlockDiagonal(params.n_embd, params.n_embd, params.depth, bias=False)
        self.rz_gate = BlockDiagonal(params.n_embd, params.n_embd, params.depth, bias=False)

        self.ln_i = nn.LayerNorm(params.n_embd)
        self.ln_f = nn.LayerNorm(params.n_embd)
        self.ln_o = nn.LayerNorm(params.n_embd)
        self.ln_z = nn.LayerNorm(params.n_embd)
        
        self.GN = nn.LayerNorm(params.n_embd)
        self.ln_c = nn.LayerNorm(params.n_embd)
        self.ln_n = nn.LayerNorm(params.n_embd)
        self.ln_h = nn.LayerNorm(params.n_embd)
        
        self.left_linear = nn.Linear(params.n_embd, int(params.n_embd*(4/3)))
        self.right_linear = nn.Linear(params.n_embd, int(params.n_embd*(4/3)))

        self.ln_out = nn.LayerNorm(int(params.n_embd*(4/3)))
        
        self.proj = nn.Linear(int(params.n_embd*(4/3)), params.n_embd)
        
        self.init_states(params)
        
    def init_states(self, params):
        self.nt_1 = torch.zeros(1, 1, params.n_embd, device=params.device)
        self.ct_1 = torch.zeros(1, 1, params.n_embd, device=params.device)
        self.ht_1 = torch.zeros(1, 1, params.n_embd, device=params.device)
        self.mt_1 = torch.zeros(1, 1, params.n_embd, device=params.device)
        
    def forward(self, x):
        x = self.ln(x)
        
        x_conv = F.silu( self.drop(self.conv( x.transpose(1, 2) ).transpose(1, 2) ) )
        
        # start sLSTM
        ht_1 = self.ht_1
        
        i = torch.exp(self.ln_i( self.i_gate(x_conv) + self.ri_gate(ht_1) ) )
        f = torch.exp( self.ln_f(self.f_gate(x_conv) + self.rf_gate(ht_1) ) )

        m = torch.max(torch.log(f)+self.mt_1[:, 0, :].unsqueeze(1), torch.log(i))
        i = torch.exp(torch.log(i) - m)
        f = torch.exp(torch.log(f) + self.mt_1[:, 0, :].unsqueeze(1)-m)
        self.mt_1 = m.detach()
        
        o = torch.sigmoid( self.ln_o(self.o_gate(x) + self.ro_gate(ht_1) ) )
        z = torch.tanh( self.ln_z(self.z_gate(x) + self.rz_gate(ht_1) ) )
        
        ct_1 = self.ct_1
        ct = f*ct_1 + i*z
        ct = torch.mean(self.ln_c(ct), [0, 1], keepdim=True)
        self.ct_1 = ct.detach()
        
        nt_1 = self.nt_1
        nt = f*nt_1 + i
        nt = torch.mean(self.ln_n(nt), [0, 1], keepdim=True)
        self.nt_1 = nt.detach()
        
        ht = o*(ct/nt) # torch.Size([4, 8, 16])
        ht = torch.mean(self.ln_h(ht), [0, 1], keepdim=True)
        self.ht_1 = ht.detach()
        # end sLSTM
        
        slstm_out = self.GN(ht)
        
        left = self.left_linear(slstm_out)
        right = F.gelu(self.right_linear(slstm_out))
        
        out = self.ln_out(left*right)
        out = self.proj(out)
        return out
  