import torch
import torch.nn as nn

class xLSTMCell(nn.Module):
    def __init__(self, input_size, hidden_size):
        super(xLSTMCell, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size

        # Input gate
        self.W_xi = nn.Parameter(torch.Tensor(input_size, hidden_size))
        self.W_hi = nn.Parameter(torch.Tensor(hidden_size, hidden_size))
        self.b_i = nn.Parameter(torch.Tensor(hidden_size))

        # Forget gate
        self.W_xf = nn.Parameter(torch.Tensor(input_size, hidden_size))
        self.W_hf = nn.Parameter(torch.Tensor(hidden_size, hidden_size))
        self.b_f = nn.Parameter(torch.Tensor(hidden_size))

        # Cell gate
        self.W_xc = nn.Parameter(torch.Tensor(input_size, hidden_size))
        self.W_hc = nn.Parameter(torch.Tensor(hidden_size, hidden_size))
        self.b_c = nn.Parameter(torch.Tensor(hidden_size))

        # Output gate
        self.W_xo = nn.Parameter(torch.Tensor(input_size, hidden_size))
        self.W_ho = nn.Parameter(torch.Tensor(hidden_size, hidden_size))
        self.b_o = nn.Parameter(torch.Tensor(hidden_size))

        # Extended memory
        self.W_xm = nn.Parameter(torch.Tensor(input_size, hidden_size))
        self.W_hm = nn.Parameter(torch.Tensor(hidden_size, hidden_size))
        self.b_m = nn.Parameter(torch.Tensor(hidden_size))

        self.reset_parameters()

    def reset_parameters(self):
        for p in self.parameters():
            if p.data.ndimension() >= 2:
                nn.init.xavier_uniform_(p.data)
            else:
                nn.init.zeros_(p.data)

    def forward(self, x, hidden):
        h_prev, c_prev, m_prev = hidden

        # Input gate
        i = torch.sigmoid(x @ self.W_xi + h_prev @ self.W_hi + self.b_i)

        # Forget gate
        f = torch.sigmoid(x @ self.W_xf + h_prev @ self.W_hf + self.b_f)

        # Cell gate
        c_tilde = torch.tanh(x @ self.W_xc + h_prev @ self.W_hc + self.b_c)
        c = f * c_prev + i * c_tilde

        # Output gate
        o = torch.sigmoid(x @ self.W_xo + h_prev @ self.W_ho + self.b_o)

        # Extended memory
        m = torch.tanh(x @ self.W_xm + h_prev @ self.W_hm + self.b_m)

        # Hidden state
        h = o * torch.tanh(c) + m

        return h, c, m

class xLSTM(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, output_size):
        super(xLSTM, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers

        self.cells = nn.ModuleList([xLSTMCell(input_size if i == 0 else hidden_size, hidden_size) for i in range(num_layers)])
        self.fc = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        batch_size, seq_len, _ = x.shape

        # Initialize hidden states
        h = [torch.zeros(batch_size, self.hidden_size, device=x.device) for _ in range(self.num_layers)]
        c = [torch.zeros(batch_size, self.hidden_size, device=x.device) for _ in range(self.num_layers)]
        m = [torch.zeros(batch_size, self.hidden_size, device=x.device) for _ in range(self.num_layers)]

        outputs = []
        for t in range(seq_len):
            x_t = x[:, t, :]
            for i, cell in enumerate(self.cells):
                h[i], c[i], m[i] = cell(x_t, (h[i], c[i], m[i]))
                x_t = h[i]
            outputs.append(h[-1])

        # Stack outputs along the sequence dimension
        outputs = torch.stack(outputs, dim=1)

        # Apply the final linear layer
        outputs = self.fc(outputs)

        return outputs