"""
LSTM Language Model - LSTM Cell Module

Author: Apala Pramanik
Description: LSTM cell implementation from scratch with explicit gate computations.
"""

import torch
import torch.nn as nn


class LSTMCell(nn.Module):
    """
    Single LSTM cell implemented from scratch.

    Gates:
    - input gate
    - forget gate
    - output gate
    - candidate cell state
    """

    def __init__(self, input_dim, hidden_dim):
        super().__init__()
        self.hidden_dim = hidden_dim

        # Combine all gates into one linear layer for efficiency
        self.linear = nn.Linear(
            input_dim + hidden_dim,
            4 * hidden_dim
        )

    def forward(self, x, h, c):
        """
        x : (B, input_dim)
        h : (B, hidden_dim)
        c : (B, hidden_dim)
        """

        combined = torch.cat([x, h], dim=1)
        gates = self.linear(combined)

        # Split gates
        i, f, o, g = gates.chunk(4, dim=1)

        i = torch.sigmoid(i)          # input gate
        f = torch.sigmoid(f)          # forget gate
        o = torch.sigmoid(o)          # output gate
        g = torch.tanh(g)             # candidate

        c_next = f * c + i * g
        h_next = o * torch.tanh(c_next)

        return h_next, c_next
