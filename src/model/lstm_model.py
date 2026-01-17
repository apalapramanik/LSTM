import torch
import torch.nn as nn
from .lstm_cell import LSTMCell


class LSTMLanguageModel(nn.Module):
    """
    Character-level LSTM language model.

    Architecture:
    tokens → embedding → LSTM → linear head
    """

    def __init__(
        self,
        vocab_size,
        embed_dim,
        hidden_dim,
        num_layers,
    ):
        super().__init__()

        self.embed = nn.Embedding(vocab_size, embed_dim)

        self.layers = nn.ModuleList(
            [
                LSTMCell(
                    embed_dim if i == 0 else hidden_dim,
                    hidden_dim
                )
                for i in range(num_layers)
            ]
        )

        self.output_head = nn.Linear(hidden_dim, vocab_size)

        self.hidden_dim = hidden_dim
        self.num_layers = num_layers

    def forward(self, x, mask=None):
        """
        x : (B, T)
        mask : ignored (kept for API compatibility)
        """

        B, T = x.size()
        x = self.embed(x)

        # Initialize hidden & cell states
        h = [
            torch.zeros(B, self.hidden_dim, device=x.device)
            for _ in range(self.num_layers)
        ]
        c = [
            torch.zeros(B, self.hidden_dim, device=x.device)
            for _ in range(self.num_layers)
        ]

        outputs = []

        for t in range(T):
            inp = x[:, t]

            for layer_idx, layer in enumerate(self.layers):
                h[layer_idx], c[layer_idx] = layer(
                    inp, h[layer_idx], c[layer_idx]
                )
                inp = h[layer_idx]

            outputs.append(inp.unsqueeze(1))

        outputs = torch.cat(outputs, dim=1)
        logits = self.output_head(outputs)

        return logits
