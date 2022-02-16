import torch
import torch.nn as nn


class LSTM(nn.Module):

    def __init__(self, lag=1, horizon=1, hidden_size=1, cell_size=1, layers=1):

        super(LSTM, self).__init__()

        self._layers = layers
        self._input_size = 1
        self._seq_len = lag
        self._hidden_size = hidden_size
        self._cell_size = cell_size
        self._output_size = horizon

        self._lstm = nn.LSTM(self._input_size, self._hidden_size, self._layers, batch_first=True)
        self._linear = nn.Linear(self._hidden_size, 1)

    def forward(self, x):

        h0 = torch.randn(self._layers, x.size(0), self._hidden_size)
        c0 = torch.randn(self._layers, x.size(0), self._cell_size)

        output, _ = self._lstm(x, (h0, c0))
        output = output[:, -self._output_size:, :]
        output = self._linear(output)

        return output