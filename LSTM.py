import torch
import torch.nn as nn
import pytorch_lightning as pl


class LSTM(pl.LightningModule):

    def __init__(self, lag=1, horizon=1, hidden_size=1, input_size=1, learning_rate=.01):

        super(LSTM, self).__init__()

        self._input_size = input_size
        self._seq_len = lag
        self._hidden_size = hidden_size
        self._output_size = horizon
        self._learning_rate = learning_rate

        self._in_cell = nn.LSTMCell(self._input_size, self._hidden_size)
        self._out_cell = nn.LSTMCell(0, self._hidden_size)
        self._linear = nn.Linear(self._hidden_size, 1)

    def forward(self, x, horizon=None):

        hi = torch.randn(x.size(0), self._hidden_size)
        ci = torch.randn(x.size(0), self._hidden_size)

        for i in range(self._seq_len):
            hi, ci = self._in_cell(x[:, i, :], (hi, ci))

        outputs = []
        if horizon:
            for i in range(horizon):
                hi, ci = self._out_cell(torch.empty((x.size(0), 0)), (hi, ci))
                outputs.append(self._linear(hi))
        else:
            for i in range(self._output_size):
                hi, ci = self._out_cell(torch.empty((x.size(0), 0)), (hi, ci))
                outputs.append(self._linear(hi))

        return torch.stack(outputs, dim=1)

    def training_step(self, batch, batch_idx):

        x, y = batch
        y_hat = self(x)
        loss = nn.MSELoss()(y_hat, y)

        return {'loss': loss}

    def configure_optimizers(self, learning_rate=.01):

        return torch.optim.Adam(self.parameters(), lr=self._learning_rate)

    def fit(self, loader, epochs):

        trainer = pl.Trainer(max_epochs=epochs)
        trainer.fit(model=self, train_dataloaders=loader)

    def save(self, name):

        pass

    def predict(self, x, horizon):

        return self(x, horizon=horizon).detach().numpy().reshape(-1)
