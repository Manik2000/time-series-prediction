import os
import numpy as np
import torch
import torch.nn as nn
import pytorch_lightning as pl


class LSTM(pl.LightningModule):

    def __init__(self, lag, horizon, hidden_size, input_size, learning_rate):

        super(LSTM, self).__init__()

        self._input_size = input_size
        self._seq_len = lag
        self._hidden_size = hidden_size
        self._output_size = horizon
        self._learning_rate = learning_rate

        self._lstm_cell = nn.LSTMCell(self._input_size, self._hidden_size)
        self._linear = nn.Linear(self._hidden_size, 1)

    def forward(self, x, horizon=None):

        hi = torch.zeros(x.size(0), self._hidden_size)
        ci = torch.zeros(x.size(0), self._hidden_size)

        outputs = []
        if horizon:
            for _ in range(horizon // self._seq_len + 1):
                for month in range(self._seq_len):
                    hi, ci = self._lstm_cell(x[:, month, :], (hi, ci))
                    outputs.append(self._linear(hi))
                x = torch.stack(outputs[-self._seq_len:], dim=1)
            return torch.stack(outputs[:horizon], dim=1)
        else:
            for i in range(self._output_size // self._seq_len + 1):
                for month in range(self._seq_len):
                    hi, ci = self._lstm_cell(x[:, month, :], (hi, ci))
                    outputs.append(self._linear(hi))
                x = torch.stack(outputs[-self._seq_len:], dim=1)
            return torch.stack(outputs[:self._output_size], dim=1)

    def forwardi(self, x, horizon=None):

        hi = torch.zeros(x.size(0), self._hidden_size)
        ci = torch.zeros(x.size(0), self._hidden_size)

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

        self.log("loss", loss)
        return loss

    def configure_optimizers(self):

        optimizer = torch.optim.Adam(self.parameters(), lr=self._learning_rate)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer)

        return {'optimizer': optimizer, 'lr_scheduler': {'scheduler': scheduler, 'monitor': 'val_loss'}}

    def validation_step(self, batch, batch_idx):

        x, y = batch
        y_hat = self(x)
        loss = nn.MSELoss()(y_hat, y)

        self.log("val_loss", loss, on_epoch=True)

    def test_step(self, batch, batch_idx):

        x, y = batch
        y_hat = self(x)
        loss = nn.MSELoss()(y_hat, y)

        self.log('test_loss', loss)

    def predict(self, x, horizon):

        x = torch.from_numpy(np.array(x.reshape(1, -1, 1), dtype=np.float32))
        torch.set_grad_enabled(False)
        self.eval()

        return self(x, horizon=horizon).detach().numpy().reshape(-1)


class ContinentLSTM(LSTM):

    def __init__(self, continent, lag=1, horizon=1, hidden_size=1, input_size=1, learning_rate=1e-2):

        super(ContinentLSTM, self).__init__(lag, horizon, hidden_size, input_size, learning_rate)

        self._continent = continent
        self.save_hyperparameters()

    def fit(self, train, val, test, epochs):

        path = os.path.join(os.getcwd(), 'models', 'LSTM', 'continent')
        monitor = pl.callbacks.ModelCheckpoint(monitor='val_loss', dirpath=path, filename=self._continent,
                                               save_top_k=1, mode="min")

        if os.path.exists(os.path.join(
                os.getcwd(), 'models', 'LSTM', 'continent', f'{self._continent}.ckpt')):
            os.remove(os.path.join(
                os.getcwd(), 'models', 'LSTM', 'continent', f'{self._continent}.ckpt'))
        trainer = pl.Trainer(auto_lr_find=True, max_epochs=epochs, callbacks=[monitor])
        trainer.fit(model=self, train_dataloaders=train, val_dataloaders=val)
        trainer.test(dataloaders=test)

        return trainer

    @classmethod
    def load(cls, continent):

        try:
            return cls.load_from_checkpoint(os.path.join(
                os.getcwd(), 'models', 'LSTM', 'continent', f'{continent}.ckpt'))
        except FileNotFoundError:
            raise NotImplementedError(f'There is no fitted model for {continent}.')


class CountryLSTM(LSTM):

    def __init__(self, country, continent, learning_rate=1e-2):

        self._country = country
        self._continent = continent
        continental = self._load_main()
        layers = list(continental.children())

        super(CountryLSTM, self).__init__(continental._seq_len, continental._output_size, continental._hidden_size,
                                          continental._input_size, learning_rate)

        self.save_hyperparameters()

        self._lstm_cell = layers[0]

        self._lstm_cell.eval()
        for param in self._lstm_cell.parameters():
            param.requires_grad = False

        self._linear = layers[-1]

    def _load_main(self):

        try:
            return ContinentLSTM.load_from_checkpoint(
                os.path.join(os.getcwd(), 'models', 'LSTM', 'continent', f'{self._continent}.ckpt'))
        except FileNotFoundError:
            raise NotImplementedError("Cannot load LSTM for transfer learning.")

    def fit(self, train, val, test, epochs):

        path = os.path.join(os.getcwd(), 'models', 'LSTM', 'country')
        monitor = pl.callbacks.ModelCheckpoint(monitor='val_loss', dirpath=path, filename=self._country,
                                               save_top_k=1, mode="min")

        if os.path.exists(os.path.join(
                os.getcwd(), 'models', 'LSTM', 'country', f'{self._country}.ckpt')):
            os.remove(os.path.join(
                os.getcwd(), 'models', 'LSTM', 'country', f'{self._country}.ckpt'))
        trainer = pl.Trainer(auto_lr_find=True, max_epochs=epochs, callbacks=[monitor])
        trainer.fit(model=self, train_dataloaders=train, val_dataloaders=val)
        trainer.test(dataloaders=test)

        return trainer

    @classmethod
    def load(cls, country):

        try:
            return cls.load_from_checkpoint(os.path.join(
                os.getcwd(), 'models', 'LSTM', 'country', f'{country}.ckpt'))
        except FileNotFoundError:
            raise NotImplementedError(f'There is no fitted model for {country}.')
