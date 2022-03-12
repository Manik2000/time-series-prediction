import os
import numpy as np
import torch
import torch.nn as nn
import pytorch_lightning as pl


class MainLSTM(pl.LightningModule):

    def __init__(self, lag, horizon, hidden_size, input_size, learning_rate,
                 mean_temp, mean_year, mean_month, std_temp, std_year, std_month):

        super(MainLSTM, self).__init__()

        self._input_size = input_size
        self._seq_len = lag
        self._hidden_size = hidden_size
        self._output_size = horizon
        self._learning_rate = learning_rate

        self._lstm_cell = nn.LSTMCell(self._input_size, self._hidden_size)
        self._linear = nn.Linear(self._hidden_size, 1)

        self._mean_temp = mean_temp
        self._mean_year = mean_year
        self._mean_month = mean_month

        self._std_temp = std_temp
        self._std_year = std_year
        self._std_month = std_month

    def forward(self, x, horizon=None):

        hi = torch.randn(x.size(0), self._hidden_size)
        ci = torch.randn(x.size(0), self._hidden_size)

        x[:, :, 0] = x[:, :, 0] - self._mean_temp
        x[:, :, 1] = (x[:, :, 1] - self._mean_year) / self._std_year
        x[:, :, 1] = (x[:, :, 2] - self._mean_month) / self._std_month
        outputs = []
        if horizon:
            for i in range(horizon // self._seq_len + 1):
                for month in range(self._seq_len):
                    hi, ci = self._lstm_cell(x[:, month+self._seq_len*i, :], (hi, ci))
                    outputs.append(self._linear(hi))
                x = torch.cat((x,
                               torch.stack((torch.cat(outputs[-self._seq_len:], dim=1),
                                            x[:, -self._seq_len:, 1] + self._seq_len // 12 / self._std_year,
                                            x[:, -self._seq_len:, 2]),
                                           dim=-1)),
                              dim=1)
            return torch.stack(outputs[:horizon], dim=1) + self._mean_temp
        else:
            for i in range(self._output_size // self._seq_len + 1):
                for month in range(self._seq_len):
                    hi, ci = self._lstm_cell(x[:, month+self._seq_len*i, :], (hi, ci))
                    outputs.append(self._linear(hi))
                x = torch.cat((x,
                               torch.stack((torch.cat(outputs[-self._seq_len:], dim=1),
                                            x[:, -self._seq_len:, 1] + self._seq_len // 12 / self._std_year,
                                            x[:, -self._seq_len:, 1]),
                                           dim=-1)),
                              dim=1)
            return torch.stack(outputs[:self._output_size], dim=1) + self._mean_temp

    def training_step(self, batch, batch_idx):

        x, y = batch
        y_hat = self(x)
        loss = nn.MSELoss()(y_hat, y)

        self.log("loss", loss)
        return loss

    def configure_optimizers(self):

        optimizer = torch.optim.Adam(self.parameters(), lr=self._learning_rate)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=1, threshold=.01)

        return {'optimizer': optimizer, 'lr_scheduler': {'scheduler': scheduler, 'monitor': 'val_loss'}}

    def validation_step(self, batch, batch_idx):

        x, y = batch
        y_hat = self(x)
        loss = nn.MSELoss()(y_hat, y)

        self.log("val_loss", loss, on_epoch=True)
        return loss

    def test_step(self, batch, batch_idx):

        x, y = batch
        y_hat = self(x)
        loss = nn.MSELoss()(y_hat, y)

        self.log('test_loss', loss)
        return loss

    def predict(self, x, year, month, horizon):

        x = torch.from_numpy(np.array(x.reshape(1, -1, 1), dtype=np.float32))
        year = torch.from_numpy(np.array(year.reshape(1, -1, 1), dtype=np.float32))
        month = torch.from_numpy(np.array(month.reshape(1, -1, 1), dtype=np.float32))
        torch.set_grad_enabled(False)
        self.eval()
        y = self(torch.cat((x, year, month), dim=2), horizon=horizon).detach().numpy()

        return y.reshape(-1)


class ContinentLSTM(MainLSTM):

    def __init__(self, continent, lag=24, horizon=120, hidden_size=30, input_size=3, learning_rate=1e-3,
                 mean_temp=0, mean_year=0, mean_month=0, std_temp=1, std_year=1, std_month=1):

        super(ContinentLSTM, self).__init__(lag, horizon, hidden_size, input_size, learning_rate,
                                            mean_temp, mean_year, mean_month, std_temp, std_year, std_month)

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

    def validation_epoch_end(self, loss):

        with open(os.path.join(os.getcwd(), 'loss', 'validation', 'continent', 'LSTM', f'{self._continent}.txt'), 'a') as results:
            results.write(f'{torch.mean(loss)}\n')

    def test_epoch_end(self, loss):

        with open(os.path.join(os.getcwd(), 'loss', 'test', 'LSTM', 'continent.csv'), 'a') as results:
            results.write(f'{self._continent},{torch.mean(loss)}\n')


class LSTM(MainLSTM):

    def __init__(self, country, continent, learning_rate=1e-4,
                 mean_temp=0, mean_year=0, mean_month=0, std_temp=1, std_year=1, std_month=1):

        self._country = country
        self._continent = continent
        continental = self._load_main()
        layers = list(continental.children())

        super(LSTM, self).__init__(continental._seq_len, continental._output_size, continental._hidden_size,
                                          continental._input_size, learning_rate,
                                          mean_temp, mean_year, mean_month, std_temp, std_year, std_month)

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

    def validation_epoch_end(self, loss):

        with open(os.path.join(os.getcwd(), 'loss', 'validation', 'country', 'LSTM', f'{self._country}.txt'), 'a') as results:
            results.write(f'{loss}\n')

    def test_epoch_end(self, loss):

        with open(os.path.join(os.getcwd(), 'loss', 'test', 'LSTM', 'country.csv'), 'a') as results:
            results.write(f'{self._continent},{self._country},{loss}\n')
