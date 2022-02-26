import torch
import torch.nn as nn
import pytorch_lightning as pl


class CountryLSTM(pl.LightningModule):

    def __init__(self, lag=1, horizon=1, hidden_size=1, input_size=1, learning_rate=1e-2):

        super(CountryLSTM, self).__init__()

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

    def fit(self, train, val, test, epochs):

        trainer = pl.Trainer(auto_lr_find=True, max_epochs=epochs)
        trainer.fit(model=self, train_dataloaders=train, val_dataloaders=val)
        trainer.test(dataloaders=test)

    def save(self, name):

        pass

    def predict(self, x, horizon):

        torch.set_grad_enabled(False)
        self.eval()
        return self(x, horizon=horizon).detach().numpy().reshape(-1)


class ContinentLSTM(pl.LightningModule):

    def __init__(self, lag=1, horizon=1, hidden_size=1, input_size=1, learning_rate=1e-2):

        super(ContinentLSTM, self).__init__()

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

    def fit(self, train, val, test, epochs):

        trainer = pl.Trainer(auto_lr_find=True, max_epochs=epochs)
        trainer.fit(model=self, train_dataloaders=train, val_dataloaders=val)
        trainer.test(dataloaders=test)

    def save(self, name):

        pass

    def predict(self, x, horizon):

        torch.set_grad_enabled(False)
        self.eval()
        return self(x, horizon=horizon).detach().numpy().reshape(-1)
