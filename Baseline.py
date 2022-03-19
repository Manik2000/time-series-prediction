import os
import json
import numpy as np
import pandas as pd


class Baseline:

    def __init__(self, country, continent, coef=None, lag=24, horizon=120):

        self._country = country
        self._continent = continent
        self._coef = coef
        self._seq_len = lag
        self._output_size = horizon

    def fit(self, train, test):

        means = []
        for i in range(len(train) // 12):
            means.append(np.mean(train[12 * i:(i + 1) * 12]))

        coef, intercept = np.polyfit(np.arange(len(means)), np.array(means), 1)
        self._coef = coef

        loss = np.mean((self.forward(train) - test) ** 2)

        self._save(loss)

        return coef

    def forward(self, series, horizon=None):

        horizon = horizon if horizon else self._output_size

        preds = []
        for year in range(1 + horizon // 12):
            for month in range(12, 0, -1):
                preds.append(series[-month] + (year + 1) * self._coef)

        return np.array(preds[:horizon])

    def predict(self, series, horizon=None):

        return self.forward(series, horizon)

    @staticmethod
    def _get_path(country):

        return os.path.join(os.getcwd(), 'models', 'Baseline', 'country', f'{country}.json')

    def _save(self, loss):

        with open(self._get_path(self._country), 'w') as params:
            json.dump([{'continent': self._continent, 'coef': self._coef, 'horizon': self._output_size}], params)

        loss_path = os.path.join(os.getcwd(), 'loss', 'test', 'country', f'Baseline.csv')
        losses = pd.read_csv(loss_path)
        losses = losses[losses['Country'] != self._country]
        losses = pd.concat([losses, pd.DataFrame([{'Continent': self._continent, 'Country': self._country, 'Loss': loss}])])
        losses.to_csv(loss_path, index=False)

        return losses

    @classmethod
    def load(cls, country):

        try:
            with open(cls._get_path(country), 'r') as params:
                data = json.load(params)[0]
        except FileNotFoundError:
            raise NotImplementedError(f'There is no fitted model for {country}.')

        return cls(country, data['continent'], data['coef'], data['horizon'])
