import os
import pmdarima
import joblib
import json
import numpy as np
import pandas as pd
from Decompose import Decompositor
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.forecasting.stl import STLForecast


class Arima:

    def __init__(self, country, continent, model=None, p_max=5, q_max=5, criterion='aic', seasonal=7, lag=24, horizon=120):

        self._country = country
        self._continent = continent

        self._seq_len = lag
        self._output_size = horizon

        self._p_max = p_max
        self._q_max = q_max
        self._criterion = criterion
        self._seasonal = seasonal

        self._model = model
    
    def _choose_model(self, data):

        model = pmdarima.auto_arima(data,
                                    max_p=self._p_max,
                                    max_q=self._q_max,
                                    information_criterion=self._criterion)
        return model
        
    def fit(self, train, test):

        decompositor = Decompositor(train, seasonal=self._seasonal)
        decomposed = decompositor.decompose()

        #self._season = decompositor.get_season()
        #self._trend = decompositor.get_trend()

        arima_model = self._choose_model(decomposed)
        self._model = STLForecast(train, ARIMA, period=12,
                                  model_kwargs=dict(order=arima_model.get_params()['order'],
                                                    trend='ct')).fit()
        loss = np.mean((self.predict() - test) ** 2)

        self._save(loss)

        return loss

    def predict(self, horizon=None):

        return self._model.forecast(horizon) if horizon else self._model.forecast(self._output_size)

    """
    def predict_in_sample(self):
        return self._model.predict_in_sample() + self._trend + self._season
    """

    @staticmethod
    def _get_path(country):

        return os.path.join(os.getcwd(), 'models', 'Arima', 'country', country)
    
    def _save(self, loss):

        if not os.path.exists(self._get_path(self._country)):
            os.mkdir(self._get_path(self._country))

        model_path = self._get_path(self._country)
        joblib.dump(self._model, os.path.join(model_path, f'{self._country}.pkl'))

        with open(os.path.join(model_path, f'{self._country}.json'), 'w') as params:
            json.dump([{'continent': self._continent, 'horizon': self._output_size}], params)

        loss_path = os.path.join(os.getcwd(), 'loss', 'test', 'country', f'Arima.csv')
        losses = pd.read_csv(loss_path)
        losses = losses[losses['Country'] != self._country]
        losses = pd.concat([losses, pd.DataFrame([{'Continent': self._continent, 'Country': self._country, 'Loss': loss}])])
        losses.to_csv(loss_path, index=False)

        return losses

    @classmethod
    def load(cls, country):

        try:
            with open(os.path.join(cls._get_path(country), f'{country}.json'), 'r') as json_file:
                params = json.load(json_file)[0]
        except FileNotFoundError:
            raise NotImplementedError(f'There is no fitted model for {country}.')

        return cls(country, params['continent'],
                   model=joblib.load(os.path.join(cls._get_path(country), f'{country}.pkl')), horizon=params['horizon'])
        