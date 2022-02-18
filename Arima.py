from utils import Decompositor
import numpy as np
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.forecasting.stl import STLForecast


class Arima:

    def __init__(self, data, p_max=5, q_max=5, criterion='aic', seasonal=3):

        self._data = data
        self._data.index.freq = self._data.index.inferred_freq
        self._p_max = p_max
        self._q_max = q_max
        self._criterion = criterion

        decompositor = Decompositor(self._data, seasonal=seasonal)
        self._decomposed = decompositor.decomposed()
        self._season = decompositor.get_season()
        self._trend = decompositor.get_trend()

        self._p = 0
        self._q = 0
        self._model = None
        self._choose_model()

        self._forecast = STLForecast(self._data, ARIMA, seasonal=seasonal,
                                     model_kwargs=dict(order=(self._p, 0, self._q), trend='ct')).fit()

    def _choose_model(self):

        criterion_value = np.Inf
        actual_criterion = criterion_value
        for p in range(self._p_max + 1):
            for q in range(self._q_max + 1):
                model = ARIMA(self._decomposed, order=(p, 0, q)).fit()
                try:
                    actual_criterion = eval(f'model.{self._criterion}')
                except:
                    raise ValueError('Wrong criterion.')

                if actual_criterion < criterion_value:
                    self._p = p
                    self._q = q
                    self._model = model
                    criterion_value = actual_criterion

        return self._model

    def predict(self, horizon):

        return self._forecast.forecast(horizon)
