import pmdarima
from Decompose import Decompositor
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.forecasting.stl import STLForecast
from copy import deepcopy


class Arima:

    def __init__(self, data, p_max=5, q_max=5, criterion='aic', seasonal=7):
        self._data = deepcopy(data)
        self._p_max = p_max
        self._q_max = q_max
        self._criterion = criterion

        decompositor = Decompositor(self._data, seasonal=seasonal)
        self._decomposed = decompositor.decompose()
        self._season = decompositor.get_season()
        self._trend = decompositor.get_trend()

        self._p = 0
        self._q = 0
        self._model = None
        self._choose_model()

        self._forecast = STLForecast(self._data, ARIMA, period=12, 
                                     model_kwargs=dict(order=(self._p, 0, self._q),
                                                       trend='ct')).fit()

    def _choose_model(self):
        model = pmdarima.auto_arima(self._decomposed, 
                                    max_p = self._p_max,
                                    max_q = self._q_max,
                                    information_criterion=self._criterion)
        order = model.get_params()['order']
        self._p, self._q = order[0], order[2]
        self._model = model

    def predict(self, horizon):
        return self._forecast.forecast(horizon)