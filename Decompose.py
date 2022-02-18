import numpy as np
from statsmodels.tsa.seasonal import STL


class Decompositor:

    def __init__(self, data, seasonal=3):

        self._data = data
        self._loess = STL(self._data, seasonal=seasonal).fit()
        self._season = self._loess.seasonal
        self._trend = self._loess.trend
        self._residual = self._loess.resid

    def decompose(self, season=True, trend=True):

        data = self._data
        if season:
            data -= self._season
        if trend:
            data -= self._trend
        return data

    def growth(self, order=1):

        return np.polyfit(np.arange(len(self._data)), self._data)[0]

    def get_season(self):

        return self._season

    def get_trend(self):

        return self._trend
