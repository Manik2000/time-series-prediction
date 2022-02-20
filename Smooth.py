import numpy as np
from statsmodels.nonparametric.smoothers_lowess import lowess


class Smoother:

    def __init__(self, data):

        self._data = data

    def _poly(self, order):

        return np.polyval(np.polyfit(self._data.x, self._data.AverageTemperature, order), self._data.x)

    def _lowess(self, level):

        return lowess(self._data.AverageTemperature, self._data.x, level)[:, 1]

    def smooth(self, level=None, order=1):

        if level:
            return self._lowess(level)

        else:
            return self._poly(order)
    