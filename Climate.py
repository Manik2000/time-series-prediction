from Decompose import Decompositor
import pandas as pd
import numpy as np
import seaborn as sns
from copy import deepcopy
import statsmodels.formula.api as sm
from statsmodels.tsa.seasonal import STL
from scipy.stats import pearsonr
from scipy.misc import derivative
from scipy.optimize import fsolve


class Climate:

    def __init__(self, data):

        self._data = data
        self._preprocess()

        self._start = self._data.index[0]
        self._end = self._data.index[-1]

    def _preprocess(self):

        self._data['x'] = np.arange(len(self._data))
        self._data.index = pd.to_datetime(self._data.dt)
        self._data.index.freq = self._data.index.inferred_freq
        self._data.drop(columns='dt', inplace=True)

    def regression_coef(self):

        return sm.ols(data=self._data, formula='AverageTemperature ~ x').fit().params.x

    def inflection_point(self, x0=None, deg=3):

        xs, ys = np.array(self._data.x.values), np.array(self._data.AverageTemperature.values)
        x0 = x0 if x0 else np.mean([xs[0], xs[-1]])
        coef = np.polyfit(xs, ys, deg)
        construct_polynomial = lambda coefs: np.vectorize(
            lambda x: np.dot(coefs, np.array([x ** i for i in range(len(coefs) - 1, -1, -1)])))
        return self._data.index[int(round(
            fsolve(lambda x_prime: derivative(construct_polynomial(coef), x_prime, n=2), x0)[0]))], coef[0]

    def correlation(self, decomposed=False, seasonal=3):

        return pearsonr(self._data.x, self.data(decomposed=decomposed, seasonal=seasonal).AverageTemperature)

    def _endpoints(self, start, end):

        if not start:
            start = self._start

        if not end:
            end = self._end

        return start, end

    @staticmethod
    def _decompose(data, seasonal):

        copied = deepcopy(data)
        Decompositor(copied.AverageTemperature, seasonal=seasonal).decompose(trend=False)

        return copied

    def data(self, start=None, end=None, decomposed=False, seasonal=3):

        start, end = self._endpoints(start, end)
        data = self._data.loc[start:end, :]

        return self._decompose(data, seasonal) if decomposed else data

    def view(self, rows=10):

        return self.data(end=list(self._data.index)[rows-1])

    def plot(self, start=None, end=None, year_step=10, decomposed=False, seasonal=3):

        data = self.data(start=start, end=end, decomposed=decomposed, seasonal=seasonal)
        g = sns.lineplot(data=data, x='x', y='AverageTemperature')
        g.set_xticks(np.array(data.x.values)[::12 * year_step])
        g.set_xticklabels(data.year.unique()[::year_step], rotation=45)
        return g

    def plot_regression(self, start=None, end=None, year_step=10, decomposed=False, seasonal=3):

        data = self.data(start=start, end=end, decomposed=decomposed, seasonal=seasonal)
        g = sns.lmplot(data=data, x='x', y='AverageTemperature', lowess=True, scatter=False)
        g.axes.flat[0].set_xticks(np.array(data.x.values)[::12 * year_step])
        g.axes.flat[0].set_xticklabels(data.year.unique()[::year_step], rotation=45)
        return g


class Country(Climate):

    def __init__(self, country, filename='final_data.csv'):

        self._country = country
        super().__init__(self._load_data(filename))

    def _load_data(self, filename):

        data = pd.read_csv(filename)

        return data[data.Country == self._country][['dt', 'AverageTemperature', 'AverageTemperatureUncertainty', 'year']]


class Continent(Climate):

    def __init__(self, continent, filename='final_data.csv'):

        self._continent = continent
        super().__init__(self._load_data(filename))

    def _load_data(self, filename):

        data = pd.read_csv(filename)
        data = data[data.Continent == self._continent]
        return self._aggregate_continent(data)

    def _aggregate_continent(self, data):

        data = data[data.Continent == self._continent].groupby('dt')[
            ['AverageTemperature', 'AverageTemperatureUncertainty', 'year']].mean().reset_index()
        data.year = data.year.astype(int)

        return data
