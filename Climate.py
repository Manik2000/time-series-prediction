from Smooth import Smoother
from DataLoader import Temperature
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
import statsmodels.formula.api as sm
from xicor.xicor import Xi
from copy import deepcopy
from statsmodels.tsa.seasonal import STL
from scipy.stats import pearsonr
from scipy.misc import derivative
from scipy.optimize import fsolve
from datetime import date
from LSTM import ContinentLSTM


#COLORS = px.colors.qualitative.Dark24
#CURRENT = 0
CONTINENTS = ['Asia', 'Europe', 'South America', 'Africa', 'North America', 'Oceania']


class Climate:

    def __init__(self, data, name):

        self._data = data
        self._name = name
        self._preprocess()

        self._mean_temp = np.mean(self._data.AverageTemperature)
        self._mean_year = np.mean(self._data.year)
        self._mean_month = np.mean(self._data.month)

        self._std_temp = np.std(self._data.AverageTemperature)
        self._std_year = np.std(self._data.year)
        self._std_month = np.std(self._data.month)

        self._start = self._data.index[0]
        self._end = self._data.index[-1]

    def _preprocess(self):

        self._data['x'] = np.arange(len(self._data))
        self._data.index = pd.to_datetime(self._data.dt)
        self._data.index.freq = self._data.index.inferred_freq
        self._data.drop(columns='dt', inplace=True)

    def inflection_points(self, level=None, order=3, return_idx=False, eps=1e-9):

        data = self._deriv_data(1, None, None, True, level, order).AverageTemperature.values
        indices = np.argsort([(data[i] - data[i - 1]) * (data[i + 1] - data[i])
                              for i in range(1, len(data) - 1)])[:np.max((0, order-2))]

        if return_idx:
            return indices
        else:
            return np.sort(np.array([self._to_date(idx) for idx in indices]))

    def correlation(self, smoothed=False):

        corr = Xi(self._data.x, self.data(smoothed=smoothed).AverageTemperature)
        return dict(correlation=corr.correlation, p_value=corr.pval_asymptotic())

    def _endpoints(self, start, end):

        if start:
            start = np.max((self._start, date(int(start), 1, 1)))
        else:
            start = self._start

        if end:
            end = date(int(end), 12, 1)
        else:
            end = self._end

        return start, end

    def _to_date(self, idx, start=None):

        start = start if start else self._data.index[0]

        return start + pd.DateOffset(months=idx)

    @staticmethod
    def _smooth(data, level, order):

        copied = deepcopy(data)
        copied['AverageTemperature'] = Smoother(copied).smooth(level, order)

        return copied

    def train(self, Model, lag=12, horizon=12, hidden_size=10, learning_rate=1e-2, epochs=10, iters=100):

        data = Temperature(self._name, lag=lag, horizon=horizon, normalize=True, size=iters, by_batch=False)
        model = Model(self._name, lag=lag, horizon=horizon, hidden_size=hidden_size, learning_rate=learning_rate,
                      mean_temp=self._mean_temp, mean_year=self._mean_year, mean_month=self._mean_month,
                      std_temp=self._std_temp, std_year=self._std_year, std_month=self._std_month)
        model.fit(*data.get_dataloaders(), epochs)

        return model

    def predict(self, Model, horizon):

        model = Model.load(self._name)

        preds = model.predict(self._data.AverageTemperature.values[-model._seq_len:],
                              self._data.year.values[-model._seq_len:],
                              self._data.month.values[-model._seq_len:],
                              horizon)
        pred_data = deepcopy(self._data.iloc[:horizon])
        pred_data.index = pd.date_range(start=self._end + pd.DateOffset(months=1), periods=horizon, freq='M')
        pred_data['year'] = pred_data.index.year
        pred_data['month'] = pred_data.index.month
        pred_data['x'] += len(self._data)
        pred_data['AverageTemperature'] = preds

        return pred_data

    def data(self, Model=None, start=None, end=None, smoothed=False, level=None, order=1):

        start, end = self._endpoints(start, end)
        data = self._data
        horizon = int((end.year - self._end.year) * 12 + end.month - self._end.month)
        if Model and horizon > 0:
            data = pd.concat([data, self.predict(Model, horizon)])

        return self._smooth(data, level, order).loc[start:end, :] if smoothed else data.loc[start:end, :]

    def view(self, rows=10):

        return self.data(end=list(self._data.index)[rows-1])

    def plot(self, fig, Model, pred=None, prime=0, start=None, end=None, year_step=10, smoothed=False, level=None, order=1, inflection=False):

        global COLORS
        global CURRENT

        data = self._deriv_data(prime, start, end, smoothed, level, order, Model=Model)

        line = px.line(data[:self._end+pd.DateOffset(months=1)], x='x', y='AverageTemperature').data[-1]
        line.name = self._name
        line.showlegend = True
        line.line['color'] = 'blue'

        pred_line = px.line(data[self._end+pd.DateOffset(months=1):], x='x', y='AverageTemperature').data[-1]
        pred_line.name = 'Prediction'
        pred_line.showlegend = True
        pred_line.line['color'] = 'red'

        if inflection:
            for point in self.inflection_points(level=level, order=order, return_idx=True):
                fig.add_vline(x=point,
                              #line_color=COLORS[CURRENT % len(COLORS)],
                              line_width=1)

        fig.add_trace(line)
        fig.add_trace(pred_line)
        fig.update_layout(
            xaxis=dict(
                tickmode='array',
                tickvals=np.array(data.x.values)[::12 * year_step],
                ticktext=data.year.unique()[::year_step]
            )
        )
        fig.update_xaxes(tickangle=45)

        #CURRENT += 1

        return fig

    def _derivative(self, xs, ys, prime):

        h = (xs[-1] - xs[0]) / len(xs)
        centered = np.array([ys[i + 1] - ys[i - 1] for i in range(1, len(ys) - 1)])
        return centered / 2 / h if prime == 1 else self._derivative(xs, centered / 2 / h, prime=prime - 1)

    def _deriv_data(self, prime, start, end, smoothed, level, order, Model=None):

        data = self.data(Model=Model, start=start, end=end, smoothed=smoothed, level=level, order=order)
        if prime == 0:
            return data

        y = self._derivative(data.x.values, data.AverageTemperature.values, prime)
        data = data.iloc[prime:-prime, :]
        data['AverageTemperature'] = y
        return data


class Continent(Climate):

    def __init__(self, continent, filename='final_data.csv'):

        self._continent = continent
        super().__init__(self._load_data(filename), self._continent)

    def _load_data(self, filename):

        data = pd.read_csv(filename)
        data = data[data.Country == self._continent][['dt', 'AverageTemperature', 'year', 'month']]

        if len(data) > 0:
            if self._continent not in CONTINENTS:
                raise ValueError(f'{self._continent} is a country. Choose Country class instead.')
            else:
                return data
        else:
            raise ValueError(f'There is no such a continent {self._continent}.')


class Country(Climate):

    def __init__(self, country, filename='final_data.csv'):

        self._country = country
        self._continent = None
        super().__init__(self._load_data(filename), self._country)

    def _load_data(self, filename):

        global CONTINENTS

        data = pd.read_csv(filename)
        data = data[data.Country == self._country]
        self._continent = data.Continent.values[0]
        data = data[['dt', 'AverageTemperature', 'year', 'month']]

        if len(data) > 0:
            if self._country in CONTINENTS:
                raise ValueError(f'{self._country} is a continent. Choose Continent class instead.')
            else:
                return data
        else:
            raise ValueError(f'There is no such a country {self._country}.')

    def train(self, Model, ModelMain=ContinentLSTM, lag=12, horizon=12, hidden_size=10, learning_rate=1e-2, epochs_main=20, epochs=10, iters=100):

        data = Temperature(self._country, lag=lag, horizon=horizon, normalize=True, size=iters, by_batch=False)
        try:
            model = Model(self._country, self._continent, learning_rate=learning_rate,
                          mean_temp=self._mean_temp, mean_year=self._mean_year, mean_month=self._mean_month,
                          std_temp=self._std_temp, std_year=self._std_year, std_month=self._std_month)
        except NotImplementedError:
            continent = Continent(self._continent)
            continent.train(ModelMain,
                            lag=lag, horizon=horizon, hidden_size=hidden_size,
                            epochs=epochs_main, iters=iters)
            model = Model(self._name, self._continent, learning_rate=learning_rate,
                          mean_temp=self._mean_temp, mean_year=self._mean_year, mean_month=self._mean_month,
                          std_temp=self._std_temp, std_year=self._std_year, std_month=self._std_month)

        model.fit(*data.get_dataloaders(), epochs)

        return model
