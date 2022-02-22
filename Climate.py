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


COLORS = px.colors.qualitative.Dark24
CURRENT = 0


class Climate:

    def __init__(self, data, name):

        self._data = data
        self._name = name
        self._preprocess()

        self._start = self._data.index[0]
        self._end = self._data.index[-1]

    def _preprocess(self):

        self._data['x'] = np.arange(len(self._data))
        self._data.index = pd.to_datetime(self._data.dt)
        self._data.index.freq = self._data.index.inferred_freq
        self._data.drop(columns='dt', inplace=True)

    def inflection_points(self, start=None, end=None, level=None, order=3, return_idx=False):

        data = self._deriv_data(1, start, end, True, level, order).AverageTemperature.values
        indices = np.where([(data[i] - data[i-1]) * (data[i + 1] - data[i]) < 0 for i in range(1, len(data)-1)])[0]

        if return_idx:
            return indices
        else:
            return np.array([self._to_date(idx) for idx in indices])

    def correlation(self, decomposed=False, seasonal=3):
        corr = Xi(self._data.x, self.data(decomposed=decomposed,                       
                                          seasonal=seasonal).AverageTemperature)
        return dict(correlation=corr.correlation, p_value=corr.pval_asymptotic())

    def _endpoints(self, start, end):

        if not start:
            start = self._start

        if not end:
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

        model = Model(lag=lag, horizon=horizon, hidden_size=hidden_size, learning_rate=learning_rate)
        data = Temperature(self._name, lag=lag, horizon=horizon, normalize=False, size=iters, by_batch=False)
        model.fit(data.get_dataloader(), epochs)
        model.save(self._name)
        return model

    def predict(self, horizon, Model):

        try:
            model = Model().load(self._name)
        except FileNotFoundError:
            self.train(model_class)
            model = Model().load(self._name)

        preds = model.predict(horizon)
        pred_data = deepcopy(self._data.iloc[:horizon])
        pred_data.index = pd.date_range(start=self._data.index[0] + pd.DateOffset(months=1), periods=horizon, freq='M')
        pred_data['AverageTemperature'] = preds
        self._data = pd.concat([self._data, pred_data])

        return pred_data

    def data(self, start=None, end=None, smoothed=False, level=None, order=1, pred=None):

        start, end = self._endpoints(start, end)
        data = self._data.loc[start:end, :]

        return self._smooth(data, level, order) if smoothed else data

    def view(self, rows=10):

        return self.data(end=list(self._data.index)[rows-1])

    def plot(self, fig, pred=None, prime=0, start=None, end=None, year_step=10, smoothed=False, level=None, order=1, inflection=False):

        global COLORS
        global CURRENT

        data = self._deriv_data(prime, start, end, smoothed, level, order, pred)

        line = px.line(data, x='x', y='AverageTemperature').data[-1]
        line.name = self._name
        line.showlegend = True
        line.line['color'] = COLORS[CURRENT % len(COLORS)]

        if inflection:
            for point in self.inflection_points(start=start, end=end, level=level, order=order, return_idx=True):
                fig.add_vline(x=point, line_color=COLORS[CURRENT % len(COLORS)], line_width=1)

        fig.add_trace(line)
        fig.update_layout(
            xaxis=dict(
                tickmode='array',
                tickvals=np.array(data.x.values)[::12 * year_step],
                ticktext=data.year.unique()[::year_step]
            )
        )
        fig.update_xaxes(tickangle=45)

        CURRENT += 1

        return fig

    def _derivative(self, xs, ys, prime):

        h = (xs[-1] - xs[0]) / len(xs)
        centered = np.array([ys[i + 1] - ys[i - 1] for i in range(1, len(ys) - 1)])
        return centered / 2 / h if prime == 1 else self._derivative(xs, centered / 2 / h, prime=prime - 1)

    def _deriv_data(self, prime, start, end, smoothed, level, order, pred):

        data = self.data(start=start, end=end, smoothed=smoothed, level=level, order=order, pred=pred)
        if prime == 0:
            return data

        y = self._derivative(data.x.values, data.AverageTemperature.values, prime)
        data = data.iloc[prime:-prime, :]
        data['AverageTemperature'] = y
        return data


class Country(Climate):

    def __init__(self, country, filename='final_data.csv'):

        self._country = country
        super().__init__(self._load_data(filename), self._country)

    def _load_data(self, filename):

        data = pd.read_csv(filename)

        return data[data.Country == self._country][['dt', 'AverageTemperature', 'AverageTemperatureUncertainty', 'year']]


class Continent(Climate):

    def __init__(self, continent, filename='final_data.csv'):

        self._continent = continent
        super().__init__(self._load_data(filename), self._continent)

    def _load_data(self, filename):

        data = pd.read_csv(filename)
        data = data[data.Continent == self._continent]
        return self._aggregate_continent(data)

    def _aggregate_continent(self, data):

        data = data[data.Continent == self._continent].groupby('dt')[
            ['AverageTemperature', 'AverageTemperatureUncertainty', 'year']].mean().reset_index()
        data.year = data.year.astype(int)

        return data
