import pandas as pd
import numpy as np
import pycountry_convert as pc
import statsmodels.formula.api as sm
from statsmodels.tsa.seasonal import STL
from scipy.stats import pearsonr
from scipy.misc import derivative
from scipy.optimize import fsolve


def create_continents(filename='final_data.csv'):

    def get_continent(country):

        try:
            return pc.country_alpha2_to_continent_code(pc.country_name_to_country_alpha2(country))
        except:
            pass

    df = pd.read_csv(filename)
    df.dt = pd.to_datetime(df.dt)
    df['Continent'] = df.apply(lambda row: get_continent(row.Country), axis=1)
    df.to_csv(filename, index=False)


class Climate:

    def __init__(self, data):

        self._data = data

    def regression_coef(self):

        return sm.ols(data=self._data, formula='AverageTemperature ~ x').fit().params.x

    def inflection_point(self, x0=None, deg=3):

        xs, ys = self._data.x, self._data.AverageTemperature
        x0 = x0 if x0 else np.mean([xs[0], xs[-1]])
        coef = np.polyfit(xs, ys, deg)
        construct_polynomial = lambda coefs: np.vectorize(
            lambda x: np.dot(coefs, np.array([x ** i for i in range(len(coefs) - 1, -1, -1)])))
        return self._data.index[int(round(
            fsolve(lambda x_prime: derivative(construct_polynomial(coef), x_prime, n=2), x0)[0]))], coef[0]

    def correlation(self, decompose=True, seasonal=3):

        y = self._data.AverageTemperature
        y = y - STL(y, seasonal=seasonal).fit().seasonal if decompose else y
        return pearsonr(self._data.x, y)


class Country(Climate):

    def __init__(self, country, filename='final_data.csv'):

        self._country = country
        self._df = pd.read_csv(filename)
        self._df = df[df.Country == self._country]
        self._df['x'] = np.arange(len(self._df))
        super().__init__(df)


class Continent(Climate):

    def __init__(self, continent, filename='final_data.csv'):

        self._continent = continent
        self._df = pd.read_csv(filename)
        self._df = df[df.Continent == self._continent]
        self._continent_data()
        self._df['x'] = np.arange(len(self._df))
        super().__init__(self._df)

    def _continent_data(self):

        self._df = self._df[self._df.Continent == self._continent].groupby('dt')[
            ['AverageTemperature', 'AverageTemperatureUncertainty', 'year']].mean()
        self._df.year = self._df.year.astype(int)

        return self._df


if __name__ == '__main__':

    create_continents()
