import os
import json
import xgboost as xgb
import numpy as np
import pandas as pd


class XGBoost:
    
    def __init__(self, country, continent, lag=24, horizon=120, n_estimators=1000):

        self._country = country
        self._continent = continent

        self._seq_len = lag
        self._output_size = horizon
        self._n_estimators = n_estimators
        self.model = xgb.XGBRegressor(n_estimators=n_estimators)
    
    def reshape_data(self, data):

        samples = len(data) - self._output_size - self._seq_len
        x = np.array([data[i:i + self._seq_len] for i in range(samples)])
        y = np.array([data[i:i + self._output_size] for i in range(self._seq_len, samples + self._seq_len)])

        return x, y

    """
    @staticmethod
    def create_training(data):
        new = deepcopy(data)
        np.random.shuffle(new)
        return new        
    """

    def fit(self, train, test, test_size=.1):

        data = np.concatenate((train, test))
        train, test = data[:int((1-test_size)*len(data))], data[int((1-test_size)*len(data)):]
        X_train, y_train = self.reshape_data(train)
        self.model.fit(X_train, y_train[:, 0])

        X_test, y_test = self.reshape_data(test)
        loss = np.mean((np.apply_along_axis(self.predict, -1, X_test) - y_test) ** 2)

        self._save(loss)

        return loss
        
    def predict(self, series, horizon=None):

        forecasts = []
        current = np.array(series[-self._seq_len:], ndmin=2)
        horizon = horizon if horizon else self._output_size
        for i in range(horizon):
            pred = self.model.predict(current)
            forecasts.append(pred)
            current = np.array(np.hstack([current[:, 1:], np.array([pred], ndmin=2)]), ndmin=2)

        return np.array(forecasts)[:, -1]

    """
    def predict_in_sample(self):

        X = self.to_predict[1:, :-1]
        return self.model.predict(X)
    """

    @staticmethod
    def _get_path(country):

        return os.path.join(os.getcwd(), 'models', 'XGBoost', 'country', country)

    def _save(self, loss):

        if not os.path.exists(self._get_path(self._country)):
            os.mkdir(self._get_path(self._country))

        model_path = self._get_path(self._country)
        self.model.save_model(os.path.join(model_path, f'{self._country}.json'))

        with open(os.path.join(model_path, f'{self._country}_params.json'), 'w') as params:
            json.dump([{'continent': self._continent, 'horizon': self._output_size}], params)

        loss_path = os.path.join(os.getcwd(), 'loss', 'test', 'country', f'XGBoost.csv')
        losses = pd.read_csv(loss_path)
        losses = losses[losses['Country'] != self._country]
        losses = pd.concat([losses, pd.DataFrame([{'Continent': self._continent, 'Country': self._country, 'Loss': loss}])])
        losses.to_csv(loss_path, index=False)

        return losses

    @classmethod
    def load(cls, country):

        try:
            with open(os.path.join(cls._get_path(country), f'{country}_params.json'), 'r') as json_file:
                params = json.load(json_file)[0]
        except FileNotFoundError:
            raise NotImplementedError(f'There is no fitted model for {country}.')

        model_class = cls(country, params['continent'], horizon=params['horizon'])
        model_class.model.load_model(os.path.join(cls._get_path(country), f'{country}.json'))

        return model_class
