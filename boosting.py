import xgboost as xgb
import numpy as np
from copy import deepcopy


class XGBoost:
    
    def __init__(self, data, horizon, n_estimators=1000):
        self.data = data
        self.n = len(data)
        self.horizon = horizon
        self.number_of_estimators = n_estimators
        self.model = xgb.XGBRegressor(n_estimators=n_estimators)
        self.to_predict = self.create_predict(self.data)
        self.training = self.create_training(self.to_predict)
    
    def create_predict(self, data):
        k = self.n - self.horizon
        X = np.zeros((k, self.horizon+1))
        for i in range(k):
            X[i, :] = self.data[i:i+self.horizon+1]
        return X
    
    @staticmethod
    def create_training(data):
        new = deepcopy(data)
        np.random.shuffle(new)
        return new        
        
    def train(self):
        X, y = self.training[:, :-1], self.training[:, -1]
        self.model.fit(X, y)
        
    def predict(self, n_ahead):
        forecasts = np.zeros(n_ahead)
        current = np.array(self.training[-1, :][1:], ndmin=2)
        for i in range(n_ahead):
            pred = self.model.predict(current)
            forecasts[i] = pred
            current = np.array(np.hstack([current[:, 1:], np.array([pred], ndmin=2)]), ndmin=2)
        return forecasts
    
    def predict_in_sample(self):
        X = self.to_predict[1:, :-1]
        return self.model.predict(X)        
    
    def save(self, label):
        self.model.save_model(f"{label}.json")
            
    def load(self, label):
        self.model.load_model(f"{label}.json")                