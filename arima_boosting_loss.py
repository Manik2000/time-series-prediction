import numpy as np
from boosting import XGBoost
from Arima import Arima


def rmse(y, y_hat):
    return np.sqrt(np.sum((y - y_hat)**2) / len(y))


def calculate_loss(model, path, *args):
    dużo kodu
    return None



if __name__ == "__main__":
    calculate_loss(XGBoost, "", 24)
    calculate_loss(XGBoost, "")