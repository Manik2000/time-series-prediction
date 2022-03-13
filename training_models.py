import os
import pandas as pd
from boosting import XGBoost
from Arima import Arima


current = os.getcwd()

def trainer(model, model_name, *args):
    if not os.path.isdir(rf"models\{model_name}"):
        os.makedirs(rf"models/{model_name}")
    data = pd.read_csv("final_data.csv")
    countries = data['Country'].unique()
    for i in countries:
        x = data[data['Country'] == i]["AverageTemperature"].to_numpy()
        model_ = model(x, *args)
        model_.train()
        model_.save(f"models/{model_name}/{i}")


if __name__ == "__main__":
    trainer(Arima, "Arima")
    trainer(XGBoost, "Boosting", 24)
