from LSTM import LSTM
from Baseline import Baseline
from Climate import Country
import argparse
import pandas as pd
import os

lag = 24 # two years
horizon = 120 # ten years
iters = 100
hidden_size = 30 # month

continent_epochs = 20
country_epochs = 10

extensions = ('pck', 'ckpt', 'json')
models = ('Baseline', 'LSTM', 'XGBoost', 'ARIMA')


warn_path = lambda Model: os.path.join(os.getcwd(), 'models', Model.__name__, 'country', 'warns.txt')


model_path = lambda Model, ext: os.path.join(os.getcwd(), 'models', Model.__name__, 'country', f'{country}.{ext}')


def train_all(Model, countries):

    for country in countries:
        if not any([os.path.exists(filename)
                    for filename in [model_path(Model, ext) for ext in extensions]]):
            try:
                country_model = Country(country)
            except ValueError as e:
                print(country, e)
            try:
                country_model.train(Model, lag=lag, horizon=horizon, hidden_size=hidden_size,
                                    epochs_main=continent_epochs, epochs=country_epochs, iters=iters)
            except:
                for ext in extensions:
                    try:
                        os.remove(model_path(Model, ext))
                    except FileNotFoundError:
                        pass
                with open(model_path(Model), 'a') as warns:
                    warns.write(f'{country}\n')


if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="Provide Model Name")
    parser.add_argument('Model', choices=models,
                        help=f"Possible values: {models}")
    args = parser.parse_args()
    Model = eval(args.Model)

    data = pd.read_csv('final_data.csv')
    countries = data.Country.unique()

    with open(warn_path(Model), 'a') as warns:
        for country in countries:
            warns.write(f'{country}\n')

    path = warn_path(Model)
    while os.path.exists(path):
        with open(path, 'r') as warns:
            countries = warns.readlines()
            countries = list(map(lambda country: country[:-1], countries))
        os.remove(path)
        train_all(Model, countries)
