import torch
from math import ceil
import numpy as np
import pandas as pd
import multiprocessing as mp
from torch.utils.data import DataLoader, Dataset
from torchvision.transforms import Normalize


class ToTensor():

    def __call__(self, sample):

        inputs, outputs = sample
        return torch.from_numpy(inputs), torch.from_numpy(outputs)


class TemperatureDataset(Dataset):

    def __init__(self, country, lag, horizon, normalize):

        data = pd.read_csv('final_data.csv').loc[:, ['dt', 'AverageTemperature', 'Country']]
        data_country = data[data['Country'] == country].sort_values('dt')
        xy = np.array(data_country.AverageTemperature, dtype=np.float32)
        records = len(xy)
        self._samples = records - horizon - lag
        self._x = np.array([xy[i:i+lag].reshape(-1, 1) for i in range(self._samples)])
        self._y = np.array([xy[i:i+horizon].reshape(-1, 1) for i in range(lag, records-horizon)])

        self._to_tensor = ToTensor() 
        self._normalize = normalize
        self._normalizer = Normalize(np.mean(xy), np.std(xy))

    def __getitem__(self, idx):

        x, y = self._to_tensor((self._x[idx], self._y[idx]))
        if self._normalize:
            x, y = self._normalizer(x), self._normalizer(y)
        
        return x, y

    def __len__(self):

        return self._samples


class Temperature:

    def __init__(self, country, lag=1, horizon=1, size=None, by_batch=True, workers=None, shuffle=True, normalize=False):

        self._dataset = TemperatureDataset(country, lag, horizon, normalize)
        self._records = len(self._dataset)
        self._workers = workers if workers else mp.cpu_count()
        self._shuffle = shuffle
        self._batch_size = self._records if not size else size if by_batch else ceil(self._records / size)
        self._iters = ceil(self._records / self._batch_size)
        self._dataloader = DataLoader(dataset=self._dataset, batch_size=self._batch_size, shuffle=self._shuffle, num_workers=self._workers)

    def head(self, rows=10):

        return self._dataset[:rows]

    def tail(self, rows=10):

        return self._dataset[-rows:]

    def get_lengths(self):

        return {'records': self._records, 'batch_size': self._batch_size, 'iterations': self._iters}

    def get_dataloader(self):

        return self._dataloader