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

    def __init__(self, country, lag, horizon, normalize, start=0, size=1):

        data = pd.read_csv('final_data.csv').loc[:, ['dt', 'AverageTemperature', 'Country']]
        data_country = data[data['Country'] == country].sort_values('dt')
        data_country = data_country.iloc[int(start*len(data_country)):int((start+size)*len(data_country))]
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

    def __init__(self, name, lag=1, horizon=1, size=None, by_batch=True, workers=None, normalize=False, val_size=.2, test_size=.1):

        self._name = name
        self._lag = lag
        self._horizon = horizon
        self._normalize = normalize

        self._workers = workers if workers else mp.cpu_count()
        self._size = size
        self._by_batch = by_batch

        self._train_size, self._train = self._create_dataloader(0, 1-val_size-test_size, True)
        self._val_size, self._val = self._create_dataloader(1-val_size-test_size, val_size, False)
        self._test_size, self._test = self._create_dataloader(1-test_size, test_size, False)

    def _create_dataloader(self, start, size, shuffle):

        dataset = TemperatureDataset(self._name, self._lag, self._horizon, self._normalize, start=start, size=size)
        batch_size = len(dataset) if not self._size else self._size if self._by_batch else ceil(len(dataset) / self._size)
        return len(dataset), DataLoader(dataset=dataset, batch_size=batch_size, shuffle=shuffle, num_workers=self._workers)

    def head(self, rows=10):

        return self._dataset[:rows]

    def tail(self, rows=10):

        return self._dataset[-rows:]

    def get_lengths(self):

        return {'train': self._train_size, 'val': self._val_size, 'test': self._test_size}

    def get_dataloaders(self):

        return self._train, self._val, self._test
