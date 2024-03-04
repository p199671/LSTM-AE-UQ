import pandas as pd
import torch
from torch.utils.data import Dataset

FILE_TRAIN = f'train_no_anomaly.csv'
FILE_TEST = f'test_no_anomaly.csv'


def make_dataset(config, data):
    return Sine(config, data)


class Sine(Dataset):
    def __init__(self, config, data=None):
        self.config = config
        self.data = data
        if data is not None:
            features = config['data']['features']
            self.data = self.data[features]
        self.seq_length = config['data']['seq_len']
        if config['environment']['is_train']:
            self.step_size = config['data']['step_size']
        else:
            self.step_size = config['test']['step_size']

    def __len__(self):
        return (len(self.data) - self.seq_length) // self.step_size + 1

    def __getitem__(self, idx):
        start_idx = idx * self.step_size
        end_idx = start_idx + self.seq_length

        sequence = self.data[start_idx:end_idx].values.astype('float32')

        return torch.Tensor(sequence)

    def get_train_val_data(self):
        data = pd.read_csv(f'{self.config["data"]["dir"]}/{FILE_TRAIN}')
        train_data = data[:800]
        val_data = data[800:1000]
        return Sine(self.config, data=train_data), Sine(self.config, data=val_data)

    def get_test_data(self):
        test_data = pd.read_csv(f'{self.config["data"]["dir"]}/{FILE_TEST}')
        return Sine(self.config, data=test_data)
