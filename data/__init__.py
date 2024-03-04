from importlib import import_module

from torch.utils.data import DataLoader, Dataset

from data.META import META
from data.intellab import IntelLab


def get_dataloader(dataset, config):
    batch_size = config['data']['batch_size']

    train_dataset, val_dataset = dataset.get_train_val_data()
    train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=False)
    val_dataloader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

    test_dataset = dataset.get_test_data()
    test_dataloader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    data_loader = {'train': train_dataloader, 'val': val_dataloader, 'test': test_dataloader}

    return data_loader


class MyDataset(Dataset):
    def __init__(self, config, data=None):
        self.dataset_name = config['data']['name']
        module = import_module(f'data.{self.dataset_name}')
        self.dataset = module.make_dataset(config, data)

    def __getitem__(self, idx):
        return self.dataset[idx]

    def __len__(self):
        return len(self.dataset)

    def get_train_val_data(self):
        return self.dataset.get_train_val_data()

    def get_test_data(self):
        return self.dataset.get_test_data()
