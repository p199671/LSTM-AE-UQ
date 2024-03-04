from unittest import TestCase

import torch

import data
from configs.config import load_configs

CONFIG_FILE = '../configs/config.yml'
configs = load_configs(CONFIG_FILE)
configs['data']['dir'] = "../data/"
configs['data']['name'] = "intellab"
configs['data']['batch_size'] = 2
configs['data']['seq_len'] = 3
configs['data']['features'] = ['epoch', 'temperature', 'humidity', 'light', 'voltage']


class TestIntelLab(TestCase):
    dataset = data.MyDataset(configs)
    data_loader = data.get_dataloader(dataset.dataset, configs)

    def test_loaded_intellab_dataset_is_intellab(self, dataset=dataset):
        type_dataset = type(dataset.dataset)
        assert dataset.dataset_name == 'intellab'
        assert type_dataset == data.intellab.IntelLab

    def test_first_item_of_dataset(self, dataset=dataset):
        first_item = dataset[0]
        sample = torch.tensor([[1077929956.02785, 3, 19.9884, 37.0933, 45.08, 2.69964],
                               [1077930376.01345, 11, 19.3024, 38.4629, 45.08, 2.68742],
                               [1077930376.013453, 17, 19.1652, 38.8039, 45.08, 2.68742]])
        assert torch.all(sample.eq(first_item))

    def test_train_loader_not_none(self, data_loader=data_loader):
        train_loader = data_loader['train']
        assert train_loader is not None

    def test_val_loader_not_none(self, data_loader=data_loader):
        val_loader = data_loader['val']
        assert val_loader is not None

    def test_test_loader_not_none(self, dataset=dataset):
        local_configs = configs.copy()
        local_configs['environment']['is_train'] = False

        test_loader = data.get_dataloader(dataset, local_configs)['test']

        assert test_loader is not None

    def test_first_batch_of_train_loader(self, train_loader=data_loader['train']):
        train_feature = next(iter(train_loader))
        true_values = torch.Tensor(
            [[2, 122.153, -3.91901, 11.04, 2.03397],
             [3, 19.9884, 37.0933, 45.08, 2.69964],
             [11, 19.3024, 38.4629, 45.08, 2.68742],
             [17, 19.1652, 38.8039, 45.08, 2.68742],
             [18, 19.175, 38.8379, 45.08, 2.69964],
             [22, 19.1456, 38.9401, 45.08, 2.68742],
             [23, 19.1652, 38.872, 45.08, 2.68742],
             [24, 19.1652, 38.8039, 45.08, 2.68742]
             ]
        )

        assert torch.all(train_feature.eq(true_values))
