from unittest import TestCase

import torch

import data
from configs.config import load_configs

CONFIG_FILE = '../configs/config.yml'
configs = load_configs(CONFIG_FILE)
configs['data']['dir'] = "../data/"
configs['data']['name'] = "META"
configs['data']['batch_size'] = 8


class TestMyDataset(TestCase):
    data_loader = data.get_dataloader(configs)
    dataset = data.MyDataset(configs)

    def test_loaded_meta_dataset_is_meta(self, dataset=dataset):
        type_dataset = type(dataset.dataset)
        assert dataset.dataset_name == 'META'
        assert type_dataset == data.META

    def test_first_item_of_dataset(self, dataset=dataset):
        first_item = dataset[0]
        sample = torch.Tensor([38.590000, 38.740002, 38.009998, 38.500000, 38.500000, 43532300])
        assert torch.all(sample.eq(first_item))

    def test_train_loader_not_none(self, data_loader=data_loader):
        train_loader = data_loader['train']
        assert train_loader is not None

    def test_val_loader_not_none(self, data_loader=data_loader):
        val_loader = data_loader['val']
        assert val_loader is not None

    def test_first_batch_of_train_loader(self, train_loader=data_loader['train']):
        train_feature = next(iter(train_loader))
        true_values = torch.Tensor(
            [[38.590000, 38.740002, 38.009998, 38.500000, 38.500000, 43532300],
             [38.200001, 38.500000, 38.099998, 38.220001, 38.220001, 31161000],
             [38.240002, 38.320000, 36.770000, 37.020000, 37.020000, 65379200],
             [36.830002, 37.549999, 36.619999, 36.650002, 36.650002, 48423900],
             [36.360001, 37.070000, 36.020000, 36.560001, 36.560001, 56521100],
             [36.970001, 37.490002, 36.900002, 37.080002, 37.080002, 45840800],
             [37.430000, 38.279999, 37.139999, 37.810001, 37.810001, 57609600],
             [38.349998, 38.580002, 37.689999, 38.410000, 38.410000, 57995200]])

        assert torch.all(train_feature.eq(true_values))
