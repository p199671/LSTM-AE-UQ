import os.path
import re
from datetime import datetime
from unittest import TestCase

import torch

from configs.config import load_configs
from experiment.operator import Operator

CONFIG_FILE = '../configs/config.yml'
config = load_configs(CONFIG_FILE)
config['environment']['device'] = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
config['environment']['tensorboard'] = False
config['environment']['model_path'] = '../models_trained/'


class TestOp(TestCase):
    def test_save_model_creates_file(self):
        path = config['environment']['model_path']

        operator = Operator(config)
        operator.save_model()

        files = os.listdir(path)
        assert len(files) == 1

        for file in files:
            os.remove(f'{path}/{file}')
        os.rmdir(path)

    def test_save_model_creates_file_with_current_timestamp(self):
        path = config['environment']['model_path']

        operator = Operator(config)
        operator.save_model()

        files = os.listdir(path)
        assert len(files) == 1

        filename = files[0]
        timestamp_match = re.search(r'(\d{4}-\d{2}-\d{2}_\d{2}-\d{2}-\d{2})', filename)
        assert timestamp_match is not None

        timestamp_str = timestamp_match.group(1)
        timestamp = datetime.strptime(timestamp_str, "%Y-%m-%d_%H-%M-%S")
        current_time = datetime.now()
        time_difference = current_time - timestamp

        assert time_difference.seconds < 10

        for file in files:
            os.remove(f'{path}/{file}')
        os.rmdir(path)

    def test_save_model_creates_file_with_model_name(self):
        path = config['environment']['model_path']
        config['model']['name'] = 'LSTMAE'

        operator = Operator(config)
        operator.save_model()

        files = os.listdir(path)
        filename = files[0]
        name_found = filename.find('LSTMAE')

        assert name_found != -1

        for file in files:
            os.remove(f'{path}/{file}')
        os.rmdir(path)

    def test_average_sliding_window_with_one_feature_and_stride_one(self):
        input = [torch.Tensor([[1],[2],[3]]), torch.Tensor([[2],[3],[4]]), torch.Tensor([[3],[4],[5]]), torch.Tensor([[4],[5],[6]])]

        operator = Operator(config)
        config['data']['seq_len'] = 3
        output = operator.average_sliding_window(input, 3, 1)

        assert output.shape == (6, 1)
        assert (output == torch.Tensor([[1], [2], [3], [4], [5], [6]])).all()

    def test_average_sliding_window_with_one_feature_and_stride_two(self):
        input = [torch.Tensor([[1], [2], [3]]), torch.Tensor([[3], [4], [5]]),
                 torch.Tensor([[5], [6], [7]]), torch.Tensor([[7], [8], [9]])]

        operator = Operator(config)
        config['data']['seq_len'] = 3
        output = operator.average_sliding_window(input, 3, 2)

        assert output.shape == (9, 1)
        assert (output == torch.Tensor([[1], [2], [3], [4], [5], [6], [7], [8], [9]])).all()

    def test_average_sliding_window_with_two_features_and_stride_one(self):
        input = [torch.Tensor([[1, 10], [2, 20], [3, 30]]), torch.Tensor([[2, 20], [3, 30], [4, 40]]),
                 torch.Tensor([[3, 30], [4, 40], [5, 50]]), torch.Tensor([[4, 40], [5, 50], [6, 60]])]

        operator = Operator(config)
        config['data']['seq_len'] = 3
        output = operator.average_sliding_window(input, 3, 1)

        assert output.shape == (6, 2)
        assert (output == torch.Tensor([[1, 10], [2, 20], [3, 30], [4, 40], [5, 50], [6, 60]])).all()

    def test_average_sliding_window_with_two_features_and_stride_two(self):
        input = [torch.Tensor([[1, 10], [2, 20], [3, 30]]), torch.Tensor([[3, 30], [4, 40], [5, 50]]),
                 torch.Tensor([[5, 50], [6, 60], [7, 70]]), torch.Tensor([[7, 70], [8, 80], [9, 90]])]

        operator = Operator(config)
        config['data']['seq_len'] = 3
        output = operator.average_sliding_window(input, 3, 2)

        assert output.shape == (9, 2)
        assert (output == torch.Tensor([[1, 10], [2, 20], [3, 30], [4, 40], [5, 50], [6, 60], [7, 70], [8, 80], [9, 90]])).all()
