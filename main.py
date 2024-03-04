from datetime import datetime

import torch

import data
from configs.config import load_configs
from experiment.operator import Operator
from experiment.plotter import Plotter
from experiment.thresholding import Thresholding

CONFIG_FILE = 'configs/config.yml'


def main(config):
    seed = config['environment']['seed']
    torch.manual_seed(seed)

    dataset = data.MyDataset(config)

    operator = Operator(config)
    data_loader = data.get_dataloader(dataset, config)

    if config['environment']['is_train']:
        print(f'Training started with model {operator.model.model_name} on dataset {dataset.dataset_name}.')
        start_time = datetime.now()
        operator.train(data_loader['train'], data_loader['val'])
        end_time = datetime.now()
        print(f'Training ended after {end_time - start_time} h.')
    else:
        print(f'Testing started with model {operator.model.model_name} on dataset {dataset.dataset_name}.')
        thresholding = Thresholding(config, operator)
        thresholding.set_threshold_by_training_set(data_loader['train'])
        results = operator.test(data_loader['test'])
        operator.save(results)
        print('Testing ended.')
        print(f'Results saved in {operator.result_path}.')
        print('Making plots.')
        plotter = Plotter(config, operator.result_path)
        plotter.make_plots()


if __name__ == '__main__':
    configs = load_configs(CONFIG_FILE)
    main(configs)
