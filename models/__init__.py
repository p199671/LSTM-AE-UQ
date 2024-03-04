import os
from importlib import import_module

import torch
from torch import nn

from configs.config import save_configs


class Model(nn.Module):
    def __init__(self, config):
        super(Model, self).__init__()

        self.config = config
        self.model_name = config['model']['name']
        self.device = config['environment']['device']
        self.mode = config['model']['mode']
        self.mc_samples = config['test']['mc_samples']

        module = import_module(f'models.{self.model_name}')
        self.model = module.make_model(config).to(self.device)

    def forward(self, x):
        if self.model.training:
            return self.model.forward(x)
        else:
            forward_function = self.model.forward
            if self.mode == 'normal':
                return forward_function(x)
            elif self.mode == 'aleatoric':
                return self.test_aleatoric(x, forward_function)
            elif self.mode == 'epistemic':
                return self.test_epistemic(x, forward_function)
            elif self.mode == 'combined':
                return self.test_combined(x, forward_function)

    def test_aleatoric(self, x, forward_function):
        results = forward_function(x)
        mean = results['mean']
        aleatoric_variance = torch.exp(results['aleatoric_var'])
        results = {'mean': mean, 'aleatoric_var': aleatoric_variance}
        return results

    def test_epistemic(self, x, forward_function):
        means = []

        for i in range(self.mc_samples):
            results = forward_function(x)
            mean = results['mean']
            means.append(mean)

        mean_values = torch.stack(means, dim=0)
        variance = mean_values.var(dim=0)

        results = {'mean': mean_values.mean(dim=0), 'epistemic_var': variance}
        return results

    def test_combined(self, x, forward_function):
        means = []
        aleatoric_variances = []

        for i in range(self.mc_samples):
            results = forward_function(x)
            mean = results['mean']
            aleatoric_variance = torch.exp(results['aleatoric_var'])
            means.append(mean)
            aleatoric_variances.append(aleatoric_variance)

        mean_values = torch.stack(means, dim=0)
        mean_variance = mean_values.var(dim=0)
        aleatoric_variance = torch.stack(aleatoric_variances, dim=0).mean(dim=0)
        predictive_variance = mean_variance + aleatoric_variance

        results = {'mean': mean_values.mean(dim=0),
                   'aleatoric_var': aleatoric_variance,
                   'epistemic_var': mean_variance,
                   'predictive_var': predictive_variance}
        return results

    def save(self, path=None):
        model_parameters = self.model.state_dict()
        if path is None:
            path = self.config['environment']['model_path']
            # timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
            # data_name = self.config['data']['name']
            # model_name = self.model_name
            # filename = f'{timestamp}_{model_name}_{data_name}'
            # path = f'{path}/{filename}'

        if not os.path.exists(path):
            os.makedirs(path)
        torch.save(model_parameters, f'{path}/model.pt')
        save_configs(self.config, f'{path}/config.yaml')
        print(f'Model and configs saved at {path}.')

    def load_trained_model(self, config):
        model_path = config['environment']['model_path']
        model_name = config['test']['model_name']

        # if model_name == '':
        #     model_name = get_most_recent_model_directory(model_path)
        #     config['test']['model_name'] = model_name

        model_path = f'{model_path}/model.pt'

        self.model.load_state_dict(torch.load(model_path))
