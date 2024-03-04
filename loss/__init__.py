from importlib import import_module

from torch import nn


class Loss(nn.Module):
    def __init__(self, config):
        super(Loss, self).__init__()

        if config['model']['mode'] in ['epistemic', 'normal']:
            module = import_module('loss.mse')
            self.loss_function = getattr(module, 'MSE')
        else:
            module = import_module('loss.mse_var')
            self.loss_function = getattr(module, 'MSE_VAR')

    def forward(self, results, inputs):
        if self.loss_function.__name__ == 'MSE_VAR':
            mean, var = results['mean'], results['aleatoric_var']
            loss = self.loss_function()(mean, var, inputs)
        elif self.loss_function.__name__ == 'MSE':
            reconstructions = results['mean']
            loss = self.loss_function()(reconstructions, inputs)

        return loss
