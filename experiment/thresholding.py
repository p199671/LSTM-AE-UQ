import torch

from utils.utils import mean_decenter, mean_center, reconstruct_temp


class Thresholding:
    def __init__(self, config, operator, threshold=None):
        self.threshold = threshold
        self.operator = operator
        self.config = config

    def set_threshold_by_training_set(self, data):
        print('Calculating threshold on training set...')
        losses = self.get_losses(data)
        self.threshold = torch.quantile(losses, 0.999)  # losses.max()
        print(f'Threshold: {self.threshold}')

    def apply_threshold(self, data):
        print('Applying threshold...')
        losses = self.get_losses(data)

        indices = losses > self.threshold
        print(f'Number of inputs above threshold: {indices.sum()}')
        return indices

    def get_losses(self, data):
        inputs_collection = []
        reconstruction_collection = []
        loss_per_input_collection = []

        with torch.no_grad():
            self.operator.model.eval()

            for i, inputs in enumerate(data):
                inputs_collection.append(inputs)
                inputs_mean_centered, mean = mean_center(inputs)
                inputs_mean_centered = inputs_mean_centered.to(self.operator.device)

                # forward pass
                results = self.operator.model(inputs_mean_centered)
                reconstruction = mean_decenter(results['mean'], mean)
                reconstruction_collection.append(reconstruction)

                loss_per_input = self.operator.criterion(results, inputs_mean_centered)
                loss_per_input_collection.append(loss_per_input)

            loss_per_input_collection = reconstruct_temp(loss_per_input_collection, self.config['data']['seq_len'],
                                                         self.config['test']['step_size'])

        return loss_per_input_collection
