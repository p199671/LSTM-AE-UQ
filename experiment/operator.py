import os

import torch
from torch import optim
from torch.utils.tensorboard import SummaryWriter

from loss import Loss
from models import Model
from utils.utils import mean_center, mean_decenter, reconstruct_temp, checkpoint, load_checkpoint


class Operator:
    def __init__(self, config):
        self.config = config
        self.epochs = config['train']['epochs']
        self.tensorboard = config['environment']['tensorboard']
        if self.tensorboard and config['environment']['is_train']:
            self.writer = SummaryWriter()
        self.device = config['environment']['device']

        self.model = Model(config)
        self.is_epistemic = config['model']['mode'] in ['epistemic', 'combined']
        self.is_aleatoric = config['model']['mode'] in ['aleatoric', 'combined']
        self.is_combined = config['model']['mode'] == 'combined'
        self.mc_samples = config['test']['mc_samples']

        self.criterion = Loss(config)

        if config['environment']['is_train']:
            self.optimizer = optim.Adam(self.model.parameters(), lr=config['train']['learning_rate'])
        else:
            self.model.load_trained_model(config)
            self.result_path = f'{self.config["environment"]["results_path"]}/{self.config["test"]["model_name"].split(".")[0]}/testrun_{datetime.now().strftime("%Y-%m-%d_%H-%M-%S")}/'

    def train(self, train_loader, valid_loader):
        early_stopping_threshold = self.config['train']['early_stopping_threshold']
        best_valid_loss = float('inf')
        best_epoch = 0

        for epoch in range(self.epochs):
            self.model.train()
            train_loss = 0.0

            for i, inputs in enumerate(train_loader):
                inputs_mean_centered, _ = mean_center(inputs)
                inputs_mean_centered = inputs_mean_centered.to(self.device)

                self.optimizer.zero_grad()

                results = self.model(inputs_mean_centered)

                loss_per_input = self.criterion(results, inputs_mean_centered)
                loss = loss_per_input.mean()

                loss.backward()
                self.optimizer.step()

                train_loss += loss.item()

            train_loss = train_loss / len(train_loader)

            if (epoch + 1) % 10 == 0:
                self.model.eval()
                valid_loss = 0.0

                with torch.no_grad():
                    for i, inputs in enumerate(valid_loader):
                        inputs_mean_centered, _ = mean_center(inputs)
                        inputs_mean_centered = inputs_mean_centered.to(self.device)

                        results = self.model(inputs_mean_centered)

                        loss_per_input = self.criterion(results, inputs_mean_centered)
                        loss = loss_per_input.mean()

                        valid_loss += loss.item()

                    valid_loss = valid_loss / len(valid_loader)

                    if valid_loss < best_valid_loss:
                        best_valid_loss = valid_loss
                        best_epoch = epoch
                        checkpoint(self.model)
                    elif (epoch - best_epoch) / 10 >= early_stopping_threshold:
                        print(f'Early stopping at epoch {epoch + 1}.')
                        self.model = load_checkpoint(self.model)
                        break

                if self.tensorboard:
                    self.writer.add_scalar('loss/valid', valid_loss, epoch)

            if self.tensorboard:
                self.writer.add_scalar('loss/train', train_loss, epoch)

            if (epoch + 1) % 10 == 0:
                print(f'Epoch: {epoch + 1}/{self.epochs} | Training loss: {train_loss} | Validation loss: {valid_loss}')
            else:
                print(f'Epoch: {epoch + 1}/{self.epochs} | Training loss: {train_loss}')

        self.model.save()

    def test(self, test_loader):
        inputs_collection = []
        reconstruction_collection = []
        if self.is_epistemic:
            outputs_epistemic_collection = []
        if self.is_aleatoric:
            outputs_aleatoric_collection = []
        if self.is_combined:
            outputs_predictive_collection = []
        loss_per_input_collection = []

        with torch.no_grad():
            self.model.eval()

            test_loss = 0.0

            for i, inputs in enumerate(test_loader):
                inputs_collection.append(inputs)
                inputs_mean_centered, mean = mean_center(inputs)
                inputs_mean_centered = inputs_mean_centered.to(self.device)
                mean = mean.to(self.device)

                # forward pass
                results = self.model(inputs_mean_centered)
                reconstruction = mean_decenter(results['mean'], mean)
                reconstruction_collection.append(reconstruction)
                if self.is_epistemic:
                    outputs_epistemic_collection.append(
                        results['epistemic_var'].reshape(-1, self.config['data']['seq_len'],
                                                         len(self.config['data']['features'])))
                if self.is_aleatoric:
                    outputs_aleatoric_collection.append(
                        results['aleatoric_var'].reshape(-1, self.config['data']['seq_len'],
                                                         len(self.config['data']['features'])))
                if self.is_combined:
                    outputs_predictive_collection.append(
                        results['predictive_var'].reshape(-1, self.config['data']['seq_len'],
                                                          len(self.config['data']['features'])))

                loss_per_input = self.criterion(results, inputs_mean_centered)
                loss_per_input_collection.append(loss_per_input)
                loss = loss_per_input.mean()

                test_loss += loss.item()

            test_loss = test_loss / len(test_loader)
            print(f'Test loss: {test_loss}')

            results = {'inputs': reconstruct_temp(inputs_collection, self.config['data']['seq_len'],
                                                  self.config['test']['step_size']),
                       'reconstructions': reconstruct_temp(reconstruction_collection, self.config['data']['seq_len'],
                                                           self.config['test']['step_size']),
                       'losses': reconstruct_temp(loss_per_input_collection, self.config['data']['seq_len'],
                                                  self.config['test']['step_size']),
                       }

            if self.is_epistemic:
                results['epistemic_uncertainty'] = reconstruct_temp(outputs_epistemic_collection,
                                                                    self.config['data']['seq_len'],
                                                                    self.config['test']['step_size'])
            if self.is_aleatoric:
                results['aleatoric_uncertainty'] = reconstruct_temp(outputs_aleatoric_collection,
                                                                    self.config['data']['seq_len'],
                                                                    self.config['test']['step_size'])
            if self.is_combined:
                results['predictive_uncertainty'] = reconstruct_temp(outputs_predictive_collection,
                                                                     self.config['data']['seq_len'],
                                                                     self.config['test']['step_size'])

            return results

    def save(self, results):
        if not os.path.exists(f'{self.result_path}/data/'):
            os.makedirs(f'{self.result_path}/data/')

        for key, collection in results.items():
            filename = key
            torch.save(collection, f'{self.result_path}/data/{filename}.pt')
