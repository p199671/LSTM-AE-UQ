import os

import torch
from matplotlib import pyplot as plt


class Plotter:
    def __init__(self, config, result_path):
        self.config = config
        self.result_path = result_path
        self.is_epistemic = config['model']['mode'] in ['epistemic', 'combined']
        self.is_aleatoric = config['model']['mode'] in ['aleatoric', 'combined']
        self.is_combined = config['model']['mode'] == 'combined'

    def make_plots(self):
        result_path = self.result_path
        plots_path = f'{result_path}/plots/'
        data_path = f'{result_path}/data/'

        if not os.path.exists(plots_path):
            os.makedirs(plots_path)

        inputs = torch.load(f'{data_path}/inputs.pt')
        reconstructions = torch.load(f'{data_path}/reconstructions.pt')

        self.plot_input_and_output(plots_path, inputs, reconstructions)

        if self.is_epistemic:
            outputs_var = torch.load(f'{data_path}/epistemic_uncertainty.pt')
            self.plot_epistemic_uncertainty(plots_path, outputs_var)

        if self.is_aleatoric:
            outputs_var = torch.load(f'{data_path}/aleatoric_uncertainty.pt')
            self.plot_aleatoric_uncertainty(plots_path, outputs_var)

        if self.is_combined:
            outputs_var = torch.load(f'{data_path}/predictive_uncertainty.pt')
            self.plot_predictive_uncertainty(plots_path, outputs_var)

        losses = torch.load(f'{data_path}/losses.pt')
        self.plot_loss(plots_path, losses)

    def plot_input_and_output(self, plots_path, inputs, outputs_mean):
        for i in range(len(self.config['data']['features'])):
            plt.figure()
            plt.plot(inputs[:, i].cpu().numpy(), label='inputs')
            plt.plot(outputs_mean[:, i].cpu().numpy(), label='reconstruction')
            plt.title("Input and Reconstruction")
            plt.xlabel('time')
            plt.ylabel(self.config['data']['features'][i])
            plt.legend()
            plt.savefig(f'{plots_path}/{self.config["data"]["features"][i]}.png')
            plt.close()

    def plot_loss(self, plots_path, losses):
        plt.figure()
        plt.plot(losses.cpu().numpy(), label='reconstruction loss')
        plt.title('Reconstruction Error')
        plt.xlabel('time')
        plt.ylabel('loss')
        plt.legend()
        plt.savefig(f'{plots_path}/loss.png')
        plt.close()

    def plot_epistemic_uncertainty(self, plots_path, outputs_var):
        for i in range(len(self.config['data']['features'])):
            plt.figure()
            plt.plot(outputs_var[:, i].cpu().numpy(), label='epistemic uncertainty')
            plt.xlabel('time')
            plt.ylabel(r'$\mu^2$')
            plt.title("Model Uncertainty")
            plt.legend()
            plt.savefig(f'{plots_path}/{self.config["data"]["features"][i]}_epistemic_uncertainty.png')
            plt.close()

    def plot_aleatoric_uncertainty(self, plots_path, outputs_var):
        for i in range(len(self.config['data']['features'])):
            plt.figure()
            plt.plot(outputs_var[:, i].cpu().numpy(), label='aleatoric uncertainty')
            plt.xlabel('time')
            plt.ylabel(r'$\sigma^2$')
            plt.title("Data Uncertainty")
            plt.legend()
            plt.savefig(f'{plots_path}/{self.config["data"]["features"][i]}_aleatoric_uncertainty.png')
            plt.close()

    def plot_predictive_uncertainty(self, plots_path, outputs_var):
        for i in range(len(self.config['data']['features'])):
            plt.figure()
            plt.plot(outputs_var[:, i].cpu().numpy(), label='predictive uncertainty')
            plt.xlabel('time')
            plt.ylabel(r'$\sigma^2$')
            plt.title("Predictive Uncertainty")
            plt.legend()
            plt.savefig(f'{plots_path}/{self.config["data"]["features"][i]}_predictive_uncertainty.png')
            plt.close()
