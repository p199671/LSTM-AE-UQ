# MA Project

This is a repository for the MA project of the MSc in Computer Science.
In this project, we implement an anomaly detection algorithm for time series data considering uncertainty.
Specifically, we implement a Bayesian LSTM Autoencoder with a split head to account for aleatoric and epistemic
uncertainty.
The model is build to work with univariate time series data, but can be easily extended to multivariate time series
data.


## Installation

The following tools are required to run the code:

- git
- conda (anaconda or miniconda)

To install the code, run the following commands in the terminal:

```bash
git clone https://gitlab.lrz.de/paul_wiessner/ma.git
# alternatively use
# git clone https://gitlab.lis-lab.fr/wiessner/ma.git
cd ma
conda env create -f environment.yml
pip3 install -r requirements.txt
conda activate ma
```

## Usage

Before running the code, have a look on the configuration file `configs/config.yml` and adjust the parameters to your
needs.
The configuration file is structured as follows:

| Parameter                        | Description                                                                      |
|----------------------------------|----------------------------------------------------------------------------------|
| `environment`                    |                                                                                  |
| `environment.model_path`         | Path to store and load the trained models.                                       |
| `environment.results_path`       | Path to where the resuls are stored.                                             |
| `environment.is_train`           | Boolean, whether train or test.                                                  |
| `environment.tensorboard`        | Boolean, whether to use tensorboard.                                             |
| `environment.tensorboard_path`   | Path to tensorboard files.                                                       |
| `environment.seed`               | Seed for the random number generator.                                            |
| `data`                           |                                                                                  |
| `data.dir`                       | Path to the data.                                                                |
| `data.name`                      | Name of the python file for the dataset.                                         |
| `data.batch_size`                | Batch size for training.                                                         |
| `data.seq_len`                   | Sequence length for training.                                                    |
| `data.step_size`                 | Step size for training.                                                          |
| `data.features`                  | Features of data to be used for training.                                        |
| `model`                          |                                                                                  |
| `model.name`                     | Name of the model.                                                               |
| `model.in_channels`              | Number of input channels.                                                        |
| `model.hidden_dim`               | List with number of hidden units.                                                |
| `model.embedding_dim`            | Dimension of the embedding.                                                      |
| `model.mode`                     | Mode of the model (one of `normal`, `aleatoric`, `epistemic`, `combined`).       |
| `model.dropout`                  | Dropout rate.                                                                    |
| `train`                          |                                                                                  |
| `train.epochs`                   | Number of epochs.                                                                |
| `train.learning_rate`            | Learning rate.                                                                   |
| `train.early_stopping_threshold` | Threshold for early stopping.                                                    |
| `test`                           |                                                                                  |
| `test.model_name`                | Name of the trained model to be tested. If empty, the latest model will be used. |
| `test.mc_samples`                | Number of Monte Carlo samples.                                                   |
| `test.step_size`                 | Step size for testing.                                                           |

You can run the code with the following command:

```bash
python3 main.py
```

Pay attention to the `environment.is_train` parameter in the configuration file, as it determines whether the model is
trained or tested.




