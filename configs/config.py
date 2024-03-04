import argparse

import torch
import yaml

parser = argparse.ArgumentParser(description='PyTorch ImageNet Training')
parser.add_argument('--mode', choices=['train', 'test'],
                    help='train or test')
parser.add_argument('--results-path', type=str,
                    help='path to save results')
parser.add_argument('--results-model-path', type=str,
                    help='path to load model')
parser.add_argument('--tensorboard-path', type=str,
                    help='path to save tensorboard logs')
parser.add_argument('--seq-len', type=int,
                    help='sequence length')
parser.add_argument('--data-dir', type=str,
                    help='path to data directory')
parser.add_argument('--hidden-dim', nargs='+', type=int,
                    help='hidden dimensions of the autoencoder')
parser.add_argument('--embedding-dim', type=int,
                    help='embedding dimension of the autoencoder')
parser.add_argument('--test-model-path', type=str,
                    help='path to load model for testing')


def overwrite_configs_with_command_line_args(config, args):
    if args.seq_len is not None:
        config['data']['seq_len'] = args.seq_len
    if args.data_dir is not None:
        config['data']['dir'] = args.data_dir
    if args.mode is not None:
        config['environment']['is_train'] = True if args.mode == 'train' else False
    if args.results_path is not None:
        config['environment']['results_path'] = args.results_path
    if args.results_model_path is not None:
        config['environment']['model_path'] = args.results_model_path
    if args.tensorboard_path is not None:
        config['environment']['tensorboard_path'] = args.tensorboard_path
    if args.hidden_dim is not None:
        config['model']['hidden_dim'] = args.hidden_dim
    if args.embedding_dim is not None:
        config['model']['embedding_dim'] = args.embedding_dim
    if args.test_model_path is not None:
        config['test']['model_name'] = args.test_model_path

    return config


def add_additional_configs(config):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    config['environment']['device'] = device
    return config


def load_configs(config_file):
    with open(config_file, 'r') as f:
        config = yaml.safe_load(f)

    args = parser.parse_args()
    config = overwrite_configs_with_command_line_args(config, args)
    config = add_additional_configs(config)

    config = check_configs(config)

    return config


def save_configs(config, path):
    with open(path, 'w') as f:
        yaml.dump(config, f)


def check_configs(config):
    validate_model_mode(config['model']['mode'])
    validate_step_size_smaller_than_seq_len(config['test']['step_size'], config['data']['seq_len'])
    return config


def validate_model_mode(mode):
    if mode not in ["normal", "aleatoric", "epistemic", "combined"]:
        raise ValueError(
            f"Unknown mode: {mode}\nMode must be one of 'normal', 'aleatoric', 'epistemic', 'combined'")


def validate_step_size_smaller_than_seq_len(step_size, seq_len):
    if step_size >= seq_len:
        raise ValueError(
            f"Step size ({step_size}) must be smaller than sequence length ({seq_len})")
