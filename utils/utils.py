import os

import pandas as pd
import torch


def date_to_timestamp(date):
    return pd.Timestamp(date).timestamp()


def get_most_recent_model_directory(models_path):
    model_names = [name for name in os.listdir(models_path)]
    model_names = [name for name in model_names if 'tfevents' not in name]
    model_names = sorted(model_names)
    return model_names[-1]


def mean_center(sequence):
    seq_mean = sequence.mean(dim=1)
    seq_mean_expanded = seq_mean.unsqueeze(1).repeat(1, sequence.shape[1], 1)
    sequence_centered = sequence - seq_mean_expanded
    return sequence_centered, seq_mean_expanded.to(sequence.device)


def mean_decenter(reconstruction, mean):
    reconstruction = reconstruction + mean
    return reconstruction


def reconstruct_temp(collection, seq_len, step_size=1):
    collection = torch.cat(collection, 0)
    shape = (seq_len + (collection.shape[0] - 1) * step_size, collection.shape[2])
    collection_sum = torch.zeros(shape).to(collection.device)
    divisor = torch.zeros(shape).to(collection.device)

    for i in range(len(collection)):
        collection_sum[i * step_size:i * step_size + seq_len, :] += collection[i]
        divisor[i * step_size:i * step_size + seq_len, :] += 1

    collection_avg = collection_sum / divisor

    return collection_avg


def checkpoint(model):
    torch.save(model.state_dict(), 'checkpoint.pth')


def load_checkpoint(model):
    model.load_state_dict(torch.load('checkpoint.pth'))
    os.remove('checkpoint.pth')
    return model
