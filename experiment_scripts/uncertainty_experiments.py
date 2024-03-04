import os

DATA_BASE_DIR = 'data/sine'

data_sets = ['sine-noise-00%', 'sine-noise-01%', 'sine-noise-10%', 'sine-noise-30%', 'sine-noise-50%']
hidden_dims = [50, 100]
encoding_dims = [40, 80]
window_sizes = [10, 25, 50, 100]

for data_set in data_sets:
    for hidden_dim, encoding_dim in zip(hidden_dims, encoding_dims):
        for window_size in window_sizes:
            print(
                f'Running experiment for {data_set} with hidden dim {hidden_dim}, encoding dim {encoding_dim} and window size {window_size}')

            result_path = f'experiments/uncertainty/sine/{data_set}/{hidden_dim}-{encoding_dim}/window_size_{window_size}'
            os.system(
                f'python3 main.py --mode train --results-path {result_path} --results-model-path {result_path} --tensorboard-path {result_path} --seq-len {window_size} --data-dir {DATA_BASE_DIR}/{data_set} --hidden-dim {hidden_dim} --embedding-dim {encoding_dim}')

print('Finished experiments.')
