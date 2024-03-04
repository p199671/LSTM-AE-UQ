import os

DATA_BASE_DIR = 'data/sine'

data_sets = ['sine-noise-00%', 'sine-noise-01%', 'sine-noise-10%', 'sine-noise-30%', 'sine-noise-50%']
hidden_dims = [50, 100]
encoding_dims = [40, 80]
window_sizes = [10, 25]

for data_set_model in data_sets:
    for hidden_dim, encoding_dim in zip(hidden_dims, encoding_dims):
        for window_size in window_sizes:
            for data_set in data_sets:
                print(
                    f'Running experiment for {data_set} with hidden dim {hidden_dim}, encoding dim {encoding_dim} and window size {window_size}')

                model_path = f'experiments/uncertainty/sine/{data_set_model}/{hidden_dim}-{encoding_dim}/window_size_{window_size}'
                result_path = f'experiments/uncertainty/sine/{data_set_model}/{hidden_dim}-{encoding_dim}/window_size_{window_size}/test/{data_set}'
                os.system(
                    f'python3 main.py --mode test --results-path {result_path} --test-model-path {model_path} --results-model-path ./ --tensorboard-path {result_path} --seq-len {window_size} --data-dir {DATA_BASE_DIR}/{data_set} --hidden-dim {hidden_dim} --embedding-dim {encoding_dim}')
