import os

DATA_BASE_DIR = 'data/anomalies'

main = 'main.py'

data_sets = ['amplitude-larger', 'amplitude-smaller', 'global-extremum', 'local-extremum', 'pattern-anomaly']
hidden_dims = [50, 100]
encoding_dims = [40, 80]
window_sizes = [10, 25, 50, 100]

for data_set in data_sets:
    for hidden_dim, encoding_dim in zip(hidden_dims, encoding_dims):
        for window_size in window_sizes:
            print(
                f'Running experiment for {data_set} with hidden dim {hidden_dim}, ecoding dim {encoding_dim} and window size {window_size}')

            result_path = f'experiments/uncertainty/anomalies_test/{data_set}/{hidden_dim}-{encoding_dim}/window_size_{window_size}'
            os.system(
                f'python3 {main} --mode train --results-path {result_path} --results-model-path {result_path} --tensorboard-path {result_path} --seq-len {window_size} --data-dir {DATA_BASE_DIR}/{data_set} --hidden-dim {hidden_dim} --embedding-dim {encoding_dim}')

            model_path = f'experiments/uncertainty/anomalies_test/{data_set}/{hidden_dim}-{encoding_dim}/window_size_{window_size}'
            result_path = f'experiments/uncertainty/anomalies_test/{data_set}/{hidden_dim}-{encoding_dim}/window_size_{window_size}/test'
            os.system(
                f'python3 {main} --mode test --results-path {result_path} --test-model-path {model_path} --results-model-path {model_path} --tensorboard-path {result_path} --seq-len {window_size} --data-dir {DATA_BASE_DIR}/{data_set} --hidden-dim {hidden_dim} --embedding-dim {encoding_dim}')

print('Finished experiments.')
