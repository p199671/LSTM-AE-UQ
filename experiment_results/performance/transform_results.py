import os

import pandas as pd


def merge_metrics(df, alg):
    metr = ['AUC_ROC', 'Precision', 'Recall', 'F']

    for metric in metr:
        mergedTable = pd.read_csv(f'comparison/mergedTable_{metric}.csv')

        if alg not in mergedTable.columns:
            metric_algorithm = df[['filename', metric]].rename(columns={metric: alg})
            mergedTable = pd.merge(mergedTable, metric_algorithm, on='filename', how='left')
            mergedTable.to_csv(f'comparison/mergedTable_{metric}.csv', index=False)


def get_metrics(path):
    # Initialize empty lists to store values for each metric
    filenames = []
    auc_roc_values = []
    precision_values = []
    recall_values = []
    f_values = []

    # Iterate over each folder in the directory
    for dataset_folder in os.listdir(path):
        dataset_path = os.path.join(path, dataset_folder)

        # Check if the item in the directory is a folder
        if os.path.isdir(dataset_path):

            # Iterate over each CSV file in the dataset folder
            for csv_file in os.listdir(dataset_path):
                if csv_file.endswith('.csv'):
                    csv_path = os.path.join(dataset_path, csv_file)
                    filename = csv_file.removesuffix('_metrics.csv')
                    if dataset_folder == 'KDD21':
                        filename = filename.replace('out', 'txt')
                    elif dataset_folder == 'NASA-MSL' or dataset_folder == 'NASA-SMAP':
                        filename = filename.replace('.test', '_data')
                        filename = "SMAP" + filename

                    df = pd.read_csv(csv_path)

                    # Extract values from the single line in the CSV
                    filenames.append(filename)
                    auc_roc_values.append(df['AUC_ROC'].iloc[0])
                    precision_values.append(df['Precision'].iloc[0])
                    recall_values.append(df['Recall'].iloc[0])
                    f_values.append(df['F'].iloc[0])

    df = pd.DataFrame(
        {'filename': filenames, 'AUC_ROC': auc_roc_values, 'Precision': precision_values, 'Recall': recall_values,
         'F': f_values})

    return df


data_dir = 'tests'
algorithms = ['AE_combined', 'LSTMAE_normal', 'LSTMAE_combined', 'AE_normal']

for algorithm in algorithms:
    metrics = get_metrics(os.path.join(data_dir, algorithm))
    merge_metrics(metrics, algorithm)
