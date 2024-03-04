# Sine Dataset Creation

This dataset was created using the synthetic data generation
tool [gutenTAG](https://github.com/HPI-Information-Systems/gutentag/tree/main).

## Dataset Creation

We created 5 datasets to analyse models uncertainty detection capabilities.
Since gutenTAG only generates test datasets with anomaly, the existing datasets were created in two rounds.
The first dataset is created using the specifications in overview_1.yaml with seed 11.
The second is created with the specifications in overview_2.yaml with seed 12. From there, the train_no_anomaly.csv was
taken, renamed to test_no_anomaly.csv and put into the corresponding directory of the first dataset.

Matching the directory names, there are 5 different levels of noise: 0%, 1%, 10%, 30% and 50%.

The commands used for dataset creation are:

```bash
gutenTAG --config-yaml config.yaml --seed 11 --output-dir sine
gutenTAG --config-yaml config.yaml --seed 12 --output-dir sine
```
