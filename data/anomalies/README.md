# Anomalies Data Set Creation

This dataset was created using the synthetic data generation
tool [gutenTAG](https://github.com/HPI-Information-Systems/gutentag/tree/main).

We created 5 datasets with different anomaly types to analyse the models uncertainty behavior.
In totel, 5 different anomaly types were used

- **Anomaly 1**: A smaller amplitude anomaly
- **Anomaly 2**: A larger amplitude anomaly
- **Anomaly 3**: A single global point anomaly
- **Anomaly 4**: A single local point anomaly
- **Anomaly 5**: A pattern anomaly

The commands used for dataset creation are:

``` bash
gutenTAG --config-yaml config.yaml --seed 17 --output-dir anomalies
```
