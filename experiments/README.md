# Experiments

## Running an Experiment with Training

To run an experiment and train a model, execute the following command from the repository's root directory:

```bash
poetry run python experiments/wind_power_Xd_test_turbine_Y/main.py
```

Here, `X` denotes the input dimensionality (typically `1` or `5`), and `Y` denotes the test turbine ID.

## Running an Experiment with a Pretrained Model

To run an experiment using a previously trained model, execute:

```bash
poetry run python experiments/wind_power_Xd_test_turbine_Y/main.py --config.restore=logs/wind-energy-regression-one-turbine-20250616-eval/TRAINED_MODEL_FOLDER
```

Here, `X` denotes the input dimensionality (typically `1` or `5`), `Y` denotes the test turbine ID, and `TRAINED_MODEL_FOLDER` specifies the directory containing the pretrained model.

For example, for the `wind_power_1d_test_turbine_1` experiment, the pretrained model directory is `Jul27_213009_vakr`. The corresponding command is:

```bash
poetry run python experiments/wind_power_1d_test_turbine_1/main.py --config.restore=logs/wind-energy-regression-one-turbine-20250616-eval/Jul27_213009_vakr
```
