# Experiments

## Running an Experiment - with Training

```
poetry run python experiments/wind_power_Xd_test_turbine_Y/main.py
```

Where X is the dimension (1 or 5 typically and Y is the test turbine id.

## Running an Experiment - with pre-trained model

```
poetry run python experiments/wind_power_Xd_test_turbine_Y/main.py --config.restore=logs/wind-energy-regression-one-turbine-20250616-eval/TRAINED_MODEL_FOLDER
```
Where X is the dimension (1 or 5 typically and Y is the test turbine id and where TRAINED_MODEL_FOLDER is the folder name of the pre-trained model for example Jul27_213009_vakr in the case of wind_power_1d_test_turbine_1
