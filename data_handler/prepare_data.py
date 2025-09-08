import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
import os
import pickle
from typing import Union, List
import pathlib
import jax.numpy as jnp

def get_script_or_parent_dir(n_parents=1) -> pathlib.Path:
    _file = pathlib.Path(__file__)
    for _ in range(n_parents):
        _file = _file.parent
    return _file

def generate_wind_data_one_turbine_npz(
        train_raw_path: str,
        test_raw_path: str,
        num_function_train_samples_per_turbine: int,
        num_function_test_samples: int,
        test_turbine_id: int,
        x_cols: Union[str, List[str]],
        y_col: str,
        output_dir: str,
        dataset_name: str,
        max_n_context: int,
        jump_n_context: int,
        function_sample_size: int,
        seed: int,
        train_turbine_ids: List[int] | None = None
):
    print(f"Loading raw data from {train_raw_path} and {test_raw_path}...")
    train_df = pd.read_pickle(train_raw_path)
    full_train_df = train_df.copy()
    test_df = pd.read_pickle(test_raw_path)

    train_target_df = train_df[train_df.turbine == test_turbine_id]
    test_target_df = test_df[test_df.turbine == test_turbine_id]

    if train_turbine_ids:
        train_df = train_df[train_df.turbine.isin(train_turbine_ids)]
        print(f"Using Turbines {train_turbine_ids} for training... ")
    else:
        print(f"Using all non-test turbines to train...")
        train_df = train_df[train_df.turbine != test_turbine_id]

    if isinstance(x_cols, str):
        x_cols_list = [x_cols]
    else:
        x_cols_list = x_cols

    turbine_scalers = {}
    for tid in full_train_df.turbine.unique():
        turbine_data = full_train_df[full_train_df["turbine"] == tid]
        x = turbine_data[x_cols_list].values
        y = turbine_data[y_col].values.reshape(-1, 1)

        x_scaler = StandardScaler().fit(x)
        y_scaler = StandardScaler().fit(y)
        turbine_scalers[tid] = {
            'x_scaler': x_scaler,
            'y_scaler': y_scaler
        }

    rng = np.random.default_rng(seed=seed)
    print("Generating training samples...")
    x_target_all, y_target_all = [], []
    mt_x_context_all, mt_y_context_all, mt_mask_context_all = [], [], []
    turbine_ids_all = []

    for tid in train_df['turbine'].unique():
        turbine_data = train_df[train_df['turbine'] == tid]
        x = turbine_scalers[tid]['x_scaler'].transform(turbine_data[x_cols_list].values)
        y = turbine_scalers[tid]['y_scaler'].transform(turbine_data[y_col].values.reshape(-1, 1))

        for _ in range(num_function_train_samples_per_turbine):
            idx_target = rng.choice(len(x), function_sample_size, replace=False)
            x_target = x[idx_target]
            y_target = y[idx_target]

            idx_ctx = rng.choice(len(x), max_n_context, replace=False)
            n_ctx = rng.integers(0, max_n_context + 1)

            x_ctx = np.zeros((max_n_context, x.shape[1]))
            y_ctx = np.zeros((max_n_context, 1))
            mask_ctx = np.ones((max_n_context,), dtype=np.float32)

            if n_ctx > 0:
                x_ctx[:n_ctx] = x[idx_ctx[:n_ctx]]
                y_ctx[:n_ctx] = y[idx_ctx[:n_ctx]]
                mask_ctx[:n_ctx] = 0.0

            x_target_all.append(x_target)
            y_target_all.append(y_target)
            mt_x_context_all.append(x_ctx)
            mt_y_context_all.append(y_ctx)
            mt_mask_context_all.append(mask_ctx)
            turbine_ids_all.append(tid)

    x_target_train = np.stack(x_target_all)
    y_target_train = np.stack(y_target_all)
    mt_x_context = np.stack(mt_x_context_all)
    mt_y_context = np.stack(mt_y_context_all)
    mt_mask_context = jnp.array(mt_mask_context_all)
    train_turbine_ids = np.array(turbine_ids_all)

    output_subdir = f"{output_dir}/{dataset_name}"
    os.makedirs(output_subdir, exist_ok=True)

    np.savez(
        os.path.join(output_subdir, "training.npz"),
        x_target=x_target_train,
        y_target=y_target_train,
        x_context=jnp.zeros((len(x_target_train), 0, x_target_train.shape[2]), dtype=jnp.float32),
        y_context=jnp.zeros((len(x_target_train), 0, 1), dtype=jnp.float32),
        mask_context=jnp.zeros((len(x_target_train), 0), dtype=jnp.float32),
        mask_target=jnp.zeros((len(x_target_train), function_sample_size), dtype=jnp.float32),
        mt_x_context=mt_x_context,
        mt_y_context=mt_y_context,
        mt_mask_context=mt_mask_context,
        train_turbine_id=train_turbine_ids
    )

    print("Generating test dataset...")
    target_scaler = turbine_scalers[test_turbine_id]
    x_test_raw = test_target_df[x_cols_list].values
    y_test_raw = test_target_df[y_col].values.reshape(-1, 1)
    x_test_scaled = target_scaler['x_scaler'].transform(x_test_raw)
    y_test_scaled = target_scaler['y_scaler'].transform(y_test_raw)

    # get random rows from the test data set
    total_test_samples = num_function_test_samples * function_sample_size
    test_idx = rng.choice(len(x_test_scaled), total_test_samples, replace=False)
    x_test_scaled = x_test_scaled[test_idx]
    y_test_scaled = y_test_scaled[test_idx]

    x_train_target_raw = train_target_df[x_cols_list].values
    y_train_target_raw = train_target_df[y_col].values.reshape(-1, 1)
    x_train_target_scaled = target_scaler['x_scaler'].transform(x_train_target_raw)
    y_train_target_scaled = target_scaler['y_scaler'].transform(y_train_target_raw)

    total_context_samples = (len(x_train_target_scaled) // num_function_test_samples) * num_function_test_samples
    context_idx = rng.choice(len(x_train_target_scaled), total_context_samples, replace=False)

    context_values = np.arange(0, max_n_context + 1, jump_n_context)
    num_context_sizes = len(context_values)

    x_target_eval = x_test_scaled[:num_function_test_samples * function_sample_size].reshape(num_function_test_samples, function_sample_size, -1)
    y_target_eval = y_test_scaled[:num_function_test_samples * function_sample_size].reshape(num_function_test_samples, function_sample_size, -1)

    x_target_eval = np.repeat(x_target_eval, num_context_sizes, axis=0)
    y_target_eval = np.repeat(y_target_eval, num_context_sizes, axis=0)
    mask_target_eval = jnp.zeros((len(x_target_eval), function_sample_size), dtype=jnp.float32)

    x_context_eval = np.zeros((len(x_target_eval), max_n_context, x_test_scaled.shape[1]))
    y_context_eval = np.zeros((len(x_target_eval), max_n_context, 1))
    mask_context_eval = np.ones((len(x_target_eval), max_n_context), dtype=np.float32)

    for i in range(num_function_test_samples):
        idx = context_idx[i * (total_context_samples // num_function_test_samples):(i + 1) * (total_context_samples // num_function_test_samples)]
        for j, n_ctx in enumerate(context_values):
            batch_idx = i * num_context_sizes + j
            x_context_eval[batch_idx, :n_ctx] = x_train_target_scaled[idx[:n_ctx]]
            y_context_eval[batch_idx, :n_ctx] = y_train_target_scaled[idx[:n_ctx]]
            mask_context_eval[batch_idx, :n_ctx] = 0.0

    np.savez(
        os.path.join(output_subdir, "interpolation.npz"),
        x_target=x_target_eval,
        y_target=y_target_eval,
        mask_target=mask_target_eval,
        mask_context=mask_context_eval,
        x_context=x_context_eval,
        y_context=y_context_eval,
        context_sizes=np.tile(context_values, num_function_test_samples)
    )

    with open(os.path.join(output_subdir, "scalers.pkl"), 'wb') as f:
        pickle.dump(turbine_scalers, f)

    print("Done generating per-turbine NPZ files and scalers.")


if __name__ == "__main__":
    # Define paths relative to the project root (assuming script is in experiments/wind_farm/)
    train_data_path = get_script_or_parent_dir(n_parents=4) / "data/train.pkl"
    test_data_path =  get_script_or_parent_dir(n_parents=4) / "data/test.pkl"

    ### --------- One Dimensional - Test on Turbine 1 --------- ###

    print("--- Starting Data Generation for Wind Power Prediction ONE DIMENSIONAL ---")
    # Configuration for data generation
    train_function_sample_size = 100
    num_function_train_samples_per_turbine = 500 # note there are 5 other turbiens so this is 5 * 500 samples
    num_test_samples = 100
    test_turbine_id = 1
    train_turbine_ids = [] # all turbines but the test turbine be default
    max_n_context = 50
    jump_n_context = 25

    assert max_n_context % jump_n_context == 0

    x_columns = ["Wind.speed.me"]  # Can be a list for multiple inputs
    y_column = "Power.me"

    # Output directory for the.npz files and scalers.pkl
    output_data_directory = get_script_or_parent_dir(n_parents=2) / "data"
    subfolder_name = "wind_data_1d_test_turbine_1"

    generate_wind_data_one_turbine_npz(
        train_raw_path=train_data_path,
        test_raw_path=test_data_path,
        num_function_train_samples_per_turbine=num_function_train_samples_per_turbine,
        num_function_test_samples=num_test_samples,
        test_turbine_id=test_turbine_id,
        x_cols=x_columns,
        y_col=y_column,
        output_dir=str(output_data_directory),
        dataset_name=subfolder_name,
        max_n_context=max_n_context,
        jump_n_context=jump_n_context,
        function_sample_size=train_function_sample_size,
        seed=1,
        train_turbine_ids=train_turbine_ids,
    )
    print("--- Data Generation Complete for Wind Power Prediction ONE DIMENSIONAL ---")

    ### --------- Three Dimensional - Test on Turbine 1 --------- ###

    print("--- Starting Data Generation for Wind Power Prediction THREE DIMENSIONAL ---")
    # Configuration for data generation
    train_function_sample_size = 100
    num_function_train_samples_per_turbine = 500  # note there are 5 other turbiens so this is 5 * 500 samples
    num_test_samples = 100
    test_turbine_id = 1
    train_turbine_ids = []  # all turbines but the test turbine be default
    max_n_context = 50
    jump_n_context = 25

    assert max_n_context % jump_n_context == 0

    x_columns = [
        "Wind.speed.me", 'Wind.dir.sin.me', 'Wind.dir.cos.me'
    ]
    y_column = "Power.me"

    # Output directory for the.npz files and scalers.pkl
    output_data_directory = get_script_or_parent_dir(n_parents=2) / "data"
    subfolder_name = "wind_data_3d_test_turbine_1"

    generate_wind_data_one_turbine_npz(
        train_raw_path=train_data_path,
        test_raw_path=test_data_path,
        num_function_train_samples_per_turbine=num_function_train_samples_per_turbine,
        num_function_test_samples=num_test_samples,
        test_turbine_id=test_turbine_id,
        x_cols=x_columns,
        y_col=y_column,
        output_dir=str(output_data_directory),
        dataset_name=subfolder_name,
        max_n_context=max_n_context,
        jump_n_context=jump_n_context,
        function_sample_size=train_function_sample_size,
        seed=1,
        train_turbine_ids=train_turbine_ids,
    )
    print("--- Data Generation Complete for Wind Power Prediction THREE DIMENSIONAL ---")

    ### --------- Five Dimensional - Test on Turbine 1 --------- ###

    print("--- Starting Data Generation for Wind Power Prediction FIVE DIMENSIONAL ---")
    # Configuration for data generation
    train_function_sample_size = 100
    num_function_train_samples_per_turbine = 500  # note there are 5 other turbiens so this is 5 * 500 samples
    num_test_samples = 100
    test_turbine_id = 1
    train_turbine_ids = []  # all turbines but the test turbine be default
    max_n_context = 50
    jump_n_context = 25

    assert max_n_context % jump_n_context == 0

    x_columns = [
        "Wind.speed.me", 'Wind.dir.sin.me', 'Wind.dir.cos.me',
        'Nacelle.temp.me', 'Transformer.temp.me'
    ]
    y_column = "Power.me"

    # Output directory for the.npz files and scalers.pkl
    output_data_directory = get_script_or_parent_dir(n_parents=2) / "data"
    subfolder_name = "wind_data_5d_test_turbine_1"

    generate_wind_data_one_turbine_npz(
        train_raw_path=train_data_path,
        test_raw_path=test_data_path,
        num_function_train_samples_per_turbine=num_function_train_samples_per_turbine,
        num_function_test_samples=num_test_samples,
        test_turbine_id=test_turbine_id,
        x_cols=x_columns,
        y_col=y_column,
        output_dir=str(output_data_directory),
        dataset_name=subfolder_name,
        max_n_context=max_n_context,
        jump_n_context=jump_n_context,
        function_sample_size=train_function_sample_size,
        seed=1,
        train_turbine_ids=train_turbine_ids,
    )
    print("--- Data Generation Complete for Wind Power Prediction FIVE DIMENSIONAL ---")

    ### --------- One Dimensional - Test on Turbine 6 --------- ###

    print("--- Starting Data Generation for Wind Power Prediction ONE DIMENSIONAL (turbine 6)---")
    # Configuration for data generation
    train_function_sample_size = 100
    num_function_train_samples_per_turbine = 500  # note there are 5 other turbiens so this is 5 * 500 samples
    num_test_samples = 100
    test_turbine_id = 6
    train_turbine_ids = []  # all turbines but the test turbine be default
    max_n_context = 50
    jump_n_context = 25

    assert max_n_context % jump_n_context == 0

    x_columns = ["Wind.speed.me"]
    y_column = "Power.me"

    # Output directory for the.npz files and scalers.pkl
    output_data_directory = get_script_or_parent_dir(n_parents=2) / "data"
    subfolder_name = "wind_data_1d_test_turbine_6"

    generate_wind_data_one_turbine_npz(
        train_raw_path=train_data_path,
        test_raw_path=test_data_path,
        num_function_train_samples_per_turbine=num_function_train_samples_per_turbine,
        num_function_test_samples=num_test_samples,
        test_turbine_id=test_turbine_id,
        x_cols=x_columns,
        y_col=y_column,
        output_dir=str(output_data_directory),
        dataset_name=subfolder_name,
        max_n_context=max_n_context,
        jump_n_context=jump_n_context,
        function_sample_size=train_function_sample_size,
        seed=1,
        train_turbine_ids=train_turbine_ids,
    )
    print("--- Data Generation Complete for Wind Power Prediction ONE DIMENSIONAL (turbine 6) ---")

    ### --------- Five Dimensional - Test on Turbine 6 (based on encoder analysis)  --------- ###

    print("--- Starting Data Generation for Wind Power Prediction FIVE DIMENSIONAL (turbine 6)---")
    # Configuration for data generation
    train_function_sample_size = 100
    num_function_train_samples_per_turbine = 500  # note there are 5 other turbiens so this is 5 * 500 samples
    num_test_samples = 100
    test_turbine_id = 6
    train_turbine_ids = []  # all turbines but the test turbine be default
    max_n_context = 50
    jump_n_context = 25

    assert max_n_context % jump_n_context == 0

    x_columns = [
        "Wind.speed.me", 'Wind.dir.sin.me', 'Wind.dir.cos.me',
        'Nacelle.temp.me', 'Transformer.temp.me'
    ]
    y_column = "Power.me"

    # Output directory for the.npz files and scalers.pkl
    output_data_directory = get_script_or_parent_dir(n_parents=2) / "data"
    subfolder_name = "wind_data_5d_test_turbine_6"

    generate_wind_data_one_turbine_npz(
        train_raw_path=train_data_path,
        test_raw_path=test_data_path,
        num_function_train_samples_per_turbine=num_function_train_samples_per_turbine,
        num_function_test_samples=num_test_samples,
        test_turbine_id=test_turbine_id,
        x_cols=x_columns,
        y_col=y_column,
        output_dir=str(output_data_directory),
        dataset_name=subfolder_name,
        max_n_context=max_n_context,
        jump_n_context=jump_n_context,
        function_sample_size=train_function_sample_size,
        seed=1,
        train_turbine_ids=train_turbine_ids,
    )
    print("--- Data Generation Complete for Wind Power Prediction FIVE DIMENSIONAL (turbine 6) ---")