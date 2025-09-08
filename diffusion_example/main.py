import os
import pathlib
import pickle
import jax
import haiku as hk
import jax.numpy as jnp
import pandas as pd
from tqdm.auto import tqdm
import tensorflow as tf
import tensorflow_datasets as tfds
import matplotlib.pyplot as plt
import optax
from functools import partial
import jaxlinop
from gpjax.gaussian_distribution import GaussianDistribution

from diffusion_example.model_load_helpers import get_all_checkpoint_step_to_name

# Disable all GPUs for TensorFlow. Load data using CPU.
tf.config.set_visible_devices([], 'GPU')
AUTOTUNE = tf.data.experimental.AUTOTUNE

from ml_tools.config_utils import setup_config
from ml_tools.state_utils import TrainingState
from ml_tools import state_utils
from ml_tools import writers
from ml_tools import actions

import neural_diffusion_processes as ndp
from neural_diffusion_processes.types import Dataset, Batch, Rng
from neural_diffusion_processes.model import BiDimensionalAttentionModel
from neural_diffusion_processes.process import cosine_schedule, GaussianDiffusion

from config import Config

import logging
logger = logging.getLogger("1d-turbine-diffusion-demo")
logger.setLevel(logging.INFO)
console_handler = logging.StreamHandler()
formatter = logging.Formatter(
    fmt="%(asctime)s - %(levelname)-8s | %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S"
)
console_handler.setFormatter(formatter)
logger.addHandler(console_handler)

HERE = pathlib.Path(__file__).parent.parent / "experiments" / "wind_power_3d_test_turbine_1"

config: Config = setup_config(Config)
key = jax.random.PRNGKey(config.seed)

# Set up diffusion process
beta_t = cosine_schedule(
    config.diffusion.beta_start,
    config.diffusion.beta_end,
    config.diffusion.timesteps
)
process = GaussianDiffusion(beta_t)


@hk.without_apply_rng
@hk.transform
def network(t, y, x, mask):
    model = BiDimensionalAttentionModel(
        n_layers=config.network.n_layers,
        hidden_dim=config.network.hidden_dim,
        num_heads=config.network.num_heads,
    )
    return model(x, y, t, mask)


@jax.jit
def net(params, t, yt, x, mask, *, key):
    del key  # the network is deterministic
    # NOTE: Network awkwardly requires a batch dimension for the inputs
    return network.apply(params, t[None], yt[None], x[None], mask[None])[0]


def loss_fn(params, batch: Batch, key):
    net_with_params = partial(net, params)
    kwargs = dict(num_timesteps=config.diffusion.timesteps, loss_type=config.loss_type)
    return ndp.process.loss(process, net_with_params, batch, key, **kwargs)


learning_rate_schedule = optax.warmup_cosine_decay_schedule(
    init_value=config.optimizer.init_lr,
    peak_value=config.optimizer.peak_lr,
    warmup_steps=config.steps_per_epoch * config.optimizer.num_warmup_epochs,
    decay_steps=config.steps_per_epoch * config.optimizer.num_decay_epochs,
    end_value=config.optimizer.end_lr,
)

optimizer = optax.chain(
    optax.clip_by_global_norm(1.0),
    optax.scale_by_adam(),
    optax.scale_by_schedule(learning_rate_schedule),
    optax.scale(-1.0),
)


@jax.jit
def init(batch: Batch, key: Rng) -> TrainingState:
    key, init_rng = jax.random.split(key)
    t = 1. * jnp.zeros((batch.x_target.shape[0]))
    initial_params = network.init(
        init_rng, t=t, y=batch.y_target, x=batch.x_target, mask=batch.mask_target
    )
    initial_opt_state = optimizer.init(initial_params)
    return TrainingState(
        params=initial_params,
        params_ema=initial_params,
        opt_state=initial_opt_state,
        key=key,
        step=0,
    )


# Load trained model
if config.eval.float64:
    from jax.config import config as jax_config
    jax_config.update("jax_enable_x64", True)

batch_init = Batch(
    x_target=jnp.zeros((config.batch_size, 10, config.input_dim)),
    y_target=jnp.zeros((config.batch_size, 10, 1)),
    x_context=jnp.zeros((config.batch_size, 10, config.input_dim)),
    y_context=jnp.zeros((config.batch_size, 10, 1)),
    mask_context=jnp.zeros((config.batch_size, 10)),
    mask_target=jnp.zeros((config.batch_size, 10)),
)
state = init(batch_init, jax.random.PRNGKey(config.seed))

experiment_dir_if_exists = HERE / config.restore
if config.restore != "":
    assert os.path.exists(experiment_dir_if_exists), f"Cannot restore from {config.restore} as path {experiment_dir_if_exists} does not exist..."
    logger.info(f"Path {experiment_dir_if_exists} does exist!")
else:
    assert False

batch_id = 2
model_step = None

if (experiment_dir_if_exists / "checkpoints").exists():
    step_to_name = get_all_checkpoint_step_to_name(str(experiment_dir_if_exists))
    logger.info(f"All saved model checkpoints are:\n{step_to_name}")
    if step_to_name is None:
        logger.warning(f"No models saved - nothing to do")
        raise ValueError(f"No models saved - nothing to do")

    step_to_name = pd.Series(step_to_name).sort_index()

    # can select other indices here
    # index = state_utils.find_latest_checkpoint_step_index(str(experiment_dir_if_exists))
    # _maybe_index = step_to_name.iloc[-1] # the final model
    # _maybe_index = "ckpt-750.eqx" # too undertrained to see
    _maybe_index = "ckpt-1500.eqx" # trained for 1500 steps

    index = _maybe_index.replace("ckpt-","").replace(".eqx","")
    if index is not None:
        state = state_utils.load_checkpoint(state, str(experiment_dir_if_exists), step_index=index)
        logger.info(f"Restored checkpoint at step {state.step}")

_model_nane = config.restore.split('\\')[-1]
output_path = HERE / f"ndp_diffusion_demo_model_{_model_nane}_step_{state.step}_batch_{batch_id}.pkl"
logger.info(f"{output_path = }")

logger.info(f"Loaded model checkpoint from {experiment_dir_if_exists}")
net_with_params = partial(net, state.params_ema)

num_samples = config.eval.n_samples_per_function
logger.info(f"Using {num_samples} samples per function for backward trajectories.")


if (experiment_dir_if_exists / "batches").exists():
    batch_path = experiment_dir_if_exists / f"batches/{batch_id}.pkl"
else:
    assert False

with open(batch_path, "rb") as f:
    batch_dict, _ = pickle.load(f)

x_target = batch_dict["x_target"][0] # 0 as there was a batch dim of 1!
y_target = batch_dict["y_target"][0]
x_context = batch_dict["x_context"][0]
y_context = batch_dict["y_context"][0]
mask_context = batch_dict["mask_context"][0]

num_points = x_target.shape[0]
timesteps = list(range(config.diffusion.timesteps))

logger.info(f"Loaded batch {batch_id}: {num_points} target points, {len(x_context)} context points.")
logger.info(f"Saving ALL timesteps: 0, 1, ..., {config.diffusion.timesteps - 1}")

# --------------- FORWARD ---------------
logger.info("Running forward diffusion trajectories...")
forward_trajectories = []
for t in tqdm(timesteps, desc="Forward"):
    key, subkey = jax.random.split(key)
    yt, _ = process.forward(subkey, y_target, jnp.array(t))
    forward_trajectories.append(yt)
forward_trajectories = jnp.stack(forward_trajectories)  # [num_timesteps, N, y_dim]

# --------------- BACKWARD (Conditional) ---------------
logger.info("Sampling conditional backward trajectories...")
conditional_trajectories = []
for i in tqdm(range(num_samples), desc="Conditional"):
    key, subkey = jax.random.split(key)
    y_all = process.full_conditional_trajectory(
        subkey,
        x_target,
        mask=None,
        x_context=x_context,
        y_context=y_context,
        mask_context=mask_context,
        model_fn=net_with_params
    )
    conditional_trajectories.append(y_all)  # [num_timesteps, N, y_dim]
    logger.info(f"y_all shape at {i}th sample: {y_all.shape}")
conditional_trajectories = jnp.stack(conditional_trajectories)  # [num_samples, num_timesteps, N, y_dim]

# --------------- BACKWARD (Unconditional) ---------------
logger.info("Sampling unconditional backward trajectories...")
unconditional_trajectories = []
for i in tqdm(range(num_samples), desc="Unconditional"):
    key, subkey = jax.random.split(key)
    y_all = process.full_unconditional_trajectory(
        subkey,
        x_target,
        mask=None,
        model_fn=net_with_params,
        output_dim=y_target.shape[-1]
    )
    logger.info(f"y_all shape at {i}th sample: {y_all.shape}")
    unconditional_trajectories.append(y_all)  # [num_timesteps, N, y_dim]
unconditional_trajectories = jnp.stack(unconditional_trajectories)  # [num_samples, num_timesteps, N, y_dim]


# --------------- SAVE --------------
save_dict = {
    "x_target": x_target,
    "y_target": y_target,
    "x_context": x_context,
    "y_context": y_context,
    "mask_context": mask_context,
    "timesteps": timesteps,
    "forward": forward_trajectories,
    "unconditional": unconditional_trajectories,
    "conditional": conditional_trajectories,
}

with open(output_path, "wb") as f:
    pickle.dump(save_dict, f)

logger.info(f"Saved raw trajectories to {output_path}")
logger.info((f"{forward_trajectories.shape = }")
logger.info((f"{unconditional_trajectories.shape = }")
logger.info((f"{conditional_trajectories.shape = }")