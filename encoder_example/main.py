import os
import pathlib
import jax
import haiku as hk
import jax.numpy as jnp
import pandas as pd
from tqdm.auto import tqdm
import numpy as np
import tensorflow as tf
import tensorflow_datasets as tfds
import matplotlib.pyplot as plt
import optax
from functools import partial

import jaxlinop

from model_load_helpers import get_all_checkpoint_step_to_name

# Disable all GPUs for TensorFlow. Load data using CPU.
tf.config.set_visible_devices([], 'GPU')
AUTOTUNE = tf.data.experimental.AUTOTUNE

from ml_tools.config_utils import setup_config
from ml_tools.state_utils import TrainingState
from ml_tools import state_utils
from ml_tools import writers
from ml_tools import actions

import mt_ndp as ndp
from mt_ndp.types import Dataset, Batch, Rng
from mt_ndp.model import BiDimensionalAttentionModel
from mt_ndp.process import cosine_schedule, GaussianDiffusion
from mt_ndp.model import SetEncoder

from config import Config

import logging
logger = logging.getLogger("encoder-demo")
logger.setLevel(logging.INFO)
console_handler = logging.StreamHandler()
formatter = logging.Formatter(
    fmt="%(asctime)s - %(levelname)-8s | %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S"
)
console_handler.setFormatter(formatter)
logger.addHandler(console_handler)


config: Config = setup_config(Config)
logger.info(f"{config = }")

TEST_TURBINE_ID = config.test_turbine_id
HERE = pathlib.Path(__file__).parent.parent / "experiments" / f"wind_power_5d_test_turbine_{TEST_TURBINE_ID}_mt"
logger.info(f"{HERE = }")

key = jax.random.PRNGKey(config.seed)

@hk.without_apply_rng
@hk.transform
def network(t, y, x, mask, x_context, y_context, mask_context):
    # do task encoding step
    task_encoder = SetEncoder(
        mlp_layers=config.task_encoder.n_layers,
        mlp_hidden_dim=config.task_encoder.hidden_dim,
        mlp_output_dim=config.task_encoder.embed_dim,
        use_self_attention=False,  # NOT IMPLEMENTED
        self_attn_heads=4,  # NOT IMPLEMENTED
    )
    r = task_encoder(x_context, y_context, mask_context)  # [B, r_dim]
    r = jnp.repeat(r[:, None, :], x.shape[1], axis=1)  # repeat across target points
    x = jnp.concatenate([x, r], axis=-1)

    # do the usual NDP step
    model = BiDimensionalAttentionModel(
        n_layers=config.network.n_layers,
        hidden_dim=config.network.hidden_dim,
        num_heads=config.network.num_heads,
    )
    return model(x, y, t, mask)

@jax.jit
@jax.jit
def net(params, t, yt, x, mask, x_context, y_context, mask_context, *, key):
    del key  # the network is deterministic
    return network.apply(
        params,
        t[None],
        yt[None],
        x[None],
        mask[None],
        x_context[None],
        y_context[None],
        mask_context[None]
    )[0]

@hk.without_apply_rng
@hk.transform
def encoder_only_network(x_context, y_context, mask_context):
    encoder =  SetEncoder(
        mlp_layers=config.task_encoder.n_layers,
        mlp_hidden_dim=config.task_encoder.hidden_dim,
        mlp_output_dim=config.task_encoder.embed_dim,
        use_self_attention=False,  # TODO # NOT IMPLEMENTED
        self_attn_heads=4,  # TODO # NOT IMPLEMENTED
    )
    return encoder(x_context, y_context, mask_context)

@jax.jit
def encoder_only_net(params, x_context, y_context, mask_context, *, key):
    del key  # the network is deterministic
    return encoder_only_network.apply(
        params,
        x_context,
        y_context,
        mask_context,
    )

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
        init_rng, t=t, y=batch.y_target, x=batch.x_target, mask=batch.mask_target,
        x_context=batch.x_context,y_context=batch.y_context, mask_context=batch.mask_context,
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

halfway = False # if we did want to see a less trained model

if (experiment_dir_if_exists / "checkpoints").exists():
    step_to_name = get_all_checkpoint_step_to_name(str(experiment_dir_if_exists))
    logger.info(f"All saved model checkpoints are:\n{step_to_name}")
    if step_to_name is None:
        logger.warning(f"No models saved - nothing to do")
        raise ValueError(f"No models saved - nothing to do")

    step_to_name = pd.Series(step_to_name).sort_index()

    # can select other indices here
    # index = state_utils.find_latest_checkpoint_step_index(str(experiment_dir_if_exists))
    _maybe_index = step_to_name.iloc[(-1 if halfway == False else -2)] # the final model
    index = _maybe_index.replace("ckpt-","").replace(".eqx","")
    if index is not None:
        state = state_utils.load_checkpoint(state, str(experiment_dir_if_exists), step_index=index)
        logger.info(f"Restored checkpoint at step {state.step}")

logger.info(f"Loaded model checkpoint from {experiment_dir_if_exists}")
logger.info("Initialising model for encoder only")
net_with_params = partial(encoder_only_net, state.params_ema)



_model_nane = config.restore.split('\\')[-1]
output_file = HERE / f"mt_ndp_encoder_demo_model_{_model_nane}_step_{state.step}.npz"
logger.info(f"{output_file = }")

# now we want to load:
# 1. all the training data so we can get the embeddings for the 5 training turbines
# 2. the testing data so we can get the embedding for tubine 1 (the test turbine)

from dataclasses import dataclass
from typing import Any, Generator, Union
from simple_pytree import Pytree
ndarray = Union[jnp.ndarray, np.ndarray]

@dataclass
class NewBatch(Pytree):
    x_target: ndarray
    y_target: ndarray
    x_context: ndarray | None = None
    y_context: ndarray | None = None
    mask_target: ndarray | None = None
    mask_context: ndarray | None = None
    turbine_id: ndarray | None = None

    # helper function so we can dump this into pickle without needing to import the ndp module
    def to_dict(self) -> dict:
        """Convert the Batch into a plain dict, useful for pickling or saving."""
        return {
            "x_target": self.x_target,
            "y_target": self.y_target,
            "x_context": self.x_context,
            "y_context": self.y_context,
            "mask_target": self.mask_target,
            "mask_context": self.mask_context,
            "turbine_id": self.turbine_id,
        }


NewDataset = Generator[NewBatch, None, None]

### new get data (not actualyl for training or testing

def get_data(
    dataset: str,
    input_dim: int = 1,
    train: bool = True,
    batch_size: int = 1,
    num_epochs: int = 1,
    test_turbine_id: int | None = None,
) -> NewDataset:
    assert test_turbine_id is not None
    task = "training" if train else "interpolation"
    data_path = pathlib.Path(__file__).parent / "../data" / f"{dataset}/{task}.npz"
    logger.info(f"Loading data from {data_path} for {train = }")
    data = np.load(data_path)

    if train:
        ds = tf.data.Dataset.from_tensor_slices({
            "x_target": data["x_target"].astype(np.float32),
            "y_target": data["y_target"].astype(np.float32),
            "x_context": data["mt_x_context"].astype(np.float32),
            "y_context": data["mt_y_context"].astype(np.float32),
            "mask_context": data["mt_mask_context"].astype(np.float32),
            # "mask_target": data["mask_target"].astype(np.float32),
            "turbine_id": data["train_turbine_id"].astype(np.float32)
        })
    else:
        ds = tf.data.Dataset.from_tensor_slices({
            "x_target": data["x_target"].astype(np.float32),
            "y_target": data["y_target"].astype(np.float32),
            "x_context": data["x_context"].astype(np.float32),
            "y_context": data["y_context"].astype(np.float32),
            "mask_context": data["mask_context"].astype(np.float32),
            # "mask_target": data["mask_target"].astype(np.float32),
            "turbine_id": jnp.ones_like(data["mask_context"].astype(np.float32)[:, 0]) * test_turbine_id
        })

    # no shuffling here, we just use it to get data
    # if train:
    #     ds = ds.repeat(count=num_epochs)
    #     ds = ds.shuffle(seed=42, buffer_size=10000)
    ds = ds.batch(batch_size, drop_remainder=True)
    ds = ds.prefetch(AUTOTUNE)
    ds = ds.as_numpy_iterator()
    return map(lambda d: NewBatch(**d), ds)


ds_train: NewDataset = get_data(
    config.dataset,
    input_dim=config.input_dim,
    train=True,
    batch_size=config.batch_size,
    num_epochs=config.num_epochs,
    test_turbine_id=TEST_TURBINE_ID
)

ds_test = get_data(
    config.dataset,
    input_dim=config.input_dim,
    train=False,
    batch_size=config.eval.batch_size,
    num_epochs=1,
    test_turbine_id=TEST_TURBINE_ID, # TODO: could add to config
)

# note all batch sizes are 1 for ease

all_ids = []
all_embeddings = []
all_x_context = []
all_y_context = []
all_mask_context = []

for batch in ds_train:
    _id = batch.turbine_id
    _embedding =  net_with_params(batch.x_context, batch.y_context, batch.mask_context, key=key)
    # print(_id, _embedding)
    # print(f"{_id.shape = }, {_embedding.shape = }, "
    #       f"{batch.x_context.shape = }, {batch.y_context.shape = }, "
    #       f"{batch.mask_context.shape = }")
    all_ids.append(_id)
    all_embeddings.append(_embedding)
    all_x_context.append(batch.x_context)
    all_y_context.append(batch.y_context)
    all_mask_context.append(batch.mask_context)


for _i, batch in enumerate( ds_test ):
    _id = batch.turbine_id

    _embedding =  net_with_params(batch.x_context, batch.y_context, batch.mask_context, key=key)

    all_ids.append(_id)
    all_embeddings.append(_embedding)
    all_x_context.append(batch.x_context)
    all_y_context.append(batch.y_context)
    all_mask_context.append(batch.mask_context)

final_ids = np.concatenate(all_ids, axis=0)
final_embeddings = np.concatenate(all_embeddings, axis=0)
final_x_context = np.concatenate(all_x_context, axis=0)
final_y_context = np.concatenate(all_y_context, axis=0)
final_mask_context = np.concatenate(all_mask_context, axis=0)

logger.info(f"{final_ids.shape = }, {final_embeddings.shape = }, "
      f"{final_x_context.shape = }, {final_y_context.shape = }, "
      f"{final_mask_context.shape = }")

np.savez(
    output_file,
    turbine_id = final_ids,
    embeddings = final_embeddings,
    x_context = final_x_context,
    y_context = final_y_context,
    mask_context = final_mask_context,
)

logger.info(f"Saved file to {output_file}")
