# To see source code from old pacakge version...
# import inspect
# from ml_tools import state_utils
# print(inspect.getsource(state_utils.find_latest_checkpoint_step_index))
# print(inspect.getsource(state_utils))

import os
import equinox as eqx
import optax
from jaxtyping import Array, PyTree

_DIRECTORY_PATH_CHECKPOINTS = "{path}/checkpoints"
_STEP_FILENAME_PREFIX = "ckpt-{step}"
_STEP_FILENAME_EXT = ".eqx"

def get_all_checkpoint_step_to_name(directory_path: str):
    path = _DIRECTORY_PATH_CHECKPOINTS.format(path=directory_path)
    if not os.path.exists(path):
        return None
    step_to_index = {
        _checkpoint_filename_to_index(name): name
        for name in os.listdir(path)
        if name.endswith(_STEP_FILENAME_EXT)
    }
    if len(step_to_index) == 0:
        return None
    return step_to_index

def find_latest_checkpoint_step_index(directory_path: str):
    path = _DIRECTORY_PATH_CHECKPOINTS.format(path=directory_path)
    if not os.path.exists(path):
        return None
    indices = [
        _checkpoint_filename_to_index(name)
        for name in os.listdir(path)
        if name.endswith(_STEP_FILENAME_EXT)
    ]
    if len(indices) == 0:
        return None
    return max(indices)

def _index_to_checkpoint_filename(step_index: int):
    return _STEP_FILENAME_PREFIX.format(step=step_index) + _STEP_FILENAME_EXT


def _checkpoint_filename_to_index(filename: str):
    start = filename.index("-") + 1
    end = filename.index(_STEP_FILENAME_EXT)
    filename_part = filename[start:end]
    return int(filename_part)
