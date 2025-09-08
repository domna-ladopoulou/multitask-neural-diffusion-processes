from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Generator, Union

import jax
import jax.numpy as jnp
import numpy as np
import optax
from simple_pytree import Pytree

ndarray = Union[jnp.ndarray, np.ndarray]
Dtype = Any
Rng = jax.random.KeyArray
Params = optax.Params
Config = Any


@dataclass
class Batch(Pytree):
    x_target: ndarray
    y_target: ndarray
    x_context: ndarray | None = None
    y_context: ndarray | None = None
    mask_target: ndarray | None = None
    mask_context: ndarray | None = None

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
        }


Dataset = Generator[Batch, None, None]
