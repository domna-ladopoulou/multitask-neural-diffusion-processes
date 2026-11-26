import haiku as hk
import jax.numpy as jnp
from mt_ndp.model import SetEncoder
from mt_ndp import model as ndp_model


def build_synthetic_mt_ndp(task_embedding_dim: int):
    
    if task_embedding_dim not in (4, 8, 16):
        raise ValueError('Paper-specified task embedding dimensions: 4, 8, 16')

    def forward(x, y, t, xc, yc, mask, mask_context):
        encoder = SetEncoder(
            mlp_layers=4,
            mlp_hidden_dim=64,
            mlp_output_dim=task_embedding_dim,
            use_self_attention=False,
            self_attn_heads=4,  # Constructor argument; encoder attention disabled.
        )
        task_embedding = encoder(xc, yc, mask_context)
        task_embedding = jnp.repeat(task_embedding[:, None, :], x.shape[1], axis=1)
        x_aug = jnp.concatenate([x, task_embedding], axis=-1)
        denoiser = ndp_model.BiDimensionalAttentionModel(
            n_layers=4, hidden_dim=64, num_heads=4, init_zero=True
        )
        return denoiser(x_aug, y, t, mask)

    return hk.transform(forward)
