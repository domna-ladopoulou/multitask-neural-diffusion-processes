import haiku as hk
import jax.numpy as jnp


def build_school_mt_ndp(SetEncoder, ndp_model, *, embedding_dim=16):

    if embedding_dim not in (4, 16):
        raise ValueError('Paper-described school embedding dimensions: 16 (main), 4 (LOSO).')

    def forward(x, y, t, xc, yc, mask, mask_context):
        context_unbatched = xc.ndim == 2
        if context_unbatched:
            xc, yc, mask_context = xc[None], yc[None], mask_context[None]
        encoder = SetEncoder(
            mlp_layers=4, mlp_hidden_dim=64, mlp_output_dim=embedding_dim,
            use_self_attention=False, self_attn_heads=4,
        )
        embedding = encoder(xc, yc, mask_context)
        target_unbatched = x.ndim == 2
        if target_unbatched:
            embedding = jnp.broadcast_to(embedding[0][None, :], (x.shape[0], embedding_dim))
            x_aug = jnp.concatenate((x, embedding), axis=-1)[None]
            y, t, mask = y[None], jnp.atleast_1d(t), mask[None]
        else:
            embedding = jnp.broadcast_to(embedding[:, None, :], (*x.shape[:2], embedding_dim))
            x_aug = jnp.concatenate((x, embedding), axis=-1)
        denoiser = ndp_model.BiDimensionalAttentionModel(
            n_layers=4, hidden_dim=64, num_heads=4, init_zero=True,
        )
        result = denoiser(x_aug, y, t, mask)
        return result[0] if target_unbatched else result

    return hk.transform(forward)
