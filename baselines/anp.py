import torch
from torch import nn
from torch.nn import functional as F


class LatentSetEncoder(nn.Module):
    def __init__(self, x_dim=5, hidden=128, z_dim=32):
        super().__init__()
        self.point = nn.Sequential(
            nn.Linear(x_dim + 1, hidden), nn.ReLU(),
            nn.Linear(hidden, hidden), nn.ReLU(),
            nn.Linear(hidden, hidden), nn.ReLU(),
        )
        self.to_stats = nn.Sequential(
            nn.Linear(hidden, hidden), nn.ReLU(),
            nn.Linear(hidden, 2 * z_dim),
        )

    def forward(self, x, y, valid):
        h = self.point(torch.cat([x, y], dim=-1))
        w = valid.unsqueeze(-1)
        r = (h * w).sum(1) / w.sum(1).clamp_min(1.0)
        mu, raw = self.to_stats(r).chunk(2, -1)
        return mu, 0.05 + F.softplus(raw)


class DeterministicCrossAttention(nn.Module):
    def __init__(self, x_dim=5, hidden=128, heads=8):
        super().__init__()
        self.key = nn.Sequential(
            nn.Linear(x_dim, hidden), nn.ReLU(), nn.Linear(hidden, hidden)
        )
        self.query = nn.Sequential(
            nn.Linear(x_dim, hidden), nn.ReLU(), nn.Linear(hidden, hidden)
        )
        self.value = nn.Sequential(
            nn.Linear(x_dim + 1, hidden), nn.ReLU(),
            nn.Linear(hidden, hidden), nn.ReLU(),
        )
        self.attn = nn.MultiheadAttention(hidden, heads, batch_first=True)

    def forward(self, xc, yc, valid, xt):
        k = self.key(xc)
        q = self.query(xt)
        v = self.value(torch.cat([xc, yc], dim=-1))
        pad = valid < 0.5

        # Handle a context with no observed points, as in the original notebook.
        empty = pad.all(dim=1)
        if empty.any():
            k = k.clone()
            v = v.clone()
            pad = pad.clone()
            k[empty, 0] = 0.0
            v[empty, 0] = 0.0
            pad[empty, 0] = False

        r, _ = self.attn(q, k, v, key_padding_mask=pad, need_weights=False)
        if empty.any():
            r = r.clone()
            r[empty] = 0.0
        return r


class ANPDecoder(nn.Module):
    def __init__(self, x_dim=5, hidden=128, z_dim=32):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(x_dim + hidden + z_dim, hidden), nn.ReLU(),
            nn.Linear(hidden, hidden), nn.ReLU(),
            nn.Linear(hidden, hidden), nn.ReLU(),
            nn.Linear(hidden, 2),
        )

    def forward(self, x, r, z):
        zz = z[:, None, :].expand(-1, x.shape[1], -1)
        out = self.net(torch.cat([x, r, zz], dim=-1))
        return out[..., 0], 0.02 + F.softplus(out[..., 1])


class ANP(nn.Module):
    """ANP-style latent encoder, deterministic cross-attention and decoder."""

    def __init__(self, x_dim=5, hidden=128, z_dim=32, heads=8):
        super().__init__()
        self.latent = LatentSetEncoder(x_dim, hidden, z_dim)
        self.det = DeterministicCrossAttention(x_dim, hidden, heads)
        self.decoder = ANPDecoder(x_dim, hidden, z_dim)

    def training_loss(self, xc, yc, valid, xt, yt):
        """Notebook's variational loss: mean Gaussian NLL + KL / 100."""
        p_mu, p_sd = self.latent(xc, yc, valid)
        vt = torch.ones(xt.shape[:2], device=xt.device)
        q_mu, q_sd = self.latent(xt, yt, vt)
        z = q_mu + q_sd * torch.randn_like(q_mu)
        r = self.det(xc, yc, valid, xt)
        mu, sd = self.decoder(xt, r, z)
        nll = -torch.distributions.Normal(mu, sd).log_prob(yt[..., 0]).mean()
        kl = (
            torch.log(p_sd / q_sd)
            + (q_sd.square() + (q_mu - p_mu).square()) / (2 * p_sd.square())
            - 0.5
        ).sum(-1).mean()
        return nll + kl / 100.0

    @torch.no_grad()
    def sample(self, xc, yc, valid, xt, n_draws=200, chunk_size=50):
        """Sample predictions using only context data (no target-y leakage).

        Expects a single function (batch size 1). Returns [n_draws,T].
        """
        if xc.shape[0] != 1 or xt.shape[0] != 1:
            raise ValueError("sample expects a batch size of one")
        p_mu, p_sd = self.latent(xc, yc, valid)
        r = self.det(xc, yc, valid, xt)
        draws = []
        for start in range(0, n_draws, chunk_size):
            m = min(chunk_size, n_draws - start)
            z = p_mu + p_sd * torch.randn(m, p_mu.shape[-1], device=p_mu.device)
            mu, sd = self.decoder(xt.expand(m, -1, -1), r.expand(m, -1, -1), z)
            draws.append(mu + sd * torch.randn_like(mu))
        return torch.cat(draws, dim=0)
