import torch
from torch import nn
from torch.nn import functional as F


class SetLatentEncoder(nn.Module):
    """Permutation-invariant encoder for a padded set of (x, y) pairs."""

    def __init__(self, x_dim=5, hidden=128, z_dim=32):
        super().__init__()
        self.point = nn.Sequential(
            nn.Linear(x_dim + 1, hidden), nn.ReLU(),
            nn.Linear(hidden, hidden), nn.ReLU(),
            nn.Linear(hidden, hidden), nn.ReLU(),
        )
        self.to_stats = nn.Sequential(
            nn.Linear(hidden, hidden), nn.ReLU(), nn.Linear(hidden, 2 * z_dim)
        )

    def forward(self, x, y, valid):
        """Return latent mean and standard deviation; valid is 1 for real points."""
        h = self.point(torch.cat([x, y], dim=-1))
        w = valid.unsqueeze(-1)
        r = (h * w).sum(1) / w.sum(1).clamp_min(1.0)
        mu, raw = self.to_stats(r).chunk(2, -1)
        return mu, 0.05 + F.softplus(raw)


class NPDecoder(nn.Module):
    """Independent Gaussian predictive decoder conditioned on a shared latent."""

    def __init__(self, x_dim=5, z_dim=32, hidden=128):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(x_dim + z_dim, hidden), nn.ReLU(),
            nn.Linear(hidden, hidden), nn.ReLU(),
            nn.Linear(hidden, hidden), nn.ReLU(),
            nn.Linear(hidden, 2),
        )

    def forward(self, x, z):
        zz = z[:, None, :].expand(-1, x.shape[1], -1)
        out = self.net(torch.cat([x, zz], -1))
        return out[..., 0], 0.02 + F.softplus(out[..., 1])


class LatentMTNP(nn.Module):
    """Shared latent NP used as the MT-NP baseline in the experiment."""
    
    def __init__(self):
        super().__init__()
        self.encoder = SetLatentEncoder()
        self.decoder = NPDecoder()


def training_loss(model, x_context, y_context, context_valid, x_target, y_target):
    """Original objective: Gaussian NLL + 0.01 * KL(q(z|T) || p(z|C)).

    Target outputs must have shape [batch, targets, 1].
    """
    p_mu, p_sd = model.encoder(x_context, y_context, context_valid)
    target_valid = torch.ones(x_target.shape[:2], device=x_target.device)
    q_mu, q_sd = model.encoder(x_target, y_target, target_valid)
    z = q_mu + q_sd * torch.randn_like(q_mu)
    mu, sd = model.decoder(x_target, z)
    nll = -torch.distributions.Normal(mu, sd).log_prob(y_target[..., 0]).mean()
    kl = (
        torch.log(p_sd / q_sd)
        + (q_sd.pow(2) + (q_mu - p_mu).pow(2)) / (2 * p_sd.pow(2))
        - 0.5
    ).sum(-1).mean()
    return nll + kl / 100.0


@torch.no_grad()
def sample(model, x_context, y_context, context_valid, x_target, n_draws=1):
    """Draw predictive samples, shape [n_draws, batch, targets]."""
    p_mu, p_sd = model.encoder(x_context, y_context, context_valid)
    draws = []
    for _ in range(n_draws):
        z = p_mu + p_sd * torch.randn_like(p_mu)
        mu, sd = model.decoder(x_target, z)
        draws.append(mu + sd * torch.randn_like(mu))
    return torch.stack(draws)
