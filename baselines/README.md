# Baseline models

This directory contains standalone implementations of the probabilistic regression baselines used for comparison with **Multi-task Neural Diffusion Processes (MT-NDPs)** in the accompanying paper.

| File | Model | Implementation |
| --- | --- | --- |
| [`anp.py`](anp.py) | Attentive Neural Process (ANP) | Latent set encoder, deterministic multi-head cross-attention, Gaussian decoder, variational training loss, and predictive sampling. |
| [`mtnp.py`](mtnp.py) | Multi-task Neural Process (MT-NP) | Shared latent neural process with a permutation-invariant set encoder, Gaussian decoder, variational training loss, and predictive sampling. |
| [`mtgp.py`](mtgp.py) | Multi-task Gaussian Process (MT-GP) | Exact Gaussian process with an intrinsic coregionalisation (ICM) covariance: an ARD RBF input kernel and a rank-2 task covariance. |

The neural-process implementations use **PyTorch**; the MT-GP implementation uses **GPyTorch**. The supplied model definitions reflect the five-input-feature wind-energy baseline setting.

## Scope

These files provide **model implementations**, rather than end-to-end experiment reproduction. For the baseline descriptions and experimental protocol, refer to the paper's **Baseline Models**, **Experimental Setup and Implementation Details**, and **Baseline Implementation Details** sections.
