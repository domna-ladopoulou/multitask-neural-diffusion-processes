# Neural Diffusion Processes (NDP)

This directory contains the original implementation of [Neural Diffusion Processes (NDP)](https://arxiv.org/abs/2206.03992), with minor modifications. The original source code is available in the [official GitHub repository](https://github.com/vdutor/neural-diffusion-processes).

The implementation remains largely unchanged, with the following modifications:

1. **Diffusion process visualisation:** The forward and reverse diffusion processes have been modified to save intermediate states at every timestep, enabling visualisation of the diffusion trajectories.
2. **Additional logging:** Minor logging functionality has been introduced to facilitate monitoring of computationally intensive operations, particularly during inference.

All other components of the original NDP implementation remain unchanged.
