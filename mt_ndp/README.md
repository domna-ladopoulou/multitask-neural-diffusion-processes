# Multi-Task Neural Diffusion Processes (MT-NDP)

This directory contains the implementation of the **Multi-Task Neural Diffusion Processes (MT-NDP)** architecture, introduced in [Multi-Task Neural Diffusion Processes](https://arxiv.org/abs/2510.03419).

The implementation builds upon the original [Neural Diffusion Processes (NDP)](https://arxiv.org/abs/2206.03992) framework, whose source code is available in the [original GitHub repository](https://github.com/vdutor/neural-diffusion-processes).

The MT-NDP implementation extends the original NDP architecture to support multi-task learning. The main modifications include:

1. **Task encoder:** Introducing a task encoder to capture task-specific information and facilitate information sharing across related tasks.
2. **Architectural integration:** Integrating the task encoder into the existing NDP framework to incorporate task representations into the diffusion-based modelling process.

The original NDP implementation, with minor modifications, is retained separately in the `neural-diffusion-processes/` directory.
