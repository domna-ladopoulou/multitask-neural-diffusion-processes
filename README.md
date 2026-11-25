[![arXiv](https://img.shields.io/badge/arXiv-2510.03419-b31b1b.svg)](https://arxiv.org/abs/2510.03419)

# Multitask Neural Diffusion Processes

This repository contains the code for the paper [Multitask Neural Diffusion Processes](https://arxiv.org/abs/2510.03419) by Joseph Rawson, Domniki Ladopoulou and Petros Dellaportas. We provide the implementation of Multitask Neural Diffusion Processes (MT-NDP), an extension of Neural Diffusion Processes (NDPs) designed to model dependencies across multiple related tasks.

The implementation builds upon the original [Neural Diffusion Processes](https://arxiv.org/abs/2206.03992) framework.

The `neural-diffusion-processes/` directory retains the original implementation, with only minor modifications:

1. **Intermediate diffusion states:** The forward and reverse diffusion processes have been modified to save intermediate states at every timestep, enabling visualisation of the diffusion trajectories.
2. **Additional logging:** Logging functionality has been introduced to facilitate monitoring and debugging, particularly during computationally intensive inference.

The `mt-ndp/` directory extends the original implementation to incorporate the proposed multitask architecture. The main modifications include:

1. **Task encoder:** Introducing a task encoder into the original NDP architecture.
2. **Architectural integration:** Integrating the task encoder with the downstream NDP components to enable multitask modelling.

## Experiments

The `experiments/` directory contains the implementations and configurations for the experiments presented in the paper.

Detailed instructions for running experiments, including the use of pretrained models, are provided in [`experiments/README.md`](experiments/README.md).

Each experiment includes a configuration file specifying the model hyperparameters, training settings, and input features. Additional information is available in the README files within the respective experiment directories.

## Data

The experiments use publicly available data from [Kelmarsh Wind Farm, UK](https://zenodo.org/records/5841834). The dataset was previously cleaned and preprocessed as described in the author's previous published work [Probabilistic Multilayer Perceptrons for Wind Farm Condition Monitoring](https://onlinelibrary.wiley.com/doi/full/10.1002/we.70012). The processed and scaled data used in the experiments are provided in the `data/` directory.

Synthetic regression data are used to investigate the ability of MT-NDPs to learn shared representations across related tasks and generalise to previously unseen functions. The dataset consists of 25 synthetic regression tasks, with 20 used for training and five held out for evaluation. Each task is generated from a combination of sinusoidal, absolute-value, and hyperbolic tangent components, allowing the resulting functions to exhibit different nonlinear patterns and local changes. Inputs are sampled uniformly from the interval [-3, 3], with independent Gaussian observation noise of standard deviation 0.1.

The School Examination (Exam) dataset is used as an additional real-world benchmark for multi-task regression. Originally described by Goldstein et al. (1993), the dataset contains examination results from 4,059 students across 65 schools. It includes normalised examination scores, student-level characteristics, and school-level information. Each school is treated as a separate task, allowing the evaluation of MT-NDPs on previously unseen schools using limited context observations.

## Alternative Installation and JAX Support

The project uses Poetry for dependency management. If difficulties arise during installation, a `requirements.txt` file is also provided as an alternative.

JAX is not included in the Poetry environment by default and must be installed separately. This allows users to select the appropriate installation for their hardware, including GPU acceleration.

For further installation instructions and configuration details, please refer to [README.md](https://github.com/vdutor/neural-diffusion-processes).
