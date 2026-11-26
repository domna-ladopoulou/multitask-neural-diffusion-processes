# Additional Experiments

This directory contains the model architectures and experimental configurations for the two additional datasets considered in the accompanying paper, Multi-task Neural Diffusion Processes.

# Experiments

* synthetic/: Synthetic non-smooth function experiments investigating the effect of task embedding dimensions (4, 8, 16) and context sizes (0, 5, 10, 25).
* schools/: School examination experiments, including the main evaluation and leave-one-school-out setting, with task embedding dimensions of 16 and 4, respectively.

Each subdirectory contains:

* model.py: MT-NDP model architecture.
* config.py: Paper-specified experimental configurations.
* README.md: Dataset-specific implementation details.

These files document the architectures and configurations described in the paper. 
