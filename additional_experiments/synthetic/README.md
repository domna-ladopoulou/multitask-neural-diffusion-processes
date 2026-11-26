# Synthetic experiments — architecture and paper settings

This directory documents the synthetic MT-NDP experiment described in the accompanying paper. 

* model.py: synthetic MT-NDP architecture: four-layer, width-64 SetEncoder (self-attention disabled), task embedding dimension 4/8/16, and four-layer, width-64 BiDimensionalAttentionModel with four attention heads. The task embedding is concatenated to the one-dimensional input at each target point.
* config.py: the manuscript experiment settings (50 diffusion steps, cosine schedule, context sizes 0/5/10/25, Adam learning rate 1e-4, batch size 8, up to 10,000 training steps, and 100 predictive draws).
  
Dependencies: JAX, Haiku, and the existing project’s mt_ndp package. The model.py function expects batched arrays and is not a standalone runnable experiment.
