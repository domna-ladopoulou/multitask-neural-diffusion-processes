# NDP

This is almost exactly the same as in the original NDP developed in https://arxiv.org/abs/2206.03992 and the code was also provided at https://github.com/vdutor/neural-diffusion-processes

I made a few changes here to:
1. save out data for all timesteps of the forward and backward processes, so we can visualise the diffusion
2. added some minor logging as some steps of inference can be quite slow