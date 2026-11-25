# Diffusion Process Demonstration

This directory contains an ad hoc script demonstrating the diffusion process in the NDP framework.

The `main.py` script uses the diffusion process configuration specified in `config.py`. **The configuration must match that of the pretrained model being loaded**, which, in this case, corresponds to the 3D model.

The configuration includes specifying the pretrained model used to demonstrate the forward and reverse diffusion processes.

In addition to specifying the model name and directory, the checkpoint to be restored can be selected in `main.py`.
