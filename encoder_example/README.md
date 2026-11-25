# Task Encoder Demonstration

This directory contains an ad hoc script demonstrating the task encoder in the MT-NDP architecture.

The script uses the configuration specified in `config.py` to load a pretrained model. **The configuration must match that of the selected pretrained model**, which, in this case, corresponds to the 5D model for either test turbine 1 or 6.

As with the diffusion demonstration, the checkpoint used to load the pretrained model can be specified in `main.py`.

The script processes the provided training and testing data by passing the input samples through the task encoder.

Finally, the input batches (samples) and their corresponding task encodings are saved as serialised Python objects using `pickle`, allowing them to be loaded for subsequent analysis and visualisation.
