# Adhoc script to demonstrate task encoder

This takes the confg specified in config.py to load a pretrained model. Onc eagain it is vital the config matches the config of the model loaded (in thisc case the 5D model for either test turbine 1 or 6).

As with the diffusion demo, it is possible also to specify what checkpoitn to laod teh model in at in main.py

This takes the provided training / testing data and runs the samples through the task encoder.

It then pickels the input batches (samples) and resulting encodings.
