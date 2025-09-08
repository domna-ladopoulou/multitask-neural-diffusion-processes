# Wind Energy Prediction with MT-NDP


## Code
Please see ORIGINAL_README.md for the original README for the neural-diffsuion-proccess developed in https://arxiv.org/abs/2206.03992 and the code was also provided at https://github.com/vdutor/neural-diffusion-processes.

The neural-diffsuion-processes folder is left mostly unchagned except for:
1. forcing the forward and backward diffusion processes to save out data at al timesteps so we can visualise the processes
2. Added logging to the code as inference in particular can be quite slow

the mt-ndp folder is a copy of neural-diffsuion-processses fodler with some changes to:
1. add the new Encoder to the architecture
2. Ensure this task encoder integrates with the downstream NDP

## Experiments
Please see the experiments fodler for the various experiments we have conducted. Experiments/README.md explains how to run new or pre-trained models. Within each experiment, the config specifies the paramaters used to train the model as well as which input features were used. Further details can also be found in each of their respective README.md.

## Data
The data is from Kelmarsh wind farm and is publically available at: https://zenodo.org/records/5841834. In this case, the data was alreayd pre-cleaned by: https://arxiv.org/abs/2404.16496 and provided to us.

The processes scaled data needed for the experiments can be found in data/.

## Poetry
If you have any issues with setup using poetry, please not there is also a requirements.txt file that can be used.

Also notaworthy, is that JAX is not in the poetry env as this should be installed yourself. THis allows you to install with GPU support. Further details of this are provided in the ORIGINAL_README.md.

