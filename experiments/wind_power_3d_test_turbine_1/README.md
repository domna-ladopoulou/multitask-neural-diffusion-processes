# 3D NDP - Test on Turbine 1

- Model folder: Jul28_084845_sejk
- trains on turbines 2-6
- tests on turbine 1
- uses 3 input feature (mean wind speed, mean sine of wind direction, mean cosine of wind direction) to predict mean power output
- all features are normalised
- full details of config in config.py
- also in this dir are the ndp_diffusion_demo_model_Jul28_084845_sejk_step_*_batch_2
  - This ran teh diffusion process for the model after 750, 1500, and 3750 (fully trained) steps
  - Ultimately, we use the 1500 step model for demonstration purposes as 750 is undertrained, and 3750 is so well trained that we cannot distinguish between the conditional and unconditional sampling