# 5D MT-NDP - Test on Turbine 1

- Model folder: Aug08_004613_gtmm
    - **IMPORTANT** we purposefully disabled the REPAINT algorithm step for this model
- trains on turbines 2-6
- tests on turbine 1
- uses 5 input feature (mean wind speed, mean sine of wind direction, mean cosine of wind direction, mean transformer temperature, and mean nacelle temperature) to predict mean power output
- all features are normalised
- full details of config in config.py
