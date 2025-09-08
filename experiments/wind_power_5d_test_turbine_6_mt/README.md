# 5D MT-NDP - Test on Turbine 6

- Model folders:
    - Aug08_031202_rift_copy: Here we purposefully disabled REPAINT
    - Aug08_031202_rift_with_repaint: Does have REPAINT
- trains on turbines 1-5
- tests on turbine 6
- uses 5 input feature (mean wind speed, mean sine of wind direction, mean cosine of wind direction, mean transformer temperature, and mean nacelle temperature) to predict mean power output
- all features are normalised
- full details of config in config.py
