# School Examination Experiments

This directory contains the MT-NDP architecture and configurations described in the accompanying paper for the school examination dataset. 

The School Examination dataset (`Exam`) is publicly available through
the R package [`mlmRev`](https://cran.r-project.org/package=mlmRev).
The dataset can be downloaded directly from the
[CRAN GitHub mirror](https://raw.githubusercontent.com/cran/mlmRev/master/data/Exam.rda).

# Files

* model.py: MT-NDP architecture using a task encoder and a bidimensional attention-based diffusion model.
* config.py: Experimental configurations for the main school experiment and leave-one-school-out evaluation.
