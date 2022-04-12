# NetBio
## Description
Source codes for generating results of "Network-based machine learning approach to predict immunotherapy response in cancer patients". 


## Requirements
- python (3.6.12)
- pandas (1.1.15)
- numpy (1.19.2)
- scipy (1.5.4)
- matplotlib (3.3.3)
- sklearn (0.24.2)
- lifelines (0.25.7)
- networkx (2.5)
- statsmodels (0.12.2)
- pytorch (1.7.1+cu110)

## Installation
All packages can be installed via pip (https://pypi.org/project/pip/). Generally, a couple of minutes is needed for installing each package.

**e.g.** pip install sklearn


## NetBio predictions
- Code for reproducing leave-one-out cross-validation (LOOCV) and Monte-Carlo cross-validation is provided under the './code/1_within_study_prediction' folder.
- Expected results are provided under './result/1_within_study_prediction' folder.
- The expected run time is under 15 minutes for running LOOCV or Monte-Carlo cross-validation.




```
To make LOOCV predictions, run 'compute_LOOCV.py' under the './code/1_within_study_prediction' folder.

Use the following command line to run 'compute_LOOCV.py' in linux.

  $ python3 compute_LOOCV.py

LOOCV results are generated under the './result/1_within_study_prediction' folder.
```

