This repository is for the SBI-SGM-FC project, the paper is ``Bayesian Inference of Frequency Specific Functional Connectivity in MEG Imaging Using a Spectral Graph Model''

## Environment

The code is based on Python, to run the code, you can create a conda environment with the following command:

```bash
conda env create -f environment.yml --name SGM_FC
```

**Note that you should run the code under `root` directory**



## Files


### `python_scripts/`

The folder contains the python scripts for running the experiments in the paper
- `M1_RUN_ANN.py` is the main script for running the experiments for SGM for annealing fitting 
- `M2_RUN_NMM.py` is the main script for running the experiments with NMM model
- `M3_RUN_BW_WANN.py` is the main script for running the experiments with SBI model based on the prior from the annealing fitting (so you should run `M1_RUN_ANN.py` first)
- `M4_RUN_ANN_ALLBD.py`: is the main script for running the experiments with SGM for annealing fitting with all the bands (SGM-shared)
- `M5_RUN_ALLBD_WANN.py`: is the main script for running the experiments with SBI model based on the prior from the annealing fitting with all the bands (SBI-SGM-shared) (so you should run `M4_RUN_ANN_ALLBD.py` first)

They will procdure the results for Figures 4, 5 and S3. (Of course, you should run some jupyter notebook to get the exact figures, which will be provided in the following)



### `notebooks`

The folder contains the jupyter notebooks for running simple analysis and generating the figures in the paper
Note that you may need powerpoint to combine them).
- `1_MEG_FC_emp.ipynb`: Estimate the FCs we used for the experiments

- `2_ANA_BW.ipynb`: Extract the metrics for SGM-SBI
    - You should run python scripts in `python_scripts/` first to get the results

- `3_ANA_BW_ALLBD.ipynb`: Extract the metrics for SGM-SBI-shared
    - You should run python scripts in `python_scripts/` first to get the results

- `4_All_in_one_ana.ipynb`: Generate the figures in the paper
    - You should run python scripts in `python_scripts/` first to get the results
    - You should run `2_ANA_BW.ipynb` and `3_ANA_BW_ALLBD.ipynb` first to get the metrics
    - This file only include code to run `Eigen` method 
    - It can generate the figures (Figs 2-5)
- `5_ind_ana.ipynb`: generate the individual figures
    - it can generate the figures (Figs S1-S2)

- `6_evaluate_on_ROI.ipynb`: generate Figure 6

- `7_SGM_FC_explore.ipynb`: Generate Figure 1

- `8_XX`: Generate Figure S4

### `data`

It contains the data used in the paper