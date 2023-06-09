# FAIR: <ins>F</ins>air Collaborative <ins>A</ins>ctive Learning with <ins>I</ins>ndividual <ins>R</ins>ationality for Scientific Discovery [AISTATS 2023]
Official implementation of our AISTATS 2023 paper ["FAIR: Fair Collaborative Active Learning with Individual Rationality for Scientific Discovery"](https://xinyi-xu.com/research.html) (__29%__ acceptance rate).

## Requirements
1. Linux machine (experiments were run on Ubuntu 18.04.5 LTS and Ubuntu 20.04.2 LTS)
2. Anaconda (alternatively, you may install the packages in `environment.yml` manually)

## Setup
1. Run the following command to install the required Python packages into a new environment named CAL using Anaconda.
```shell
conda env create -f environment.yml
```

#### Notes on the differential equation experiments:

- The `npde_master` directory provides the necessary functions for handling differential equation-based data generation/computation.
- To  run the differential equations experiments requires setting up and activating a tensorflow-based environment, which is found in `environment-TF.yml`.

## Running experiments

In the main directory,
1. Change current environment to the CAL environment.
```shell
conda activate CAL
```
2. Run the desired experiment. Files with names `exp_*.py` are scripts for different experiments. 
- `python synthetic_run_1D.py` executes the experiments on the 1-dimensional synthetic data
- `python de-ode_run.py` executes the experiments on the differential equation data where the true function is an ordinary differential equation. Note that to run this experiments requires the tensorflow-based environment.

#### Jupyter notebooks
Several notebooks for experiments with different datasets and analyzing results are provided for interactively developling and testing the methods. Visualization of the data and results is also provided. You can execute the notebooks under the correct conda environment: `environment.yml` for non-differential equation experiments and `environment-TF.yml` for differential equation experiments.

## License
This code is released under the MIT License.

## Citing our paper
If you find our paper relevant or use our code in your research, please consider citing our paper:
```
@InProceedings{Xu2023,
  title={FAIR: Fair Collaborative Active Learning with Individual Rationality for Scientific Discovery},
  author={Xinyi Xu and Zhaoxuan Wu and Chuan Sheng Foo and Bryan Kian Hsiang Low},
  booktitle={Proceedings of the 26th International Conference on Artificial Intelligence and Statistics (AISTATS-23)},
  year={2023}
}
```

