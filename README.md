# learning-occupant-network
[![DOI](https://zenodo.org/badge/295826352.svg)](https://zenodo.org/badge/latestdoi/295826352)

Tool for inferring occupant network structure from time-series sensor data. The novel tool is located in `interaction_model.py`. This repository, largely structured using Jupyter notebooks, contains the code for all analysis in the following paper:

Andrew Sonta and Rishee K. Jain. (2020). "Learning socio-organizational network structure in buildings with ambient sensing data." _Data-Centric Engineering_. DOI:10.1017/dce.2020.9

Note: Some notebooks use `R` code with IPython magic. We recommend using a virtual environment with the packages in `requirements.txt`.

## Notebook 1: Inferring activity states from clean data
The first part of the workflow is clustering time-series sensor data to "activity states," which are abstractions of the sensor data that can be used to describe building occupant activities.

`state_classification.py` is used on the following form: `python src/state_classification.py data/clean-example1.csv data/classified-example1.csv` where clean-example1.csv is the cleaned data file and classified-example1.csv is a new file containing the time-series activity states.

## Notebook 2: Network inference
Once the occupant activity states are defined, we can infer the occupant network. Three methods are used:
1. Graphical Lasso
2. Influence Model
3. Interaction Model (custom algorithm)

The Interaction Model makes use of the code in the `interaction_model.py` file.

## Notebook 3: Ground truth data loading
To test the performance of the methods used to infer the network, we compare the inferred networks to ground truth data collected through a survey. This notebook loads the ground truth networks from the relevant data files.

## Notebook 4: Network tests
Having inferred networks in notebook 2 and loaded the ground truth networks in notebook 3, we can now test the similarities between the inferred and ground truth networks. This notebook involves the use of the Pearson product-moment correlation as a similarity measure and the Quadratic Assignment Procedure as a test of correlation significance.

## Notebook 5: Auxiliary
This notebook contains additional tests and data exploration.
