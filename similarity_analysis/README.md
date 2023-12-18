# Similarity Analysis

This folder contains source code for performing Similarity Analysis on brain data and model activations. The analysis involves computing Representational Dissimilarity Matrices (RDMs), performing Representational Similarity Analyses (RSAs), estimating noise ceilings, and running experiments.

## Source Code

### `src` Subfolder

1. [Representational Dissimilarity Matrices (RDMs)](/src/rdm.py):

    - This module provides functions to compute RDMs for MEG data and layers' activations. It offers flexibility to compute RDMs in sensor or source space. Two methods are available for MEG data:
        * Compute RDMs for a specified time segment (e.g., 500ms). Optionally, apply a time window to compute RDMs for different time segments (e.g., 500ms with a time window of 50ms).
        * Compute an RDM for each time point by collapsing the sensors or voxels of a region. Parallel processing is available to expedite the computation.

2. [Representational Similarity Analyses (RSAs)](/src/rsa.py):
    - This module calculates similarity scores between brain RDMs and layers' RDMs.

3. [Noise Ceilings (NCs)](/src/noise_ceiling.py):
    - This module computes noise ceilings using bootstrapping or leave-one-out methods.

### Main Folder

1. [Computing RDMs](compute_rdms.py):
    - Script for executing the computation of RDMs.

2. [Computing RSAs](compute_rsa.py):
    - Script for executing the computation of RSAs.

3. [Computing Noise Ceiling](compute_noise_ceiling.py):
    - Script for executing the computation of noise ceilings.

4. [Plotting functions](plot_functions.py):
    - Functions for generating plots related to Similarity Analysis.

5. [Plotting utils](plot_utils.py):
    - Utility functions for plotting.

#### Notebooks
6. [Plots for OHBM 2023](plots_0_OHBM.ipynb):
7. [Plots for Neuro-AI workshop](plots_1_neuro_ai_workshop.ipynb):
8. [Plots for Master update 18/12/2023](plots_2_master_update_18_12_23.ipynb):
    - These notebooks contain examples and visualizations related to the conducted analyses.

## Usage
This repository serves as an example for computing RDMs or MEG data and can be used for any time series type of data, or fMRI. Make sure to adapt the scripts and settings to your specific data and analysis needs.
