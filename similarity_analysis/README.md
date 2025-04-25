# Similarity Analysis

This folder contains source code for performing Similarity Analysis on brain data and model activations. The analysis involves computing Representational Dissimilarity Matrices (RDMs), performing Representational Similarity Analyses (RSAs), estimating noise ceilings, and running experiments. 

- RDMs quantify the dissimilarity or similarity between the neural representations of different stimuli or conditions. Each element of an RDM represents the dissimilarity between the neural response patterns corresponding to a pair of stimuli.
- RSA involves comparing the structure of neural representations across different conditions or models. By correlating RDMs, RSA quantifies the similarity between the underlying representations, providing insights into the organization of information in the brain.
- NC is a crucial concept in neuroscience that helps assess the upper limit of similarity between neural representations. It accounts for the inherent variability or noise present in the data. When conducting similarity analyses, it's essential to consider that not all variability can be attributed to meaningful differences between conditions or models.


## Source Code

### `src` Subfolder

1. [Representational Dissimilarity Matrices (RDMs)](src/rdm.py):

    - This module provides functions to compute RDMs for MEG data and layers' activations. It offers flexibility to compute RDMs in sensor or source space. Two methods are available for MEG data:
        * Compute RDMs for a specified time segment (e.g., 500ms). Optionally, apply a time window to compute RDMs for different time segments (e.g., 500ms with a time window of 50ms).
        * Compute an RDM for each time point by collapsing the sensors or voxels of a region. Parallel processing is available to expedite the computation.

2. [Representational Similarity Analyses (RSAs)](src/rsa.py):
    - This module calculates similarity scores between brain RDMs and layers' RDMs.

3. [Noise Ceilings (NCs)](src/noise_ceiling.py):
    - This module computes noise ceilings using bootstrapping or leave-one-out methods.
        * Bootstrapping Method:
            - The similarity analysis is performed on each resampled dataset to create a distribution of similarity scores.
            - The mean and standard deviation of this distribution represent the noise ceiling.
        * Leave-One-Out Method:
            - This method systematically leaves out one subject at a time and computes the similarity analysis.
            - The average similarity across all iterations serves as an estimate of the noise ceiling.

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

## Usage
This repository serves as an example for computing RDMs or MEG data and can be used for any time series type of data, or fMRI. Make sure to adapt the scripts and settings to your specific data and analysis needs.
