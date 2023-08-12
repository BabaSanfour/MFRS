# MEG Data Processing

This repository contains scripts for processing MEG (Magnetoencephalography) data from a multi-subject, multi-modal human neuroimaging dataset used in this project. The data is available on the open-source platform OpenNeuro. The dataset comprises simultaneous MEG/EEG recordings from 19 healthy participants performing a visual recognition task. Subjects were presented with images of famous, unfamiliar, and scrambled faces. Each subject participated in 6 runs, each lasting 7.5 minutes. The data were acquired using an Elekta Neuromag Vectorview 306 system. Additionally, a 70-channel Easycap EEG system was used for simultaneous EEG recording. Anatomical data and fMRI recordings are also included.

## Dataset Information

- **Dataset:** Multi-subject, multi-modal human neuroimaging dataset
- **Accession Number:** ds000117
- **Version:** 1.0.5
- **Created:** 2021-09-27
- **OpenNeuro:** [Access Dataset](https://openneuro.org/datasets/ds000117/versions/1.0.5)
- **Download Instructions:** Ensure to download the data into your working directory; instructions are available on the OpenNeuro website.

This data has been previously analyzed in several studies, particularly by Jas et al. and Meunier, Pascarella, et al. The scripts utilized in this project draw inspiration from these two analyses:

## Jas et al.

- **Paper:** [NeuroPycon: An open-source python toolbox for fast multi-modal and reproducible brain connectivity pipelines](https://doi.org/10.1016/j.neuroimage.2020.117020)
- **MNE Biomag Demo:** [Demo Link](https://mne.tools/mne-biomag-group-demo/index.html)
- **GitHub Repository:** [GitHub Link](https://github.com/mne-tools/mne-biomag-group-demo)

## Meunier, Pascarella, et al.

- **Paper:** [A reproducible MEG/EEG group study with the MNE software](https://doi.org/10.3389/fnins.2018.00530)
- **NeuroPycon Demo:** [Demo Link](https://neuropycon.github.io/ephypype/#)
- **GitHub Repository:** [GitHub Link](https://github.com/neuropycon/neuropycon_demo)

## Contents

1. [Preprocessing Raw Data](01-preprocess_raw_data.py): Script for preprocessing raw MEG data, including noise reduction, filtering, and data transformation.

   This script performs the following steps:
   - Removal of environmental noise using noise covariance estimation.
   - Application of Maxwell filtering to reduce environmental interference.
   - Low-pass filtering to remove high-frequency noise above 90 Hz.
   - Transformation to the head position of the 4th run for consistent alignment.


2. [Create Epochs](02-create_epochs.py): Script for segmenting the preprocessed data into epochs, making it suitable for further analysis.

   This script performs the following steps:
   - Reads and matches events from MEG data with corresponding trigger values from CSV files.
   - Filters and updates trigger values based on specific conditions and mappings.
   - Aggregates disregarded events from individual subject JSON files into a mega JSON file.
   - Creates epochs from preprocessed MEG data using the specified time window.

3. [Anatomical Data Processing](03-process_anatomical_data.py): Script for processing anatomical MRI data and creating BEM models and source spaces.

   This script performs the following steps:
   - Reconstruction of anatomical MRI data using the FreeSurfer `recon-all` pipeline.
   - Creation of BEM surfaces including inner skull surfaces for accurate source modeling.
   - Generation of a BEM model for EEG/MEG source estimation using the linear collocation approach.
   - Creation of a surface source space for EEG/MEG source localization.
   
   Detailed parameter settings are available within the script comments.

   **Dependencies:**
   - FreeSurfer: Ensure that FreeSurfer is installed and properly configured.

   **Usage:**
   Run the script using the command: `python 03-process_anatomical_data.py --subject <subject_id>`

   **Note:**
   The script assumes that anatomical MRI data is available in the specified directory structure. The processed BEM models and source spaces will be stored in the subjects' directory for further use in EEG/MEG source estimation.

   **Important:**
   The execution of this script may take several hours, as it involves complex anatomical reconstruction and modeling processes. Make sure to allocate sufficient computing resources and monitor the progress.

   *For more details and usage instructions, refer to the script comments.*

4. [Calculate Inverse Source Time Courses](04-calculate_inverse_stc.py): Script for calculating inverse source time courses using source localization techniques.

   This script performs the following steps:
   - Reads the necessary data files and setups, including transformation matrices, source spaces, epochs, covariance matrices, and forward solutions.
   - Calculates the inverse operator using minimum-norm estimation with specified parameters.
   - Computes source estimates for each epoch using the calculated inverse operator.
   - Morphs the source estimates from the subject's brain space to the common fsaverage brain space.
   - Averages source estimates within defined regions of interest (ROIs) to obtain ROI-specific time courses.
   
   **Usage:**
   Run the script using the command: `python 04-calculate_inverse_stc.py --subject <subject_id> --overwrite`


## Usage

- The scripts are designed to be executed in sequence, with each script building upon the outputs of the previous ones.
- Ensure that you provide the correct `<subject_id>` when running the scripts.
- Some scripts support the `--overwrite` flag to force recomputation if desired.
- Depending on the size and complexity of your data, script execution times may vary. Allocate sufficient computing resources and monitor progress.


## Note

This repository serves as an example for organizing MEG data processing scripts. Make sure to adapt the scripts and settings to your specific data and analysis needs.

Feel free to contribute improvements or suggest changes if you find this repository helpful.