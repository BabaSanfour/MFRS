# Data Folder Breakdown

This document provides a breakdown of the project's data folders, including the last update timestamps and the scripts used for those updates.

## Directory Structure

```
MFRS_data/
│
├── activations/                             # CNN-extracted activations
├── ERPs/                                    # Maximum values of explained variance for raw and various bands for OFA and FFA across seven models, training tasks, and different stimuli.
├── hdf5/                                    # HDF5 files containing training, fine-tuning data, and stimuli used to generate RDMs (MEG experiment stimuli).
├── images/                                  # TAR and ZIP files with training, fine-tuning data, and stimuli used to generate RDMs (MEG experiment stimuli).
├── MEG/                                     # 
│   ├── erps/                                # Event-related potentials (ERPs) of the raw signal and different bands for various regions (Numpy files in shape N_subj, N_stim, N_time_points).
│   └── subXX/                               # 16 folders, each containing preprocessing and source reconstruction results for individual subjects.
│        ├── hilbert/                        # Hilbert-transformed files for different frequency bands.
│        ├── morph/                          # Morphed source-reconstructed data into fsaverage.
│        ├── src/                            # Source-reconstructed data.
│        └── mega_events_disregarded.json    # JSON file containing events to be disregarded.
├── noise_ceiling/                           # Noise ceilings for two different types of analysis.
│   ├── avg_nc/                              # Average noise ceiling data.
│   └── raw_nc/                              # Raw noise ceiling data.
├── rdms/                                    # Representational dissimilarity matrices (RDMs) for two different types of analysis, for subjects and models.
│   ├── avg_rdms_sbjct_avg/                  # Average RDMs for subjects.
│   ├── networks_rdms/                       # Network-specific RDMs.
│   ├── raw_rdms_sbjct_avg/                  # Raw average RDMs for subjects.
│   └── sub-XX/                              # Subject-specific RDMs.
│       ├── avg_nc/                          # RDMs for average analysis for subject XX.
│       └── raw_nc/                          # RDMs for raw analysis for subject XX.
├── sim_scores/                              # Similarity scores for two different types of analysis.
│   ├── avg_sim_scores/                      # Average similarity scores.
│   └── raw_sim_scores/                      # Raw similarity scores.
├── subjects/                                # Subject-specific data and resources.
│   ├── fsaverage/                           # FSaverage data.
│   ├── morph-maps/                          # Morphed maps for subjects.
│   ├── sub-XX/                              # Individual subject data.
│   └── subjects_MRI.tar.gz                  # Compressed file containing subject MRI data.
└── old_files_mac.tar                        # Archived similarity scores generated using the average method.
```