"""
The preprocessing pipeline for the brain data used in this project.

`A multi-subject, multi-modal human neuroimaging dataset <https://arxiv.org/abs/1912.10079>`_.
The data is availabe on the open source platform OpenNeuro.
`OpenNeuro Accession Number: ds000117, version: 1.0.5, created: 2021-09-27  <https://openneuro.org/datasets/ds000117/versions/1.0.5>`_.
! Make sure to download the data in your working directory, instructions can be found on the OpenNeuro website.

This is a copy with some modifications (add notes from paper and adapt to the needs in our study) of the original analysis produced by Jas et al.
`A reproducible MEG/EEG group study with the MNE software: Recommendations, quality assessments, and good practices <https://doi.org/10.3389/fnins.2018.00530>`_.
Original scripts are available on MNE documentation and Github.
`MNE Biomag Demo: <https://mne.tools/mne-biomag-group-demo/index.html>`_.
`Github link: https://github.com/mne-tools/mne-biomag-group-demo>`_.
! In the official analysis another version of data was used. We used the lastest available one (organized with respecting the BIDS format)
"""

import os
import sys
import os.path as op
from warnings import warn

import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from autoreject import get_rejection_threshold

import mne
from mne import Epochs
from mne.parallel import parallel_func
from mne.preprocessing import find_bad_channels_maxwell, ICA, create_ecg_epochs, create_eog_epochs, read_ica

from library.config_bids import study_path, meg_dir, N_JOBS, cal, ctc, l_freq, annot_kwargs, map_subjects, random_state, set_matplotlib_defaults

def get_input_files(n_subjects, n_runs, data_accession_number: str = "ds000117"):
    subject = "sub-%02d" % subject_id
    in_path = op.join(study_path, data_accession_number, subject, 'ses-meg/meg')
