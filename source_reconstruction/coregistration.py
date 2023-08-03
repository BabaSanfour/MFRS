import mne
import os
from utils.config import subjects_dir
import numpy as np




subject = "sub-01"
fname_raw = os.path.join("/Users/hamzaabdelhedi/Projects/data/MFRS_data/ds000117/sub-01/ses-meg/meg", f"{subject}_ses-meg_task-facerecognition_run-01_meg.fif")
info = mne.io.read_info(fname_raw)
fiducials = "estimated"  # get fiducials from fsaverage
coreg = mne.coreg.Coregistration(info, subject, subjects_dir, fiducials=fiducials)
coreg.fit_fiducials(verbose=True)
coreg.fit_icp(n_iterations=6, nasion_weight=2.0, verbose=True)

coreg.omit_head_shape_points(distance=5.0 / 1000)  # distance is in meters
coreg.fit_icp(n_iterations=20, nasion_weight=10.0, verbose=True)
mne.write_trans(f'/Users/hamzaabdelhedi/Projects/data/MFRS_data/subjects/{subject}/{subject}-trans.fif', coreg.trans)