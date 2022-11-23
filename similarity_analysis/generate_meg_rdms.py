"""
Compute MEG RDMs
"""

import mne
import numpy as np
import os.path as op
from neurora.rdm_cal import eegRDM
import sys
sys.path.append('/home/hamza97/MFRS/brain_data_preprocessing')
from library.config_bids import meg_dir

def transform_data(sub_id, n_cons, n_chls, n_time_points, name="FamUnfam", tsss=10):
    """
    This function read epoched MEG data and saves it a matrix with the shape [n_sub, n_conditions, n_channels, n_time_points]
    """
    megdata = np.zeros([len(sub_id), n_cons, n_chls, n_time_points], dtype=np.float32)
    subindex = 0
    for subject_id in sub_id:
        subject = "sub-%02d"%subject_id
        data_path = op.join(meg_dir, subject)
        file = op.join(data_path, '%s-tsss_%d_%s-epo.fif' % (subject, tsss, name))
        epochs = mne.read_epochs(file)
        subdata = np.zeros([n_cons, n_chls, n_time_points], dtype=np.float32)

        for j  in range(n_cons):
            epoch = epochs[list_names[j]]
            subdata[j] = epoch.get_data()
        megdata[subindex] = subdata
        subindex = subindex + 1
    return megdata

def compute_RDMs(megdata, n_cons, n_subj, n_trials, n_chls, n_time_points, name="FamUnfam", sub_opt=1, chl_opt=1):
    """
    This function computes the RDMs for MEG data and saves them in a numpy file.
    The shape depends on the input parameters

    Parameters
    ----------
    megdata : array
        The MEG data.
        The shape of MEG data must be [n_cons, n_subs, n_trials, n_chls, n_ts].
        n_cons, n_subs, n_trials, n_chls & n_ts represent the number of conidtions, the number of subjects, the number
        of trials, the number of channels & the number of time-points, respectively.
    sub_opt: int 0 or 1. Default is 1.
        Return the subject-result or average-result.
        If sub_opt=0, return the average result.
        If sub_opt=1, return the results of each subject.
    chl_opt : int 0 or 1. Default is 0.
        Calculate the RDM for each channel or not.
        If chl_opt=0, calculate the RDM based on all channels'data.
        If chl_opt=1, calculate the RDMs based on each channel's data respectively.

    Returns
    -------
    RDM(s) : array
        The MEG RDM.
    If sub_opt=0 & chl_opt=0, return only one RDM.
        The shape is [n_cons, n_cons].
    If sub_opt=0 & chl_opt=1, return n_chls RDM.
        The shape is [n_chls, n_cons, n_cons].
    If sub_opt=1 & chl_opt=0, return n_subs RDM.
        The shape is [n_subs, n_cons, n_cons].
    If sub_opt=1 & chl_opt=1, return n_subs*n_chls RDM.
        The shape is [n_subs, n_chls, n_cons, n_cons].
    """
    megdata = np.transpose(megdata, (1, 0, 2, 3))
    megdata = np.reshape(megdata, [n_cons, n_subj, n_trials, n_chls, n_time_points])
    # Calculate the RDM based on the data during
    rdm = eegRDM(megdata, sub_opt, chl_opt)

    out_file = op.join(meg_dir, 'RDMs_%s_%d-subject_%d-sub_opt%d-chl_opt.npy' % (name, n_subj, sub_opt, chl_opt))
    np.save(out_file, rdm)
    print('File saved successfully')


if __name__ == '__main__':

    #  list of pictures names (stimuli: used as trigger)
    # list_names = []
    list_names = [str(i) for i in range(1,301)]
    # for i in range(301,451):
    #     list_names.append(str(i))
    sub_id = [i for i in range(1,17)]
    megdata = transform_data(sub_id, 300, 306, 881, "FamUnfam")
    compute_RDMs(megdata, 300, len(sub_id), 1, 306, 881, "FamUnfam")
