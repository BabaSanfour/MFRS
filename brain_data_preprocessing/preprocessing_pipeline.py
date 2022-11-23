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
import mne
import numpy as np
import os.path as op
from warnings import warn

import mne
import pickle
from mne import Epochs
from mne.preprocessing import ICA
from mne.parallel import parallel_func
from mne.io import read_raw_fif, concatenate_raws

from library.config_bids import study_path, meg_dir, N_JOBS, l_freq, random_state, tmin, tmax, reject_tmax, map_subjects, cal, ctc


def run_maxwell_filter(subject_id):
    subject = "sub-%02d" % subject_id
    print("processing subject: %s" % subject)
    sss_fname_out = op.join(meg_dir, subject, 'run_%02d_filt_tsss_%d_raw.fif')

    raw_fname_in = op.join(study_path, 'ds000117', subject, 'ses-meg/meg',
                           'sub-%02d_ses-meg_task-facerecognition_run-%02d_meg.fif')

    sss_fname_in = op.join(study_path, 'ds000117/derivatives/meg_derivatives', subject, 'ses-meg/meg',
                           'sub-%02d_ses-meg_task-facerecognition_run-%02d_proc-sss_meg.fif')
    st_duration=10
    # To match the processing, transform to the head position of the 4th run
    info = mne.io.read_info(sss_fname_in % (subject_id, 4))
    destination = info['dev_head_t']
    # Get the origin they used
    origin = info['proc_history'][0]['max_info']['sss_info']['origin']

    for run in range(1, 7):
        raw_out = sss_fname_out % (run, st_duration)
        if op.isfile(raw_out):
            continue
        raw_in = raw_fname_in % (subject_id, run)
        try:
            raw = mne.io.read_raw_fif(raw_in)
        except AttributeError:
            # Some files on openfmri are corrupted and cannot be read.
            warn('Could not read file %s. '
                 'Skipping run %s from subject %s.' % (raw_in, run, subject))
            continue
        print('  Run %s' % (run,))

        raw.fix_mag_coil_types()

        # Read bad channels from the MaxFilter log.
        bads =[]
        with open( op.join(study_path, 'ds000117/derivatives/meg_derivatives', subject, 'ses-meg/meg',
               'sub-%02d_ses-meg_task-facerecognition_run-%02d_proc-sss_log.txt' %(subject_id, run))) as fid:
            for line in fid:
                if line.startswith('Static bad channels'):
                    chs = line.split(':')[-1].split()
                    bads = ['MEG%04d' % int(ch) for ch in chs]
                    break

        raw.info['bads'] += bads

        # Get the head positions from the existing SSS file.
        # Usually this can be done by saving to a text file with MaxFilter
        # instead, and reading with :func:`mne.chpi.read_head_pos`.
        raw_sss = mne.io.read_raw_fif(sss_fname_in % (subject_id, run), verbose='error')
        chpi_picks = mne.pick_types(raw_sss.info, meg=False, chpi=True)
        assert len(chpi_picks) == 9
        head_pos, t = raw_sss[chpi_picks]
        # Add first_samp.
        t = t + raw_sss.first_samp / raw_sss.info['sfreq']
        # The head positions in the FIF file are all zero for invalid positions
        # so let's remove them, and then concatenate our times.
        mask = (head_pos != 0).any(axis=0)
        head_pos = np.concatenate((t[np.newaxis], head_pos)).T[mask]
        # In this dataset due to old MaxFilter (2.2.10), data are uniformly
        # sampled at 1 Hz, so we save some processing time in
        # maxwell_filter by downsampling.
        skip = int(round(raw_sss.info['sfreq']))
        head_pos = head_pos[::skip]
        del raw_sss

        print('    st_duration=%d' % (st_duration,))
        raw_sss = mne.preprocessing.maxwell_filter(
            raw, calibration=cal, cross_talk=ctc, st_duration=st_duration,
            origin=origin, destination=destination, head_pos=head_pos)

        #The data are bandpass filtered (1 - 40 Hz) using linear-phase fir filter
        #with delay compensation
        # Here we only low-pass MEG (assuming MaxFilter has high-passed
        # the data already)
        picks_meg = mne.pick_types(raw.info, meg=True, exclude=())
        raw_sss.filter(
            None, 40, l_trans_bandwidth='auto', h_trans_bandwidth='auto',
            filter_length='auto', phase='zero', fir_window='hamming',
            fir_design='firwin', n_jobs=N_JOBS, picks=picks_meg)
        raw_sss.save(raw_out, overwrite=True)

if __name__ == "__main__":

    subjects_ids=[i for i in range(1,11)]
    # preprocessing for the 17 subjects
    for subject_id in subjects_ids:
        run_maxwell_filter(subject_id)
