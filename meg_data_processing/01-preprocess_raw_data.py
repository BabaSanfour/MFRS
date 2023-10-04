import os
import mne
import numpy as np
import logging
from mne.parallel import parallel_func

import sys
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
from utils.config import study_path, meg_dir, cal, ctc

# Set up logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

def run_maxwell_filter(subject_id: int) -> None:
    """
    Apply Maxwell filtering to MEG data for a specific subject.

    Args:
        subject_id (int): Subject ID.

    Returns:
        None
    """
    subject = f"sub-{subject_id:02d}"
    logger.info(f"Processing subject: {subject}")

    st_duration=10
    #To match analysis in previous studies: transform to the head position of the 4th run
    info = mne.io.read_info(os.path.join(study_path, "ds000117/derivatives/meg_derivatives", subject, 
                                'ses-meg/meg', f'{subject}_ses-meg_task-facerecognition_run-04_proc-sss_meg.fif'), verbose=False)
    destination = info['dev_head_t']
    origin = info['proc_history'][0]['max_info']['sss_info']['origin']

    for run in range(1, 7):
        raw_out = os.path.join(meg_dir, subject, f'run_{run:02d}_filt_raw.fif')
        raw_in = os.path.join(study_path, 'ds000117', subject, 
                            'ses-meg/meg', f'sub-{subject_id:02d}_ses-meg_task-facerecognition_run-{run:02d}_meg.fif')
        raw = mne.io.read_raw_fif(raw_in, verbose=False)
        raw.set_channel_types({
                'EEG061': 'eog',
                'EEG062': 'eog',
                'EEG063': 'ecg',
                'EEG064': 'misc'
            })  # EEG064 free floating el.
        raw.rename_channels({
            'EEG061': 'EOG061',
            'EEG062': 'EOG062',
            'EEG063': 'ECG063'
        })
        raw.fix_mag_coil_types()

        # Read bad channels from the MaxFilter log.
        bads =[]
        with open( os.path.join(study_path, 'ds000117/derivatives/meg_derivatives', subject, 
                            'ses-meg/meg', f'sub-{subject_id:02d}_ses-meg_task-facerecognition_run-{run:02d}_proc-sss_log.txt')) as fid:
            for line in fid:
                if line.startswith('Static bad channels'):
                    chs = line.split(':')[-1].split()
                    bads = [f'MEG{int(ch) :04d}' for ch in chs]
                    break
        raw.info['bads'] += bads

        # Get the head positions from the existing SSS file.
        # Usually this can be done by saving to a text file with MaxFilter
        # instead, and reading with :func:`mne.chpi.read_head_pos`.
        raw_sss = mne.io.read_raw_fif(os.path.join(study_path, 'ds000117/derivatives/meg_derivatives', subject, 
                            'ses-meg/meg', f'sub-{subject_id:02d}_ses-meg_task-facerecognition_run-{run:02d}_proc-sss_meg.fif'), verbose='error')
        chpi_picks = mne.pick_types(raw_sss.info, meg=False, chpi=True)
        assert len(chpi_picks) == 9
        head_pos, t = raw_sss[chpi_picks]
        # Add first_samp.
        t = t + raw_sss.first_samp / raw_sss.info['sfreq']
        # The head positions in the FIF file are all zero for invalid positions so let's remove them, and then concatenate our times.
        mask = (head_pos != 0).any(axis=0)
        head_pos = np.concatenate((t[np.newaxis], head_pos)).T[mask]
        # In this dataset due to old MaxFilter (2.2.10), data are uniformly sampled at 1 Hz, so we save some processing time in maxwell_filter by downsampling.
        skip = int(round(raw_sss.info['sfreq']))
        head_pos = head_pos[::skip]
        del raw_sss

        raw_sss = mne.preprocessing.maxwell_filter(
            raw, calibration=cal, cross_talk=ctc, st_duration=st_duration,
            origin=origin, destination=destination, head_pos=head_pos)

        #The data are bandpass filtered (1 - 90 Hz) using linear-phase fir filter with delay compensation
        # Here we only low-pass MEG (assuming MaxFilter has high-passed the data already)
        picks_meg = mne.pick_types(raw.info, meg=True, exclude=())
        raw_sss.filter(
            None, 90, l_trans_bandwidth='auto', h_trans_bandwidth='auto',
            filter_length='auto', phase='zero', fir_window='hamming',
            fir_design='firwin', n_jobs=-1, picks=picks_meg)
        picks_eog = mne.pick_types(raw.info, meg=False, eog=True)
        raw_sss.filter(
            1, None, picks=picks_eog, l_trans_bandwidth='auto',
            filter_length='auto', phase='zero', fir_window='hann',
            fir_design='firwin')
        raw_sss.save(raw_out, overwrite=True)

def run_ica(subject_id: int) -> None:
    """
    Apply ICA to MEG data for a specific subject.
    Args:
        subject_id (int): Subject ID.

    Returns:
        None
    """
    subject = f"sub-{subject_id:02d}"
    logger.info(f"Processing subject: {subject}")
    ica_name = os.path.join(meg_dir, subject, 'run_concat-ica.fif')
    raws = list()
    for run in range(1, 7):
        raw_in = os.path.join(meg_dir, subject, f'run_{run:02d}_filt_raw.fif')
        raws.append(mne.io.read_raw_fif(raw_in))
    n_components=0.999
    raw = mne.concatenate_raws(raws)
    ica = mne.preprocessing.ICA(method='fastica', n_components=n_components)
    picks = mne.pick_types(raw.info, meg=True, eeg=False, eog=False,
                        stim=False, exclude='bads')
    ica.fit(raw, picks=picks, reject=dict(grad=4000e-13, mag=4e-12),
            decim=11)
    logger.info(f" Fit {ica.n_components_} components (explaining at least {100 * n_components:.1f}% of the variance)")
    logger.info("Saving ICA solution...")
    ica.save(ica_name, overwrite=True)


if __name__ == "__main__":

    logger.info("Running Maxwell filter for all subjects...")

    parallel, run_func, _ = parallel_func(run_maxwell_filter, n_jobs=-1)
    parallel(run_func(subject_id) for subject_id in list(range(1, 17)))

    logger.info("Maxwell filtering completed for all subjects.")

    logger.info("Running ICA for all subjects...")

    parallel, run_func, _ = parallel_func(run_ica, n_jobs=-1)
    parallel(run_func(subject_id) for subject_id in list(range(1, 17)))

    logger.info("ICA completed for all subjects.")