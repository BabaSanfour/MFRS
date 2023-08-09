import os
import sys
import mne
import json
import numpy as np
from pandas import read_csv
from mne.parallel import parallel_func

sys.path.append('../../MFRS/')
from utils.config import study_path, meg_dir, reject_tmax, map_subjects, N_JOBS, conditions_mapping
from utils.arg_parser import get_similarity_parser

def run_events(subject_id: int) -> None:
    """
    Process events for a subject's MEG runs.
    
    Args:
        subject_id (int): Subject ID.
    
    Returns:
        None
    """
    subject = f"sub-{subject_id:02d}"
    print(f"Processing subject: {subject}")
    
    existing_order1 = []
    existing_order2 = []
    
    for run in range(1, 7):
        fname_events = os.path.join(meg_dir, subject, f'run_{run:02d}-eve.fif')
        run_fname = os.path.join(study_path, 'ds000117', subject, 'ses-meg/meg', f'sub-{subject_id:02d}_ses-meg_task-facerecognition_run-{run:02d}_meg.fif')
        raw = mne.io.read_raw_fif(run_fname, verbose=False)
        
        # Find events
        mask = 4096 + 256
        events = mne.find_events(raw, stim_channel='STI101', consecutive='increasing', mask=mask,
                                 mask_type='not_and', min_duration=0.003)
        
        # Read CSV for events
        csv_path = os.path.join(study_path, 'ds000117', subject, 'ses-meg/meg', f'sub-{subject_id:02d}_ses-meg_task-facerecognition_run-{run:02d}_events.tsv')
        event_csv = read_csv(csv_path, sep='\t')
        
        # Match events and CSV rows
        events = [event for event in events if event[2] in event_csv['trigger'].values]
        
        # Add order of trial and change event name
        events[:, 1] = [1 if row['trigger'] in [5, 13, 17] else 2 for _, row in event_csv.iterrows()]
        events[:, 2] = [conditions_mapping[row['stim_file']] for _, row in event_csv.iterrows()]
        
        # Get list of events (by order) that were disregarded
        existing_order1.extend(event_csv.loc[i, 'stim_file'] for i, row in event_csv.iterrows() if row['trigger'] in [5, 13, 17])
        existing_order2.extend(event_csv.loc[i, 'stim_file'] for i, row in event_csv.iterrows() if row['trigger'] not in [5, 13, 17])
        
        mne.write_events(fname_events, events, overwrite=True)
    
    order1 = [name for name in list(conditions_mapping.values()) if name not in existing_order1]
    order2 = [name for name in list(conditions_mapping.values()) if name not in existing_order2]
    
    events_disregarded = {"order 1": order1, "order 2": order2}
    
    with open(os.path.join(meg_dir, subject, "events_disregarded.json"), "w") as json_file:
        json.dump(events_disregarded, json_file)

def run_epochs(subject_id, name, tsss, fmin=0.5, fmax=4, frequency_band=None):
    subject = f"sub-{subject_id :02d}"
    print(f"Processing subject: {subject} {(f'(tSSS={tsss})' if tsss else '')}")

    data_path = os.path.join(meg_dir, subject)

    mapping = map_subjects[subject_id]

    raw_list = list()
    events_list = list()
    print("  Loading raw data")
    for run in range(1, 7):
        bads = list()
        bad_name = os.path.join('bads', mapping, f'run_{run :02d}_raw_tr.fif_bad')
        if os.path.exists(bad_name):
            with open(bad_name) as f:
                for line in f:
                    bads.append(line.strip())

        run_fname = os.path.join(data_path, f'run_{run :02d}_filt_tsss_{tsss}_raw.fif')
        raw = mne.io.read_raw_fif(run_fname, preload=True)
        #A fixed 34 ms delay exists between the appearance of a trigger in the MEG file (on channel STI101)
        #and the appearance of the stimulus on the screen

        delay = int(round(0.0345 * raw.info['sfreq']))
        events = mne.read_events(os.path.join(data_path, f'run_{run :02d}_{name}-eve.fif'))
        events[:, 0] = events[:, 0] + delay
        events_list.append(events)
        raw.info['bads'] = bads
        raw.interpolate_bads()
        if frequency_band is not None:
            raw.filter(fmin, fmax, n_jobs=-1,  #
               l_trans_bandwidth=1,  # make sure filter params are the same
               h_trans_bandwidth=1)  # in each band and skip "auto" option.
        raw_list.append(raw)

    raw, events = mne.concatenate_raws(raw_list, events_list=events_list)
    if frequency_band is not None:
        print('  Applying hilbert transform')
        # get analytic signal (envelope)
        raw.apply_hilbert(envelope=True)

    del raw_list

    picks = mne.pick_types(raw.info, meg=True, eeg=False, stim=False, eog=False, exclude=())

    # Epoch the data
    print('  Epoching')
    events_id=[events[i][2] for i in range(len(events))]
    epochs = mne.Epochs(raw, events, events_id, 0, 0.8, proj=True,
                        picks=picks, baseline=(-0.2, 0.0), preload=True,
                        reject=None, reject_tmax=reject_tmax, on_missing='warn')


    print('  Writing to disk')
    if frequency_band is not None:
        epochs.save(os.path.join(data_path, f'{subject}-tsss_{tsss}_{name}_{frequency_band}-epo.fif'), overwrite=True)
    else:
        epochs.save(os.path.join(data_path, f'{subject}-tsss_{tsss}_{name}-epo.fif'), overwrite=True)


if __name__ == '__main__':
    parser = get_similarity_parser()
    args = parser.parse_args()

    conditions_mapping ={}
    for i in range (1,151):
        conditions_mapping[f"meg/f{i:03d}.bmp"]=str(i)
        conditions_mapping[f"meg/u{i:03d}.bmp"]=str(i+150)
        conditions_mapping[f"meg/s{i:03d}.bmp"]=str(i+300)
    # Get events
    parallel, run_func, _ = parallel_func(run_events, n_jobs=N_JOBS)
    parallel(run_func(subject_id) for subject_id in range(1, 17))

    # Create epochs
    parallel, run_func, _ = parallel_func(run_epochs, n_jobs=max(N_JOBS // 4, 1))
    parallel(run_func(subject_id, args.stimuli_file_name, 10) for subject_id in range(1, 17))

    FREQ_BANDS = {
              "theta": [4.5, 8.5],
              "alpha": [8.5, 15.5],
              "beta": [15.5, 30],
              "Gamma1": [30, 60],
              "Gamma2": [60, 90]}

    subjects_ids=[i for i in range(1,17)]
    # Create epochs for power bands
    for  frequency_band, f in FREQ_BANDS.items():
        parallel, run_func, _ = parallel_func(run_epochs, n_jobs=max(N_JOBS // 4, 1))
        parallel(run_func(subject_id, args.stimuli_file_name, 10, f[0], f[1], frequency_band) for subject_id in range(1, 17))