import os
import sys
import mne
import json
import numpy as np
from pandas import read_csv
from mne.parallel import parallel_func

sys.path.append('../../MFRS/')
from utils.config import study_path, meg_dir, reject_tmax, map_subjects, conditions_mapping


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
    
    existing_trial1 = []
    
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

        # Create a new list to store filtered events
        filtered_events = []

        # Iterate through the events list and update the trigger values
        for idx, row in event_csv.iterrows():
            if row['trigger'] in [5, 13, 17]:
                filtered_events.append(events[idx])
                events[idx][2] = conditions_mapping[row['stim_file']]        
            # Check if the event should be included in the filtered_events list
                
            existing_trial1.append(str(events[idx][2]))
        # Replace the events list with the filtered_events list
        events = filtered_events

        mne.write_events(fname_events, np.array(events), overwrite=True)
        
    trial1 = [name for name in list(conditions_mapping.values()) if name not in existing_trial1]
    events_disregarded = {"trial 1": trial1}
    with open(os.path.join(meg_dir, subject, "events_disregarded.json"), "w") as json_file:
        json.dump(events_disregarded, json_file)


def regroup_disregarded_events() -> None:
    """
    Regroups disregarded events from individual subject JSON files into a mega JSON file.

    This function iterates through subject folders and reads the 'events_disregarded.json' files
    for each subject. It aggregates the 'trial 1' lists from each file into a single list for
    the mega JSON file. Additionally, it identifies stimulus IDs that are not present in the celebA
    dataset or the list of IDs associated with celebA images. These IDs are also added to the mega
    disregarded events list. The mega JSON file is then saved as 'mega_events_disregarded.json' in
    the main MEG directory.

    Returns:
        None
    """
    # Initialize the list to store all disregarded events
    all_disregarded = []

    # Iterate through each subject's folder
    for subject_id in range(1, 17):
        subject_folder = f"sub-{subject_id:02d}"
        json_path = os.path.join(meg_dir, subject_folder, "events_disregarded.json")
        
        if os.path.exists(json_path):
            # Load the individual subject's disregarded events data
            with open(json_path, "r") as json_file:
                data = json.load(json_file)
                # Extend the list with 'trial 1' events from the current subject
                all_disregarded.extend(data.get("trial 1", []))

    # List of IDs associated with celebA images
    ids_in_celebA = [4, 5, 8, 11, 16, 19, 20, 29, 36, 101, 102, 105, 107, 109, 114, 123, 125, 130, 136,
                     138, 140, 143, 52, 53, 57, 62, 65, 68, 69, 80, 81]
    
    # List of IDs that have been added to celebA
    ids_added_to_celebA = [int(d[1:]) for d in os.listdir(os.path.join(study_path, "web_scrapping_stimuli")) if os.path.isdir(os.path.join(os.path.join(study_path, "web_scrapping_stimuli"), d))]

    # Generate a set of all possible IDs (1 to 150)
    all_ids = set(range(1, 151))
    added_ids_set = set(ids_added_to_celebA)

    # Find IDs that are not in celebA and not in the list of ids_in_celebA
    stimuli_not_in_celebA = all_ids - added_ids_set - set(ids_in_celebA)

    # Convert the IDs to strings and add them to the list of disregarded events
    all_disregarded.extend(map(str, stimuli_not_in_celebA))
    
    # Create the mega disregarded events dictionary
    mega_events_disregarded = {
        "all": all_disregarded,
    }
    
    # Save the mega JSON file
    mega_json_path = os.path.join(meg_dir, "mega_events_disregarded.json")
    with open(mega_json_path, "w") as mega_json_file:
        json.dump(mega_events_disregarded, mega_json_file)
    
    print(mega_events_disregarded)
    print("Mega disregarded events file created.")


def run_epochs(subject_id: int, fmin: float = 0.5, fmax: float = 4, frequency_band: str = None) -> None:
    """
    Process and epoch raw MEG data for a specific subject.

    Args:
        subject_id (int): The ID of the subject.
        fmin (float): The lower frequency bound for filtering.
        fmax (float): The upper frequency bound for filtering.
        frequency_band (str): The frequency band name (optional).

    Returns:
        None
    """
    subject = f"sub-{subject_id:02d}"
    print(f"Processing subject: {subject}")

    data_path = os.path.join(meg_dir, subject)

    mapping = map_subjects[subject_id]

    raw_list = []
    events_list = []

    print("  Loading raw data")
    for run in range(1, 7):
        bads = []

        bad_name = os.path.join('bads', mapping, f'run_{run:02d}_raw_tr.fif_bad')
        if os.path.exists(bad_name):
            with open(bad_name) as f:
                bads = [line.strip() for line in f]

        run_fname = os.path.join(data_path, f'run_{run:02d}_filt_raw.fif')
        raw = mne.io.read_raw_fif(run_fname, preload=True, verbose=False)

        # A fixed 34 ms delay between trigger and stimulus
        delay = int(round(0.0345 * raw.info['sfreq']))
        events = mne.read_events(os.path.join(data_path, f'run_{run:02d}-eve.fif'))
        events[:, 0] = events[:, 0] + delay
        events_list.append(events)

        raw.info['bads'] = bads
        raw.interpolate_bads()

        if frequency_band is not None:
            raw.filter(fmin, fmax, n_jobs=-1, l_trans_bandwidth=1, h_trans_bandwidth=1)

        raw_list.append(raw)

    raw, events = mne.concatenate_raws(raw_list, events_list=events_list)
    del raw_list

    with open(os.path.join(meg_dir, "mega_events_disregarded.json"), "r") as mega_json_file:
        mega_events_disregarded = json.load(mega_json_file)

    disregarded_ids = [int(id) for id in mega_events_disregarded.get("all", [])]

    filtered_events = [event for event in events if event[2] not in disregarded_ids]
    events = np.array(filtered_events)

    if frequency_band is not None:
        print('  Applying hilbert transform')
        raw.apply_hilbert(envelope=True)

    picks = mne.pick_types(raw.info, meg=True, eeg=False, stim=False, eog=False, exclude=())

    print('  Epoching')
    events_id = [event[2] for event in events]
    epochs = mne.Epochs(raw, events, events_id, -0.2, 0.8, proj=True,
                        picks=picks, baseline=(-0.2, 0.0), preload=True,
                        reject=None, reject_tmax=reject_tmax, on_missing='warn')

    print('  Writing to disk')
    if frequency_band is not None:
        epochs.save(os.path.join(data_path, f'{subject}-{frequency_band}-epo.fif'), overwrite=True)
    else:
        epochs.save(os.path.join(data_path, f'{subject}-epo.fif'), overwrite=True)


if __name__ == '__main__':
    # Get events
    parallel, run_func, _ = parallel_func(run_events, n_jobs=-1)
    parallel(run_func(subject_id) for subject_id in range(1, 17))

    # Get all disregarded events
    regroup_disregarded_events()

    # # Create epochs
    parallel, run_func, _ = parallel_func(run_epochs, n_jobs=-1)
    parallel(run_func(subject_id) for subject_id in range(1, 17))


    # Uncomment for sensor space analysis
    # from utils.arg_parser import get_similarity_parser

    # parser = get_similarity_parser()
    # args = parser.parse_args()

    # FREQ_BANDS = {
    #           "theta": [4.5, 8.5],
    #           "alpha": [8.5, 15.5],
    #           "beta": [15.5, 30],
    #           "Gamma1": [30, 60],
    #           "Gamma2": [60, 90]}

    # subjects_ids=[i for i in range(1,17)]
    # # Create epochs for power bands
    # for  frequency_band, f in FREQ_BANDS.items():
    #     parallel, run_func, _ = parallel_func(run_epochs, n_jobs=max(N_JOBS // 4, 1))
    #     parallel(run_func(subject_id, 10, f[0], f[1], frequency_band) for subject_id in range(1, 17))
