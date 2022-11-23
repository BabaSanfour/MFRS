import os
import os.path as op
from warnings import warn

import numpy as np
import pandas as pd
from pandas import read_csv

import mne
from mne import Epochs
from mne.preprocessing import ICA
from mne.parallel import parallel_func
from mne.io import read_raw_fif, concatenate_raws

from library.config_bids import study_path, meg_dir, N_JOBS, l_freq, random_state, tmin, tmax, reject_tmax, map_subjects

def run_events(subject_id, selected, name, conditions_mapping):
    subject = "sub-%02d" % subject_id
    print("processing subject: %s" % subject)
    in_path = op.join(study_path, 'ds000117', subject, 'ses-meg/meg')
    out_path = op.join(meg_dir, subject)
    for run in range(1, 7):
        run_fname = op.join(in_path, 'sub-%02d_ses-meg_task-facerecognition_run-%02d_meg.fif' % (
            subject_id,run,))

        raw = mne.io.read_raw_fif(run_fname)
        mask = 4096 + 256  # mask for excluding high order bits
        events = mne.find_events(raw, stim_channel='STI101',
                                 consecutive='increasing', mask=mask,
                                 mask_type='not_and', min_duration=0.003)
        print(events.shape)
        print("  S %s - R %s" % (subject, run))
        csv_path = op.join(in_path, 'sub-%02d_ses-meg_task-facerecognition_run-%02d_events.tsv' % (
            subject_id,run,))
        event_csv = read_csv(csv_path, sep='\t')

        # select epochs depending on stimuli type in csv file
        not_selected=[stim_type for stim_type in ['Famous', 'Unfamiliar', 'Scrambled'] if stim_type not in selected[0]]
        for stim_type in not_selected:
            event_csv=event_csv[event_csv['stim_type']!=stim_type]

        # select epochs depending on stimuli number in events list
        not_selected=[event_numb for event_numb in [5,6,7,13,14,15,17,18,19] if event_numb not in selected[1]]
        new_events = []
        for i in range(len(events)):
            if events[i][2] not in not_selected:
                new_events.append(events[i])
        events = np.array(new_events)
        del new_events

        # fix mismatch between csv file and events from raw file.
        while len(event_csv)!=len(events):
            print(len(event_csv))
            print(len(events))
            assert len(events)>=len(event_csv), "The code removes item from events while this conditions need the other way around"
            print('event_csv: ',len(event_csv))
            print('events: ',len(events))

            for i in range(len(events)):
                if i >= len(event_csv) or (events[i][2] != event_csv.iloc[i,4]):
                    l=list(events)
                    a=l.pop(i)
                    events=np.array(l)
                    print('dropped index: ', i)
                    print('dropped list: ', a)
                    break

            del a

            print('new event_csv: ',len(event_csv))
            print('new events: ',len(events))

        #Select only one trial per condition
        for i in range(len(events)):
            events[i][2] = conditions_mapping[event_csv.iloc[i,5]]
        unique = []
        new_events = []
        for i in range(len(events)):
            if events[i][2] in unique:
                continue
            else:
                unique.append(events[i][2])
                new_events.append(events[i])
        events = np.array(new_events)
        del unique, new_events

        fname_events = op.join(out_path, 'run_%02d_%s-eve.fif' % (run, name))
        mne.write_events(fname_events, events, overwrite=True)

def run_epochs(subject_id, name, tsss=False):
    subject = "sub-%02d" % subject_id
    print("Processing subject: %s%s"
          % (subject, (' (tSSS=%d)' % tsss) if tsss else ''))

    data_path = op.join(meg_dir, subject)

    # map to correct subject for bad channels
    mapping = map_subjects[subject_id]

    raw_list = list()
    events_list = list()
    print("  Loading raw data")
    for run in range(1, 7):
        bads = list()
        bad_name = op.join('bads', mapping, 'run_%02d_raw_tr.fif_bad' % run)
        if os.path.exists(bad_name):
            with open(bad_name) as f:
                for line in f:
                    bads.append(line.strip())

        if tsss:
            run_fname = op.join(data_path, 'run_%02d_filt_tsss_%d_raw.fif'
                                % (run, tsss))
        else:
            run_fname = op.join(data_path, 'run_%02d_filt_sss_'
                                'highpass-%sHz_raw.fif' % (run, l_freq))

        raw = mne.io.read_raw_fif(run_fname, preload=True)
        #A fixed 34 ms delay exists between the appearance of a trigger in the MEG file (on channel STI101)
        #and the appearance of the stimulus on the screen

        delay = int(round(0.0345 * raw.info['sfreq']))
        events = mne.read_events(op.join(data_path, 'run_%02d_%s-eve.fif' % (run, name)))
        events[:, 0] = events[:, 0] + delay
        events_list.append(events)
        raw.info['bads'] = bads
        raw.interpolate_bads()
        raw_list.append(raw)

    raw, events = mne.concatenate_raws(raw_list, events_list=events_list)
    del raw_list

    picks = mne.pick_types(raw.info, meg=True, eeg=False, stim=False,
                           eog=False, exclude=())

    # Epoch the data
    print('  Epoching')
    events_id=[events[i][2] for i in range(len(events))]
    epochs = mne.Epochs(raw, events, events_id, 0, 0.8, proj=True,
                        picks=picks, baseline=None, preload=False,
                        reject=None, reject_tmax=reject_tmax, on_missing='warn')

    print('  Writing to disk')
    if tsss:
        epochs.save(op.join(data_path, '%s-tsss_%d_%s-epo.fif' % (subject, tsss, name)), overwrite=True)
    else:
        epochs.save(op.join(data_path, '%s_highpass-%sHz-epo.fif'
                    % (subject, l_freq)), overwrite=True)


if __name__ == '__main__':

    conditions_mapping ={}
    for i in range (1,151):
        conditions_mapping["meg/f%03d.bmp"%i]=i
        conditions_mapping["meg/u%03d.bmp"%i]=i+150
        conditions_mapping["meg/s%03d.bmp"%i]=i+300

    # Get events
    parallel, run_func, _ = parallel_func(run_events, n_jobs=N_JOBS)
    parallel(run_func(subject_id, [['Famous', 'Unfamiliar'], [5,6,7, 13,14,15,]], 'FamUnfam', conditions_mapping) for subject_id in range(1, 17))
    print("FamUnfam")

    # Create epochs
    parallel, run_func, _ = parallel_func(run_epochs, n_jobs=max(N_JOBS // 4, 1))
    parallel(run_func(subject_id, 'FamUnfam', 10) for subject_id in range(1, 17))
