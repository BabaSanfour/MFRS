import os
import sys
import mne
import numpy as np
from pandas import read_csv
sys.path.append('../../MFRS/')
from utils.library.config_bids import study_path, meg_dir, l_freq, reject_tmax, map_subjects

def run_events(subject_id, selected, name, conditions_mapping):
    subject = "sub-%02d" % subject_id
    print("processing subject: %s" % subject)
    in_path = os.path.join(study_path, 'ds000117', subject, 'ses-meg/meg')
    out_path = os.path.join(meg_dir, subject)
    for run in range(1, 7):
        run_fname = os.path.join(in_path, 'sub-%02d_ses-meg_task-facerecognition_run-%02d_meg.fif' % (
            subject_id,run,))

        raw = mne.io.read_raw_fif(run_fname)
        mask = 4096 + 256  # mask for excluding high order bits
        events = mne.find_events(raw, stim_channel='STI101',
                                 consecutive='increasing', mask=mask,
                                 mask_type='not_and', min_duration=0.003)
        print("  S %s - R %s" % (subject, run))
        csv_path = os.path.join(in_path, 'sub-%02d_ses-meg_task-facerecognition_run-%02d_events.tsv' % (
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

        fname_events = os.path.join(out_path, 'run_%02d_%s-eve.fif' % (run, name))
        mne.write_events(fname_events, events, overwrite=True)

def run_epochs(subject_id, name, tsss, fmin=0.5, fmax=4, frequency_band=None):
    subject = "sub-%02d" % subject_id
    print("Processing subject: %s%s"
          % (subject, (' (tSSS=%d)' % tsss) if tsss else ''))

    data_path = os.path.join(meg_dir, subject)

    # map to correct subject for bad channels
    mapping = map_subjects[subject_id]

    raw_list = list()
    events_list = list()
    print("  Loading raw data")
    for run in range(1, 7):
        bads = list()
        bad_name = os.path.join('bads', mapping, 'run_%02d_raw_tr.fif_bad' % run)
        if os.path.exists(bad_name):
            with open(bad_name) as f:
                for line in f:
                    bads.append(line.strip())

        if tsss:
            run_fname = os.path.join(data_path, 'run_%02d_filt_tsss_%d_raw.fif'
                                % (run, tsss))
        else:
            run_fname = os.path.join(data_path, 'run_%02d_filt_sss_'
                                'highpass-%sHz_raw.fif' % (run, l_freq))

        raw = mne.io.read_raw_fif(run_fname, preload=True)
        #A fixed 34 ms delay exists between the appearance of a trigger in the MEG file (on channel STI101)
        #and the appearance of the stimulus on the screen

        delay = int(round(0.0345 * raw.info['sfreq']))
        events = mne.read_events(os.path.join(data_path, 'run_%02d_%s-eve.fif' % (run, name)))
        events[:, 0] = events[:, 0] + delay
        events_list.append(events)
        raw.info['bads'] = bads
        raw.interpolate_bads()
        if frequency_band is not None:
            raw.filter(fmin, fmax, n_jobs=-1,  # use more jobs to speed up.
               l_trans_bandwidth=1,  # make sure filter params are the same
               h_trans_bandwidth=1)  # in each band and skip "auto" option.

        raw_list.append(raw)

    raw, events = mne.concatenate_raws(raw_list, events_list=events_list)
    del raw_list

    picks = mne.pick_types(raw.info, meg=True, eeg=False, stim=False,
                           eog=False, exclude=())

    # Epoch the data
    print('  Epoching')
    events_id=[events[i][2] for i in range(len(events))]
    epochs = mne.Epochs(raw, events, events_id, 0, 0.8, proj=True,
                        picks=picks, baseline=None, preload=True,
                        reject=None, reject_tmax=reject_tmax, on_missing='warn')
    if frequency_band is not None:
        print('  Applying hilbert transform')
        # remove evoked response
        epochs.subtract_evoked()

        # get analytic signal (envelope)
        epochs.apply_hilbert(envelope=True)


    print('  Writing to disk')
    if tsss:
        if frequency_band is not None:
            epochs.save(os.path.join(data_path, '%s-tsss_%d_%s_%s-epo.fif' % (subject, tsss, name, frequency_band)), overwrite=True)
        else:
            epochs.save(os.path.join(data_path, '%s-tsss_%d_%s-epo.fif' % (subject, tsss, name)), overwrite=True)
    else:
        epochs.save(os.path.join(data_path, '%s_highpass-%sHz-epo.fif'
                    % (subject, l_freq)), overwrite=True)
