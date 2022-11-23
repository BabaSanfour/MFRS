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

import sys
sys.path.append('/home/hamza97/MFRS/utils')
from general import save_pickle, load_pickle



def get_powers_freqs(subjects_ids: list, data_type: str):
    subjects_power={}
    file_name = op.join(meg_dir, "psds_frqs_%s.pkl"%data_type)
    if op.isfile(file_name):
        return 0
    for subject_id in subjects_ids:
        subject = "sub-%02d" % subject_id
        print("Processing subject: %s"% subject)
        data_path = op.join(meg_dir, subject)
        epochs = mne.read_epochs(op.join(data_path, '%s-tsss_10_%s-epo.fif' % (subject, data_type)), preload=True)
        chan_names_p1=list(epochs.copy().pick_types(meg='planar1').info["ch_names"])
        psds_p1,freqs_p1=mne.time_frequency.psd_array_multitaper(epochs.copy().pick_types(meg='planar1').get_data(),epochs.info['sfreq'])
        chan_names_p2=list(epochs.copy().pick_types(meg='planar2').info["ch_names"])
        psds_p2,freqs_p2=mne.time_frequency.psd_array_multitaper(epochs.copy().pick_types(meg='planar2').get_data(),epochs.info['sfreq'])
        psds_grad=(psds_p2+psds_p1)/2
        freqs_grad=(freqs_p2+freqs_p1)/2
        chan_names_mag=list(epochs.copy().pick_types(meg='mag').info["ch_names"])
        psds__mag,freqs__mag=mne.time_frequency.psd_array_multitaper(epochs.copy().pick_types(meg='mag').get_data(),epochs.info['sfreq'])

        subjects_power[subject_id]= [[psds_grad, freqs_grad], [psds__mag, freqs__mag]]
    save_pickle(subjects_power, file_name)

def get_bands_freqs(FREQ_BANDS : dict = {"delta": [0.5, 4.5]}, data_type: str = "FamUnfam", scale: bool = True):
    file_name = op.join(meg_dir, "bands_psds_frqs_%s.pkl"%data_type)
    if op.isfile(file_name):
        return 0
    subjects_power=load_pickle(op.join(meg_dir, "psds_frqs_%s.pkl"%data_type))
    results=dict()
    for band_name, band in FREQ_BANDS.items():
        for chls, name in enumerate(["grad", "mag"]):
            X = [] # Data Vector
            Y = [] # Labels
            G = [] # Groups (Subjects)
            for subject_id, values in subjects_power.items():
                values=values[chls]
                psds,freqs=values[0], values[1]
                band_idxs = np.where((freqs >= band[0]) & (freqs <= band[1]))[0]
                power = np.mean(psds[:,:,band_idxs],axis=-1)
                # select equal nbre of trials for 2 classes:
                # counter=0
                # for ev in epochs.events:
                #     if ev[2]==2:
                #         counter+=1
                i=0
                Xi, Yi, Gi = [], [], []
                for ev in epochs.events:
                    # if (ev[2]==1 and counter>0) or ev[2]==2:
                    #     if (ev[2]==1 and counter>0):
                    #         counter-=1
                    Xi.append(power[i])
                    Yi.append(ev[2])
                    Gi.append(subject_id)
                    # i+=1
                X.append(Xi)
                Y.append(Yi)
                G.append(Gi)
            X = np.concatenate(X,axis=0) # axis = 0 --> epochs, 1 would be the channels
            Y = np.concatenate(Y)
            G = np.concatenate(G)
            if scale:
                scaler = StandardScaler()
                X = scaler.fit_transform(X)
            results['$s_%s'%(band_name, name)]= [X, Y, G]
    save_pickle(results, file_name)

if __name__ == "__main__":

    FREQ_BANDS = {"delta": [0.5, 4.5],
              "theta": [4.5, 8.5],
              "alpha": [8.5, 11.5],
              "sigma": [11.5, 15.5],
              "beta": [15.5, 30],
              "Gamma1": [30, 70],
              "Gamma2": [71, 150]}

    subjects_ids=[i for i in range(1,17)]
    get_powers_freqs(subjects_ids, "FamUnfam")
    get_bands_freqs(FREQ_BANDS, "FamUnfam")
    print("done")
