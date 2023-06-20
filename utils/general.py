import pickle
import numpy as np
import os
from utils.config import meg_dir, rdms_folder

def save_pickle(dictionary, file):
    """ Save dictionary in pickle files """
    if file is None:
        print("\nFile name not specificed.\n")
        print("\nContinuing without saving the file .\n")
    else:
        with open(file, 'wb') as f:
            pickle.dump(dictionary, f)
        print(f'File {os.path.basename(file)} saved successfully')


def load_pickle(file):
    """ Load dictionary from pickle files """
    with open(file, 'rb') as f:
        dictionary = pickle.load(f)
    print(f'File {os.path.basename(file)} loaded successfully')
    return dictionary

def save_npy(file, file_name):
    """ Save numpy files """
    np.save(file_name, file)
    print(f'File {os.path.basename(file_name)} saved successfully')

def load_npy(file):
    """ Load numpy files """
    rdms = np.load(file)
    print(f'File {os.path.basename(file)} loaded successfully')
    return rdms

def load_meg_rdm(stimuli_file = "FamUnfam", power = None, sub_opt = 1, ch_opt = 1):
    if power == None:
        out_file = os.path.join(meg_dir, f"RDMs_{stimuli_file}_16-subject_{sub_opt}-sub_opt{ch_opt}-chl_opt.npy")
    else:
        out_file = os.path.join(meg_dir, f"RDMs_{stimuli_file}_{power}_16-subject_{sub_opt}-sub_opt{ch_opt}-chl_opt.npy")
    meg_rdm = load_npy(out_file)
    return meg_rdm

def load_model_rdm(stimuli_file = "FamUnfam", model_name = "resnet50", activ_type = "trained", type = "main"):
    out_file = os.path.join(rdms_folder, f"{model_name}_{stimuli_file}_rdm_{type}_{activ_type}.npy")
    model_rdm = load_npy(out_file)
    return model_rdm