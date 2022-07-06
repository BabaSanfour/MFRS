import pickle
import numpy as np
import os

def save_pickle(dictionary, file):
    """ Save dictionary in pickle files """
    if file is None:
        print("\nFile name not specificed.\n")
        print("\nContinuing without saving the file .\n")
    else:
        with open(file, 'wb') as f:
            pickle.dump(dictionary, f)
        print('File %s saved successfully'%os.path.basename(file))


def load_pickle(file):
    """ Load dictionary from pickle files """
    with open(file, 'rb') as f:
        dictionary = pickle.load(f)
    print('File %s loaded successfully'%os.path.basename(file))
    return dictionary

def save_npy(rdm, file):
    """ Save numpy files """
    if file is None:
        print("\nFile name not specificed.\n")
        print("\nContinuing without saving the file .\n")
    else:
        np.save(file, rdm)
        print('File saved successfully')

def load_npy(file):
    """ Load numpy files """
    rdms = np.load(file)
    print('File %s loaded successfully'%os.path.basename(file))
    return rdms
