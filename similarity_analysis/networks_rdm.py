import os
import numpy as np
from scipy.stats import pearsonr
import sys
sys.path.append('/home/hamza97/MFRS/utils')
from general import save_npy, load_npy
from config_sim_analysis import rdms_folder, activations_folder

def limtozero(x):

    """
        zero the value close to zero
        Parameters
        ----------
        x : float
        Returns
        -------
        0
    """

    if x < 1e-15:
        x = 0

    return x


def oneRDM(activations: np.array, cons: int = None, save: bool = False, file_name: str = None, abs: bool = False):

    """
        Calculate the Representational Dissimilarity Matrix(Matrice) - RDM(s) for
            One Artificial Neural Network layer activations or for the Network concatenated activations
        Parameters
        ----------
        activations: The layer/concatenated network activations, shape [n_cons, n_neurons].
                                n_cons & n_neurons represent the number of conidtions & the number of neurons respectively.
        abs :        Calculate the absolute value of Pearson r or not, default is True.
        save         Save computed similarity scores stats, default: False.
        file_name:   The name of the file in which the computed rdm will be saved.

        Returns
        -------
        RDM(s) : array
            The activations RDM. The shape is [n_cons, n_cons].
    """

    if len(np.shape(activations)) != 2:

        print("\nThe shape of input for oneRDM function must be [n_cons, n_neurons].\n")

        return "Invalid input!"

    # initialize the RDM
    rdm = np.zeros([cons, cons], dtype=np.float64)
    for i in range(cons):
        # RDMs are symetric, so we calculate only the upper half and assign the values to the lower one.
        for j in range(i+1, cons):
            # calculate the Pearson Coefficient
            r = pearsonr(activations[i], activations[j])[0]
            # calculate the dissimilarity
            if abs == True:
                value = limtozero(1 - np.abs(r))
                rdm[j, i], rdm[i, j] = value, value
            else:
                value = limtozero(1 - r)
                rdm[j, i], rdm[i, j] = value, value

    if save:
        save_npy(rdm, file_name)


    return rdm

def groupRDMs(network_activations: dict, cons: int, method: str = "correlation", abs: bool = False,
                            save: bool = False, file_name: str = None):

    """
        Calculate the Representational Dissimilarity Matrix(Matrices) - RDM(s) for Artificial Neural Networks layers activations
        Parameters
        ----------
        network_activations:   The network  layers activations. Keys are layer names and values are layers activations.
                                    The shape of layers activations must be [n_cons, n_neurons].
                                    n_cons & n_neurons represent the number of conidtions & the number of neurons respectively.
        abs :                  Calculate the absolute value of Pearson r or not, default is True. Only works when method='correlation'.
        save                   Save computed similarity scores stats, default: False.
        file_name:             The name of the file in which the computed rdm will be saved.


        Returns
        -------
        RDM(s) : array
            The activations RDM. The shape is [n_layers, n_cons, n_cons].
    """

    # Get the number of layers
    n_layers = len(network_activations)

    # initialize the RDMs
    rdms = np.zeros([n_layers, cons, cons], dtype=np.float64)

    # counter
    l=0

    # calculate the values in RDMs
    for layer, activations in network_activations.items():
        rdms[l] = oneRDM(activations, cons)
        l+=1

    if save:
        save_npy(rdms, file_name)

    print("\nRDMs computing finished!")

    return rdms

def get_network_rdm(name: str, cons: int = 300, save: bool = True):
    """Get the model RDM from the concatinated activations of the main layers"""
    if os.path.exists(os.path.join(rdms_folder, '%s_model_rdm.npy'%name)):
        rdm=load_npy(os.path.join(rdms_folder, '%s_model_rdm.npy'%name))
        return rdm
    else:
        activations=load_npy(os.path.join(activations_folder, '%s_activations_model.npy'%name))
        model_rdm = oneRDM(activations, cons, save, os.path.join(rdms_folder, '%s_model_rdm.npy'%name))
        print("\nRDMs computing finished!")
        return model_rdm
