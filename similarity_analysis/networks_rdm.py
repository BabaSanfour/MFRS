import os
import numpy as np
from scipy.stats import pearsonr

rdms_dir="/home/hamza97/scratch/data/MFRS_data/networks_rdms"

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


def layerRDM(layer_activations: np.array, cons: int = None, method: str = "correlation", abs: bool = False):

    """
    Calculate the Representational Dissimilarity Matrix(Matrice) - RDM(s) for One Artificial Neural Network layer activations
    Parameters
    ----------
    layer_activations: The layer activations, shape [n_cons, n_neurons].
                            n_cons & n_neurons represent the number of conidtions & the number of neurons respectively.
    method:            'correlation' or 'euclidean' or 'mahalanobis'. Default is 'correlation'.
                            The method to calculate the dissimilarities.
                            If method='correlation', the dissimilarity is calculated by Pearson Correlation.
                            If method='euclidean', the dissimilarity is calculated by Euclidean Distance, the results will be normalized.
                            If method='mahalanobis', the dissimilarity is calculated by Mahalanobis Distance, the results will be normalized.
    abs :              Calculate the absolute value of Pearson r or not, default is True. Only works when method='correlation'.

    Returns
    -------
    RDM(s) : array
        The activations RDM. The shape is [n_cons, n_cons].
    """

    if len(np.shape(layer_activations)) != 2:

        print("\nThe shape of input for layerRDM function must be [n_cons, n_neurons].\n")

        return "Invalid input!"

    # initialize the RDM
    rdm = np.zeros([cons, cons], dtype=np.float64)
    for i in range(cons):
        # RDMs are symetric, so we calculate only the upper half and assign the values to the lower one.
        for j in range(i+1, cons):
            if method == 'correlation':
                # calculate the Pearson Coefficient
                r = pearsonr(layer_activations[i], layer_activations[j])[0]
                # calculate the dissimilarity
                if abs == True:
                    value = limtozero(1 - np.abs(r))
                    rdm[j, i], rdm[i, j] = value, value
                else:
                    value = limtozero(1 - r)
                    rdm[j, i], rdm[i, j] = value, value
            elif method == 'euclidean':
                value = np.linalg.norm(layer_activations[i]-layer_activations[j])
                rdm[j, i], rdm[i, j] = value, value
            elif method == 'mahalanobis':
                X = np.transpose(np.vstack((layer_activations[i], layer_activations[j])), (1, 0))
                X = np.dot(X, np.linalg.inv(np.cov(X, rowvar=False)))
                value = np.linalg.norm(X[:, 0]-X[:, 1])
                rdm[j, i], rdm[i, j] = value, value
    if method in ['euclidean', 'mahalanobis']:
        max = np.max(rdm)
        min = np.min(rdm)
        rdm = (rdm-min)/(max-min)

    return rdm

def networkRDM(network_activations: dict, cons: int, avg_rdm: int = 0, method: str = "correlation", abs: bool = False,
                            save: bool = False, model_name: str = None, data_name: str = None):

    """
    Calculate the Representational Dissimilarity Matrix(Matrices) - RDM(s) for Artificial Neural Networks layers activations
    Parameters
    ----------
    network_activations:   The network  layers activations. Keys are layer names and values are layers activations.
                                The shape of layers activations must be [n_cons, n_neurons].
                                n_cons & n_neurons represent the number of conidtions & the number of neurons respectively.
    avg_rdm:               Return the results for each layer or after averaging, default is 0.
                                If avg_rdm=0, return the results of each layer.
                                If avg_rdm=1, return the average result.
    method:            'correlation' or 'euclidean' or 'mahalanobis'. Default is 'correlation'.
                            The method to calculate the dissimilarities.
                            If method='correlation', the dissimilarity is calculated by Pearson Correlation.
                            If method='euclidean', the dissimilarity is calculated by Euclidean Distance, the results will be normalized.
                            If method='mahalanobis', the dissimilarity is calculated by Mahalanobis Distance, the results will be normalized.
    abs :              Calculate the absolute value of Pearson r or not, default is True. Only works when method='correlation'.
    save                save computed similarity scores stats, default: False
    model_name          network name, required if save=True. Default: None
    data_name           data name, required if save=True. Default: None

    Returns
    -------
    RDM(s) : array
        The activations RDM.
        If avg_rdm=0, return n_layers RDMs. The shape is [n_layers, n_cons, n_cons].
        If avg_rdm=1, return only one RDM. The shape is [n_cons, n_cons].
    """
    if save:
        assert model_name != None, "\nmodel_name not specified.\n Invalid input!"
        assert data_name != None, "\ndata_name not specified.\n Invalid input!"

    # Get the number of layers
    n_layers = len(network_activations)

    # initialize the RDMs
    rdms = np.zeros([n_layers, cons, cons], dtype=np.float64)

    # counter
    l=0

    # calculate the values in RDMs
    for layer, layer_activations in network_activations.items():
        rdms[l] = layerRDM(layer_activations, cons)
        l+=1

    if avg_rdm == 0:
        if save:
            file=os.path.join(rdms_dir, "%s_%s_data_rdm.npy"%(model_name, data_name))
            save_rdms(rdms, file)

        print("\nRDMs computing finished!")

        return rdms

    if avg_rdm == 1:

        rdms = np.average(rdms, axis=0)

        if save:
            file=os.path.join(rdms_dir, "%s_%s_data_avg_rdm.npy"%(model_name, data_name))
            save_rdms(rdms, file)

        print("\nRDM computing finished!")

        return rdms

def save_rdms(rdm, file_name):
    out_file = os.path.join(rdms_dir, file_name)
    np.save(out_file, rdm)
    print('File saved successfully')

def load_rdms(file_name):
    out_file = os.path.join(rdms_dir, file_name)
    rdms = np.load(out_file)
    print('File loaded successfully')
    return rdms
