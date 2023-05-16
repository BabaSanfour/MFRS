import os
import sys
import numpy as np
from tqdm import tqdm
from scipy.stats import pearsonr
sys.path.append('../../../MFRS/')
from utils.config import get_similarity_parser
from utils.general import save_npy, load_npy, load_pickle
from utils.config_sim_analysis import rdms_folder, activations_folder

def limtozero(value: float) -> float:
    """
    Rounds a float value to zero if it's close enough (within 1e-15).
    Parameters:
        value (float): The float value to round to zero.
    Returns:
        The original value rounded to zero if it's close enough, otherwise the original value.
    """
    if abs(value) < 1e-15:
        value = 0
    return value


def oneRDM(activations: np.array, cons: int, stimuli_file: str = None, model_name: str = None, save: bool = False, activ_type: str = "main", abs: bool = False) -> np.array:
    """
    Calculates the Representational Dissimilarity Matrix (RDM) for one Artificial Neural Network layer
    activations or for the network concatenated activations
    Parameters:
    ----------
    activations : np.array
        The layer/concatenated network activations, shape [n_cons, n_neurons].
        n_cons & n_neurons represent the number of conditions & the number of neurons respectively.
    model_name : str
        The name of the model.
    stimuli_file_name : str
        The name of the stimuli file.    
    cons : int
        The number of conditions.
    save : bool
        Save computed similarity scores stats, default: False.
    absolute : bool
        Calculate the absolute value of Pearson r or not, default is True.
    Returns:
    -------
    RDM : np.array
        The activations RDM. The shape is [n_cons, n_cons].
    """
    if activations.ndim != 2:
        raise ValueError("The input for oneRDM function must have shape [n_cons, n_neurons].")

    # initialize the RDM
    rdm = np.zeros([cons, cons], dtype=np.float64)
    for i in tqdm(range(cons), desc="Calculating RDM"):
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
        rdm_path = os.path.join(rdms_folder, f"{model_name}_{stimuli_file}_rdm_{activ_type}.npy")
        save_npy(rdm, rdm_path)
    return rdm

def groupRDMs(layers_activations: list, model_name: str, stimuli_file: str, cons: int, 
              save: bool = False, activ_type: str = "trained", abs: bool = False) -> np.array:
    """
    Calculate the Representational Dissimilarity Matrix(Matrices) - RDM(s) for Artificial Neural Networks layers activations.

    Parameters
    ----------
    layers_activations : list of arrays
        The arrays must be [n_cons, n_neurons].
        n_cons & n_neurons represent the number of conditions & the number of neurons, respectively.
    model_name : str
        The name of the model.
    stimuli_file_name : str
        The name of the stimuli file.
    cons : int
        The number of conditions.
    abs : bool, optional
        Calculate the absolute value of Pearson r or not, default is False. Only works when method='correlation'.
    save : bool, optional
        Save computed similarity scores stats, default: False.
    file_name : str, optional
        The name of the file in which the computed rdm will be saved.

    Returns
    -------
    rdms : array
        The activations RDM. The shape is [n_layers, n_cons, n_cons].
    """
    rdm_path = os.path.join(rdms_folder, f"{model_name}_{stimuli_file}_rdm_main_{activ_type}.npy")
    if os.path.exists(rdm_path):
        print(f"Loading pre-computed RDM for model mainlayers {model_name}, {stimuli_file}, {activ_type}...")
        rdms = load_npy(rdm_path)
        return rdms
    n_layers = len(layers_activations)
    rdms = np.zeros([n_layers, cons, cons], dtype=np.float64)
    for l, activations in tqdm(enumerate(layers_activations), desc="Calculating RDMs", total=n_layers):
        rdms[l] = oneRDM(activations, cons, abs=abs)
    if save:
        save_npy(rdms, rdm_path)
        print(f"RDMs saved to {rdm_path}.")
    print("\nRDMs computation finished!")
    return rdms

def get_network_rdm(model_activations: np.array, model_name: str, stimuli_file: str, cons: int = 300, save: bool = True, activ_type: str = "trained") -> np.array:
    """
    Calculate the model Representational Dissimilarity Matrix (RDM) from the concatenated activations of the main layers

    Parameters:
    -----------
    model_name : str
        The name of the model.
    stimuli_file_name : str
        The name of the stimuli file.
    cons : int, optional
        The number of conditions (stimuli) used as input to the model. Default is 300.
    save : bool, optional
        Whether to save the computed RDM. Default is True.

    Returns:
    --------
    rdm : array
        The computed model RDM. The shape is [cons, cons].
    """

    rdm_path = os.path.join(rdms_folder, f"{model_name}_{stimuli_file}_rdm__model_{activ_type}.npy")

    if os.path.exists(rdm_path):
        print(f"Loading pre-computed RDM for model activations: {model_name}, {stimuli_file}, {activ_type}...")
        rdm = load_npy(rdm_path)
    else:
        print(f"Computing RDM for model {model_name}, {stimuli_file}, {activ_type}...")
        rdm = oneRDM(model_activations, cons, save=save)
        save_npy(rdm, rdm_path)

    return rdm

def extract_network_rdms(cons: int, stimuli_file: str, model_name: str, save: bool = True, activ_type: str = "trained"):
    """
    Calculate the RDMs of the model layers and the model RDM.
    
    Parameters:
    -----------
    cons : int
        The number of images to used as stimuli.
    stimuli_file_ : str
        The name of the stimuli file.
    model_name : str
        The name of the model.
    save : bool, optional
        Whether or not to save the extracted activations.
        
    Returns:
    --------
    Model Layers' RDMs (np.array [n_layers, n_cons, n_cons]) and Model RDM (np.array [n_cons, n_cons])
    """
    rdm_file = os.path.join(rdms_folder, f"{model_name}_{stimuli_file}_rdm_main_{activ_type}.npy")
    activations_file = os.path.join(activations_folder, f"{model_name}_{stimuli_file}_activations_main_{activ_type}.pkl")
    activations = load_pickle(activations_file)
    activations = activations.values()
    rdms = groupRDMs(activations, model_name, stimuli_file, cons, save = save, activ_type = activ_type)
    rdm_file = os.path.join(rdms_folder, f"{model_name}_{stimuli_file}_rdm_model_{activ_type}.npy")
    if os.path.isfile(rdm_file):
        print(f"RDM file file (data: {model_name}) for {stimuli_file} already exists!!!")
        model_rdm = load_npy(rdm_file)
    else:
        activations_file = os.path.join(activations_folder, f"{model_name}_{stimuli_file}_activations_model_{activ_type}.pkl")
        activations = load_pickle(activations_file)
        activations = activations['model']
        model_rdm = get_network_rdm(activations, model_name, stimuli_file, cons, save = save, activ_type=activ_type)
    return rdms, model_rdm

if __name__ == '__main__':
    parser = get_similarity_parser()
    args = parser.parse_args()
    rdms, model_rdm = extract_network_rdms(args.cons, args.stimuli_file_name, args.model_name, args.save, "trained")
    rdms, model_rdm = extract_network_rdms(args.cons, args.stimuli_file_name, args.model_name, args.save, "untrained")