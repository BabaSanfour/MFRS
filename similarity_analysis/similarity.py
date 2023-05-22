import os
import sys
from typing import Dict
import numpy as np
from tqdm import tqdm
from neurora.rdm_corr import rdm_similarity, rdm_distance, rdm_correlation_kendall, rdm_correlation_spearman, rdm_correlation_pearson
sys.path.append('../../MFRS')
from bootstrapping import eval_bootstrap_pearson
from utils.general import save_pickle, load_pickle, load_meg_rdm, load_model_rdm
from utils.config_sim_analysis import similarity_folder, meg_sensors, meg_rdm, networks, channels_grad1, channels_grad2, channels_mag
from utils.config import get_similarity_parser


def score(chl_rdm: np.array, layer_rdm: np.array, method: list = ["spearman"], permutation: bool = False, iter: int = 1000) -> Dict[str, list]:
    """
    Calculate similarity scores between two RDMs using different methods.

    Parameters:
    -----------
    chl_rdm : np.array
        MEG channel RDM with shape [n_cons, n_cons].
    layer_rdm : np.array
        Layer RDM with shape [n_cons, n_cons].
    method : list, optional
        Methods to compute the correlation between the RDMs. Possible values:
        - "spearman": compute Spearman correlation. (default)
        - "pearson": compute Pearson correlation.
        - "kendall": compute Kendall correlation.
        - "cosine_sim": compute cosine similarity.
        - "euclidean": compute Euclidean distance.
    permutation : bool, optional
        Use permutation tests. Default is False.
    iter : int, optional
        Number of iterations for permutation tests. Default is 1000.

    Returns:
    --------
    sim_dict : dict
        A dictionary with similarity scores of the RDMs. The keys are the method names and the values are the similarity scores.
        For correlation methods ("spearman", "pearson", "kendall"), the values are lists [r, p] where r is the similarity measure
        and p is the p-value. For cosine similarity and Euclidean distance methods, the values are lists [r] where r is the
        similarity measure.

    """

    assert chl_rdm.shape == layer_rdm.shape, "RDM shapes don't match. Invalid input!"

    sim_dict = {}

    if "spearman" in method:
        sim_dict["spearman"] = rdm_correlation_spearman(chl_rdm, layer_rdm, permutation=permutation, iter=iter)
    if "pearson" in method:
        sim_dict["pearson"] = rdm_correlation_pearson(chl_rdm, layer_rdm, permutation=permutation, iter=iter)
    if "kendall" in method:
        sim_dict["kendall"] = rdm_correlation_kendall(chl_rdm, layer_rdm, permutation=permutation, iter=iter)
    if "cosine_sim" in method:
        sim_dict["cosine_sim"] = rdm_similarity(chl_rdm, layer_rdm, permutation=permutation, iter=iter)
    if "euclidean"in method:
        sim_dict["euclidean"] = rdm_distance(chl_rdm, layer_rdm, permutation=permutation, iter=iter)
    return sim_dict


def similarity_score(meg_rdm: np.array, meg_sensors: list, network_rdms: np.array, main_layers: list = [], save: bool = False,
            file_name: str = None, method: list = ["spearman"], permutation: bool = False, iter: int = 1000 ):
    """
    Get a dictionary with a mapping of similarity scores: all channels to given layers.

    Parameters:
    ---------------
    meg_rdm : np.array
        MEG RDMs, shape: [n_chls, n_cons, n_cons].
    meg_sensors : list
        List of MEG sensor names.
    network_rdms : np.array
        Network RDMs, shape: [n_layers, n_cons, n_cons].
    main_layers : list, optional
        List of names of the layers for which to compute similarity. Default is an empty list.
    save : bool, optional
        Save computed similarity scores or not. Default is False.
    file_name : str, optional
        The name of the file in which the computed similarity scores will be saved.
    method : list, optional
        Method to compute the correlation between the RDMs. Default is ["spearman"].
    permutation : bool, optional
        Perform permutation tests. Default is False.
    iter : int, optional
        Number of iterations for permutation tests. Default is 1000.

    Returns:
    ---------------
    sim_dict : dict
        A dictionary with 'layer_name sensor_name' as keys and the similarity measure dictionaries as values.
        Length: n_chls * n_layers
    """
    if network_rdms.shape[0] != len(main_layers):
        raise ValueError("Number of RDMs does not match the number of layer names.")

    sim_dict = {}
    n_chls = meg_rdm.shape[0]
    for i in tqdm(range(n_chls), desc="Computing Similarity Scores"):
        chl_rdm = meg_rdm[i]
        sensor_name = meg_sensors[i]
        for j in range(network_rdms.shape[0]):
            layer_rdm = network_rdms[j]
            layer_name = main_layers[j]
            similarity = score(chl_rdm, layer_rdm, method, permutation, iter)
            sim_dict[f"{layer_name} {sensor_name}"] = similarity
    if save:
        save_pickle(sim_dict, file_name)

    return sim_dict

def network_similarity_scores(meg_rdm: np.array, meg_sensors: list, network_rdm: np.array, save: bool = False,
                              file_name: str = None, method: list = ["spearman"], permutation: bool = False,
                              iter: int = 1000):
    """
    Calculate similarity scores between MEG RDMs and a single network RDM.

    Parameters:
    -----------
    meg_rdm : np.array
        MEG RDMs with shape [n_chls, n_cons, n_cons].
    meg_sensors : list
        List of MEG sensor names.
    network_rdm : np.array
        Network RDM with shape [n_cons, n_cons].
    save : bool, optional
        Save computed similarity scores or not. Default is False.
    file_name : str, optional
        The name of the file in which the computed similarity scores will be saved.
    method : list, optional
        Method to compute the correlation between the RDMs. Default is ["spearman"].
    permutation : bool, optional
        Perform permutation tests or not. Default is False.
    iter : int, optional
        Number of iterations for permutation tests. Default is 1000.

    Returns:
    --------
    sim_dict : dict
        A dictionary with similarity scores between MEG RDMs and the network RDM.
        The keys are in the format "model sensor_name", and the values are the similarity measure dictionaries.
        The similarity measure dictionaries have keys "r" and "p" (if permutation=True) or only "r" (if permutation=False).
    """
    return similarity_score(meg_rdm, meg_sensors, np.reshape(network_rdm, (1, network_rdm.shape[0], network_rdm.shape[1])),
                     ["model"], save, file_name, method, permutation, iter)
    

def get_highest_similarity(sim_dict, method="spearman"):
    """
    Get the highest similarity measure and corresponding sensor for each layer.

    Parameters:
    ----------
    sim_dict : dict
        A dictionary with 'layer_name sensor_name' as keys and the similarity measure dictionaries as values.
        The values are of the form [r, p] where r is the similarity measure and p is the p-value, or just [r].

    Returns:
    -------
    highest_similarity : dict
        A dictionary where each key is a layer name and the value is a tuple containing the highest similarity measure (r)
        and the corresponding sensor name.

    """
    highest_similarity = {}
    for key, value in sim_dict.items():
        layer_name, sensor_name = key.split(" ")
        similarity_measure = value[method][0]  # Assuming the first element in the list is the similarity measure (r)
        
        # Check if layer_name exists in highest_similarity dictionary
        if layer_name not in highest_similarity:
            highest_similarity[layer_name] = {
                "grad1": (-float('inf'), None),
                "grad2": (-float('inf'), None),
                "mag": (-float('inf'), None)
            }
        
        # Determine the sensor type based on its name
        if sensor_name in channels_grad1:
            sensor_type = "grad1"
        elif sensor_name in channels_grad2:
            sensor_type = "grad2"
        elif sensor_name in channels_mag:
            sensor_type = "mag"
        else:
            sensor_type = None
        
        # Update highest similarity for the corresponding sensor type
        if sensor_type and similarity_measure > highest_similarity[layer_name][sensor_type][0]:
            highest_similarity[layer_name][sensor_type] = (similarity_measure, sensor_name)
    
    return highest_similarity

def perumtation_tests(highest_similarity: dict, meg_rdm: np.array, layers_rdms: np.array, method: list = ["spearman"], iter=1000):
    """
    Perform permutation tests to compute similarity measures for specific layer and sensor type.

    Parameters:
    ----------
    highest_similarity : dict
        A dictionary containing the highest similarity measure and corresponding sensor for each layer and sensor type.
    average_rdm : np.array
        A NumPy array representing the average RDM (Representational Dissimilarity Matrix) for all sensors.
    layers_rdms : np.array
        A NumPy array containing the RDMs for each layer.
    method : list, optional
        A list of similarity measures to compute, e.g., ["spearman"] (default: ["spearman"]).
    iter : int, optional
        The number of iterations for the permutation test (default: 1000).

    Returns:
    -------
    permutation_tests : dict
        A dictionary where each key is a layer name, and the value is a dictionary containing the similarity measure and p-value (r, p)
        and the corresponding sensor name for each sensor type after the permutation test.

    """
    permutation_tests = {}
    for layer_index, (layer_name, sensors_type) in enumerate(highest_similarity.items()):
        permutation_tests[layer_name] = {
                "grad1": (-float('inf'), None),
                "grad2": (-float('inf'), None),
                "mag": (-float('inf'), None)
            }
        for sensor_type, results in sensors_type.items():
            sensor_name = results[1]
            sensor_meg_idx=meg_sensors.index(sensor_name)
            similarity_measure = score(meg_rdm[sensor_meg_idx], layers_rdms[layer_index], method, permutation=True, iter=iter)
            permutation_tests[layer_name][sensor_type] = (similarity_measure, sensor_name)
    return permutation_tests

def subjects_similarity_score( meg_rdms: np.array, meg_sensors: list, layers_rdms: np.array, network_rdm: np.array, main_layers: list = [], save_all: bool = False,
            save_subject: bool = False, model_name: str = None, stimuli_file: str = None, method: list = ["spearman"], activ_type: str = "trained", permutation: bool = False, iter: int = 1000 ):
    """
    Get a dictionary with subjects mappings of similarity scores.

    Parameters:
    ---------------
    meg_rdms : np.array
        MEG RDMs, shape: [n_subjects, n_chls, n_cons, n_cons].
    meg_sensors : list
        List of MEG sensor names.
    layers_rdms : np.array
        layers RDMs, shape: [n_layers, n_cons, n_cons].
    network_rdm : np.array
        network RDM, shape: [n_cons, n_cons].
    main_layers : list, optional
        List of names of the model layers. Default is an empty list.
    save_all : bool, optional
        Save computed similarity scores for all subjects in one file. Default is False.
    save_subject : bool, optional
        Save computed similarity scores for each subject in different files. Default is False.
    model_name : str, optional
        Network name, required if save_subject is True.
    stimuli_file : str, optional
        Data name, required if save_subject is True.
    method : list, optional
        Method to compute the correlation between the RDMs. Default is ["spearman"].
    activ_type : str, optional
        Activation type. Default is "trained".
    permutation : bool, optional
        Perform permutation tests. Default is False.
    iter : int, optional
        Number of iterations for permutation tests. Default is 1000.

    Returns:
    --------
    subjects_sim_dict : dict
        A dictionary with subject IDs as keys and mappings dictionary of similarity scores between layers and channels as values.
    subjts_high_sim_file : str
        File path of the saved high similarity scores.
    subjects_model_sim_dict : dict
        A dictionary with subject IDs as keys and mappings dictionary of model similarity scores as values.
    subjects_model_high_sim_dict : dict
        A dictionary with subject IDs as keys and mappings dictionary of high model similarity scores as values.
    """
    if save_subject:
        assert model_name is not None, "model_name not specified. Invalid input!"
        assert stimuli_file is not None, "data_name not specified. Invalid input!"

    subjts_sim_file = os.path.join(similarity_folder, f"{model_name}_{stimuli_file}_main_sim_scores_{activ_type}_all_subjects.pkl")
    subjts_high_sim_file = os.path.join(similarity_folder, f"{model_name}_{stimuli_file}_main_high_sim_scores_{activ_type}_all_subjects.pkl")
    subjts_model_sim_file = os.path.join(similarity_folder, f"{model_name}_{stimuli_file}_model_sim_scores_{activ_type}_all_subjects.pkl")
    subjts_model_high_sim_file = os.path.join(similarity_folder, f"{model_name}_{stimuli_file}_model_high_sim_scores_{activ_type}_all_subjects.pkl")

    if os.path.isfile(subjts_sim_file):
        print(f"Loading pre-computed Sim scores for main layers, model {model_name}, {stimuli_file}, {activ_type}...")
        subjects_sim_dict = load_pickle(subjts_sim_file)
    if os.path.isfile(subjts_high_sim_file):
        print(f"Loading pre-computed Highest Sim scores for main layers, model {model_name}, {stimuli_file}, {activ_type}...")
        subjects_high_sim_dict = load_pickle(subjts_high_sim_file)
    if os.path.isfile(subjts_model_sim_file):
        print(f"Loading pre-computed Sim scores for model {model_name}, {stimuli_file}, {activ_type}...")
        subjects_model_sim_dict = load_pickle(subjts_model_sim_file)
    if os.path.isfile(subjts_model_high_sim_file):
        print(f"Loading pre-computed Highest Sim scores for model {model_name}, {stimuli_file}, {activ_type}...")
        subjects_model_high_sim_dict = load_pickle(subjts_model_high_sim_file)


    subjects_sim_dict = {}
    subjects_high_sim_dict = {}
    subjects_model_sim_dict = {}
    subjects_model_high_sim_dict = {}
    for subject_id in tqdm(range(meg_rdms.shape[0]), desc="Computing Subject Similarity Scores"):
        subj_sim_file = os.path.join(similarity_folder, f"{model_name}_{stimuli_file}_main_sim_scores_{activ_type}_subject_{subject_id + 1:02d}.pkl")
        meg_rdm = meg_rdms[subject_id]
        if os.path.isfile(subj_sim_file):
            subject_sim_scores = load_pickle(subj_sim_file)
            subjects_sim_dict[f"subject {subject_id + 1:02d}"] = subject_sim_scores
        else:
            subject_sim_scores = similarity_score(
                meg_rdm, meg_sensors, layers_rdms, main_layers, save_subject, subj_sim_file, method, permutation, iter
            )
            subjects_sim_dict[f"subject {subject_id + 1:02d}"] = subject_sim_scores

        subj_highest_sim_file = os.path.join(similarity_folder, f"{model_name}_{stimuli_file}_main_high_sim_scores_{activ_type}_subject_{subject_id + 1:02d}.pkl")
        if os.path.isfile(subj_highest_sim_file):
            subject_high_sim_scores = load_pickle(subj_highest_sim_file)
            subjects_high_sim_dict[f"subject {subject_id + 1:02d}"] = subject_high_sim_scores
        else:
            subject_high_sim_scores = get_highest_similarity(subject_sim_scores, method[0])
            subjects_high_sim_dict[f"subject {subject_id + 1:02d}"] = subject_high_sim_scores

        subj_model_sim_file = os.path.join(similarity_folder, f"{model_name}_{stimuli_file}_model_sim_scores_{activ_type}_subject_{subject_id + 1:02d}.pkl")
        if os.path.isfile(subj_model_sim_file):
            subject_model_sim_scores = load_pickle(subj_model_sim_file)
            subjects_model_sim_dict[f"subject {subject_id + 1:02d}"] = subject_model_sim_scores
        else:
            subject_model_sim_scores = network_similarity_scores(
                meg_rdm, meg_sensors, network_rdm, save_subject, subj_sim_file, method, permutation, iter
            )
            subjects_model_sim_dict[f"subject {subject_id + 1:02d}"] = subject_model_sim_scores

        subj_model_highest_sim_file = os.path.join(similarity_folder, f"{model_name}_{stimuli_file}_model_high_sim_scores_{activ_type}_subject_{subject_id + 1:02d}.pkl")
        if os.path.isfile(subj_model_highest_sim_file):
            subject_model_high_sim_scores = load_pickle(subj_model_highest_sim_file)
            subjects_model_high_sim_dict[f"subject {subject_id + 1:02d}"] = subject_model_high_sim_scores
        else:
            subject_model_high_sim_scores = get_highest_similarity(subject_model_sim_scores, method[0])
            subjects_model_high_sim_dict[f"subject {subject_id + 1:02d}"] = subject_model_high_sim_scores


    if save_all:
        save_pickle(subjects_sim_dict, subjts_sim_file)
        save_pickle(subjects_high_sim_dict, subjts_high_sim_file)
        save_pickle(subjects_model_sim_dict, subjts_model_sim_file)
        save_pickle(subjects_model_high_sim_dict, subjts_model_high_sim_file)
    return subjects_sim_dict, subjts_high_sim_file, subjects_model_sim_dict, subjects_model_high_sim_dict

        
def average_similarity_score(average_rdm: np.array, meg_sensors: list, layers_rdms: np.array, network_rdm: np.array, main_layers: list = [],
                             save: bool = True, model_name: str = None, stimuli_file: str = None, band: str = None, method: list = ["spearman"],
                             activ_type: str = "trained", permutation: bool = False, iter: int = 1000):
    """
    Get a dictionary with similarity scores for an average RDM representing the average of multiple subjects' RDMs.

    Parameters:
    ---------------
    average_rdm : np.array
        Average RDM computed from multiple subjects' RDMs, shape: [n_chls, n_cons, n_cons].
    meg_sensors : list
        List of MEG sensor names.
    layers_rdms : np.array
        layers RDMs, shape: [n_layers, n_cons, n_cons].
    network_rdm : np.array
        network RDM, shape: [n_cons, n_cons].
    main_layers : list, optional
        List of names of the model layers. Default is an empty list.
    save : bool, optional
        Save computed similarity scores. Default is True.
    model_name : str, optional
        Network name, required if save is True.
    Stimuli_file : str, optional
        Data name, required if save is True.
    band : str, optional
        power band name, required if save is True.
    method : list, optional
        Method to compute the correlation between the RDMs. Default is ["spearman"].
    activ_type : str, optional
        Activation type. Default is "trained".
    permutation : bool, optional
        Perform permutation tests. Default is False.
    iter : int, optional
        Number of iterations for permutation tests. Default is 1000.

    Returns:
    --------
    subjects_sim_dict : dict
        A dictionary with subject IDs as keys and mappings dictionary of similarity scores between layers and channels as values.
    subjts_high_sim_file : str
        File path of the saved high similarity scores.
    subjects_model_sim_dict : dict
        A dictionary with subject IDs as keys and mappings dictionary of model similarity scores as values.
    subjects_model_high_sim_dict : dict
        A dictionary with subject IDs as keys and mappings dictionary of high model similarity scores as values.
    """
    if save:
        assert model_name is not None, "model_name not specified. Invalid input!"
        assert stimuli_file is not None, "data_name not specified. Invalid input!"
    if band != None:
        stimuli_file = f"{stimuli_file}_{band}"
    avg_sim_file = os.path.join(similarity_folder, f"{model_name}_{stimuli_file}_main_sim_scores_{activ_type}_avg.pkl")
    if os.path.isfile(avg_sim_file):
        print(f"Loading pre-computed Sim scores for model {model_name}, {stimuli_file}, {activ_type}...")
        avg_sim_scores = load_pickle(avg_sim_file)
    else:
        avg_sim_scores = similarity_score(
            average_rdm, meg_sensors, layers_rdms, main_layers, save, avg_sim_file, method, permutation, iter
        )

    avg_high_sim_file = os.path.join(similarity_folder, f"{model_name}_{stimuli_file}_main_high_sim_scores_{activ_type}_avg.pkl")
    if os.path.isfile(avg_high_sim_file):
        print(f"Loading pre-computed Sim scores for model {model_name}, {stimuli_file}, {activ_type}...")
        avg_high_sim_scores = load_pickle(avg_high_sim_file)
    else:
        avg_high_sim_scores = get_highest_similarity(avg_sim_scores, method[0])
        if save:
            save_pickle(avg_high_sim_scores, avg_high_sim_file)

    avg_perm_tests_file = os.path.join(similarity_folder, f"{model_name}_{stimuli_file}_main_perm_tests_scores_{activ_type}_avg.pkl")
    if os.path.isfile(avg_perm_tests_file):
        print(f"Loading pre-computed perm tests for model {model_name}, {stimuli_file}, {activ_type}...")
        avg_perm_tests_scores = load_pickle(avg_perm_tests_file)
    else:
        avg_perm_tests_scores = perumtation_tests(avg_high_sim_scores, average_rdm, layers_rdms, method[0])
        if save:
            save_pickle(avg_perm_tests_scores, avg_perm_tests_file)

    avg_bootstrap_file = os.path.join(similarity_folder, f"{model_name}_{stimuli_file}_main_bootstrap_scores_{activ_type}_avg.pkl")
    if os.path.isfile(avg_bootstrap_file):
        print(f"Loading pre-computed Bootsraaps for model {model_name}, {stimuli_file}, {activ_type}...")
        avg_bootstrap_scores = load_pickle(avg_bootstrap_file)
    else:
        avg_bootstrap_scores = eval_bootstrap_pearson(avg_high_sim_scores, average_rdm, layers_rdms)
        if save:
            save_pickle(avg_bootstrap_scores, avg_bootstrap_file)


    avg_model_sim_file = os.path.join(similarity_folder, f"{model_name}_{stimuli_file}_model_sim_scores_{activ_type}_avg.pkl")
    if os.path.isfile(avg_model_sim_file):
        print(f"Loading pre-computed Sim scores for model {model_name}, {stimuli_file}, {activ_type}...")
        avg_model_sim_scores = load_pickle(avg_model_sim_file)
    else:
        avg_model_sim_scores = network_similarity_scores(
            average_rdm, meg_sensors, network_rdm, save, avg_model_sim_file, method, permutation, iter
        )

    avg_model_high_sim_file = os.path.join(similarity_folder, f"{model_name}_{stimuli_file}_model_high_sim_scores_{activ_type}_avg.pkl")
    if os.path.isfile(avg_model_high_sim_file):
        print(f"Loading pre-computed Sim scores for model {model_name}, {stimuli_file}, {activ_type}...")
        avg_model_high_sim_scores = load_pickle(avg_model_high_sim_file)
    else:
        avg_model_high_sim_scores = get_highest_similarity(avg_model_sim_scores, method[0])
        if save:
            save_pickle(avg_model_high_sim_scores, avg_model_high_sim_file)

    avg_model_perm_tests_file = os.path.join(similarity_folder, f"{model_name}_{stimuli_file}_model_perm_tests_scores_{activ_type}_avg.pkl")
    if os.path.isfile(avg_model_perm_tests_file):
        print(f"Loading pre-computed perm tests for model {model_name}, {stimuli_file}, {activ_type}...")
        avg_model_perm_tests_scores = load_pickle(avg_model_perm_tests_file)
    else:
        avg_model_perm_tests_scores = perumtation_tests(avg_model_high_sim_scores, average_rdm, np.reshape(network_rdm, (1, network_rdm.shape[0], network_rdm.shape[1])), method[0])
        if save:
            save_pickle(avg_model_perm_tests_scores, avg_model_perm_tests_file)

    avg_model_bootstrap_file = os.path.join(similarity_folder, f"{model_name}_{stimuli_file}_model_bootstrap_scores_{activ_type}_avg.pkl")
    if os.path.isfile(avg_model_bootstrap_file):
        print(f"Loading pre-computed Bootsraaps for model {model_name}, {stimuli_file}, {activ_type}...")
        avg_model_bootstrap_scores = load_pickle(avg_model_bootstrap_file)
    else:
        avg_model_bootstrap_scores = eval_bootstrap_pearson(avg_model_high_sim_scores, average_rdm, np.reshape(network_rdm, (1, network_rdm.shape[0], network_rdm.shape[1])))
        if save:
            save_pickle(avg_model_bootstrap_scores, avg_model_bootstrap_file)


    return avg_sim_scores, avg_high_sim_scores, avg_model_sim_scores, avg_model_high_sim_scores

if __name__ == '__main__':
    
    parser = get_similarity_parser()
    args = parser.parse_args()
    list_layers = networks[args.model_name]
    meg_rdm = load_meg_rdm(args.stimuli_file_name, args.band)
    meg_rdm = np.mean(meg_rdm, axis=0)
    layers_rdms = load_model_rdm(args.stimuli_file_name, args.model_name, args.activ_type, type = "main")
    network_rdm = load_model_rdm(args.stimuli_file_name, args.model_name, args.activ_type, type = "model")
    avg_sim_scores, avg_high_sim_scores, avg_model_sim_scores, avg_model_high_sim_scores = average_similarity_score(meg_rdm, meg_sensors, 
                layers_rdms, network_rdm, list_layers, args.save, model_name = args.model_name, stimuli_file= args.stimuli_file_name, band = args.band, method = ["pearson"], activ_type = args.activ_type)
