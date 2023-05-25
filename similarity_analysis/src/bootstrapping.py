import copy
from tqdm import tqdm
import numpy as np
from scipy.stats import pearsonr
from utils.config_sim_analysis import meg_sensors
from rsatoolbox.inference import bootstrap_sample_pattern
import rsatoolbox.rdm as rsrdm

def get_rdms_vectors(rdms: np.array, dissimilarity_measure: str = "correlation"):
    """
    Transform RDMs into an RDM object from rsatoolbox.

    Parameters:
    -----------
    rdms : np.array
        Model or MEG RDMs with shape [N_layers/N_sensors, n_cons, n_cons].
    dissimilarity_measure : str, optional
        Method to compute the dissimilarity between the RDMs. Default is "correlation".

    Returns:
    --------
    rdms_obj : rsrdm.RDMs
        RDM object from rsatoolbox.

    """

    index = np.arange(rdms.shape[0])
    rdm_descriptors = {
        'session': [1 for _ in range(rdms.shape[0])],
        'subj': index,
        'index': index
    }

    rdms_obj = rsrdm.RDMs(
        dissimilarities=rdms,
        dissimilarity_measure=dissimilarity_measure,
        rdm_descriptors=rdm_descriptors,
    )

    return rdms_obj

def eval_bootstrap_pearson(highest_similarity: dict, meg_rdms, model_rdms, N_bootstrap: int = 100):
    """
    Perform bootstrapping over MEG brain RDMs and compute Pearson correlation.

    Parameters:
    -----------
    highest_similarity : dict
        Dictionary with layer names as keys and sensor information as values.
    meg_rdms : RDM obj
        MEG RDMs, size: [N_sensors, n_cons, n_cons].
    model_rdms : RDM obj
        Model RDMs, size: [n_layers, n_cons, n_cons].
    N_bootstrap : int, optional
        Number of bootstrap iterations. Default is 100.

    Returns:
    --------
    bootstrapped_similarity : dict
        Dictionary containing bootstrapped similarity scores for each layer and sensor type.

    """
    bootsrapped_similarity= {}
    meg_rdms=get_rdms_vectors(meg_rdms)
    model_rdms=get_rdms_vectors(model_rdms)
    for _ in tqdm(range(N_bootstrap), desc="Bootstrapping"):
        sensors_rdm, rdm_idx=bootstrap_sample_pattern(meg_rdms)
        nan_idx = ~np.isnan(sensors_rdm.dissimilarities[0])
        model_rdms_copy=copy.deepcopy(model_rdms)
        model_rdms=model_rdms.subsample_pattern("index", rdm_idx)
        for layer_index, (layer_name, sensors_type) in enumerate(highest_similarity.items()):
            if layer_name not in bootsrapped_similarity:
                bootsrapped_similarity[layer_name] = {
                        "grad1": [],
                        "grad2": [],
                        "mag": []
                    }
            layer_rdm=model_rdms.dissimilarities[layer_index][nan_idx]
            for sensor_type, results in sensors_type.items():
                sensor_name = results[1]
                sensor_meg_idx=meg_sensors.index(sensor_name)
                sensor_rdm = sensors_rdm.dissimilarities[sensor_meg_idx][nan_idx]
                boot_sim = np.array(pearsonr(sensor_rdm, layer_rdm))[0]
                bootsrapped_similarity[layer_name][sensor_type].append(boot_sim)

        model_rdms=copy.deepcopy(model_rdms_copy)

    return bootsrapped_similarity