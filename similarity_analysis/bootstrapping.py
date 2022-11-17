import copy
import time
import numpy as np
from scipy.stats import pearsonr

import rsatoolbox.rdm as rsrdm
from rsatoolbox.inference import bootstrap_sample_pattern

def get_rdms_vectors(
        rdms: np.array,
        dissimilarity_measure: str ="correlation"
    ):
    """
        Read RDMs, transform them into an RDM object from rsatoolbox (1-D vector with only upper part of the matrice).

        Parameters:
        ---------------
        rdms                      Model or MEG rdms, size: [N_layers/N_sensors, n_cons, n_cons]
        dissimilarity_measure     Method to compute the correlation between the RDMs : expecting: correlation

        returns:
        ---------------
        rdms                      an RDM object from rsatoolbox
    """

    index = np.array([x for x in range(rdms.shape[0])])
    rdm_descriptors = {'session': [1 for i in range(rdms.shape[0])],
                   'subj': index,
                   'index' : index}
    rdms= rsrdm.RDMs( dissimilarities=rdms,
                     dissimilarity_measure=dissimilarity_measure,
                     rdm_descriptors=rdm_descriptors,
    #                  pattern_descriptors=pattern_descriptors
                  )
    return rdms

def eval_bootstrap_pearson(
            RDM1,
            RDM2,
            network_layers: list,
            model_layers: list,
            N_bootstrap: int = 100,
        ):

    """
        Performs bootstrapping over MEG brain RDMs.

        Parameters:
        ---------------
        RDM1                 MEG rdms, size: [N_sensors, n_cons, n_cons]
        RDM2                 Model rdms, size: [n_layers, n_cons, n_cons]
        network_layers       All of network layers
        model_layers         Selected layers
        N_bootstrap          Number of Bootraps

        returns:
        ---------------
        sensor_r_list        spearman coorelation: Shape [N_layers, N_sensors, N_bootstrap]
    """

    sensor_r_list=[]
    for boot in range(N_bootstrap):
        sensors_rdm, rdm_idx=bootstrap_sample_pattern(RDM1)
        nan_idx = ~np.isnan(sensors_rdm.dissimilarities[0])
        # print(nan_idx.shape)
        r_list=[[] for i in range(len(model_layers))]
        RDM2_copy=copy.deepcopy(RDM2)
        RDM2=RDM2.subsample_pattern("index", rdm_idx)

        for i in range(RDM1.dissimilarities.shape[0]):
            sensor_rdm = sensors_rdm.dissimilarities[i][nan_idx]
            k=0
            test = len(network_layers) == RDM2.dissimilarities.shape[0]
            for j in range(RDM2.dissimilarities.shape[0]):
                if test:
                    if network_layers[j] not in model_layers:
                        continue
                    layer_rdm=RDM2.dissimilarities[j][nan_idx]
                    r_list[k].append(np.array(pearsonr(sensor_rdm, layer_rdm))[0])
                    k+=1
                else:
                    layer_rdm=RDM2.dissimilarities[j][nan_idx]
                    r_list[k].append(np.array(pearsonr(sensor_rdm, layer_rdm))[0])
                    k+=1

        sensor_r_list.append(r_list)
        RDM2=copy.deepcopy(RDM2_copy)

    return np.transpose(np.array(sensor_r_list), (1, 2, 0))
