import os
import sys
import pickle
import numpy as np
from neurora.rdm_corr import rdm_similarity, rdm_distance, rdm_correlation_kendall, rdm_correlation_spearman, rdm_correlation_pearson
sys.path.append('/home/hamza97/MFRS/utils')
from general import save_pickle, load_pickle
from config_sim_analysis import similarity_folder

def stats(list):
    return [np.mean(np.array(list)), np.std(np.array(list)), max(list), min(list)]

def get_stats(measures_list):
    """ Get mean, std, max, min of a measure for combination across all subjects """
    if type(measures_list[0])==np.ndarray:
        p, r = [item[0] for item in measures_list], [item[1] for item in measures_list]
        return [stats(p), stats(r)]
    else:
        return stats([item for item in measures_list])

def stats_subjects_similarity_score(subjects_sim_dict: dict, save: bool = False, file_name: str = None):
    """
        Get a dict with stats for each combination of similarity measures (layer x sensor) across subjects

        Parameters:
        ---------------
        subjects_sim_dict   a dict with subject_ids as keys and mappings dict of similarity scores between layers and channels as values.
        save                save computed similarity scores stats, default: False.
        file_name:          The name of the file in which the computed similarity scores stats for over all the subjects will be saved.


        returns:
        ---------------
        subjects_sim_dict   a dict with stats for each combination of similarity measures (layer x sensor) across subjects.
                                keys:     combination names: 'layer_name, sensor_name'
                                values:   dict of stats for each similarity method:
                                            keys:     similarity method name
                                            values:   list,  [mean, std, max, min]

    """
    if save:
        assert model_name != None, "\nmodel_name not specified.\n Invalid input!"
        assert data_name != None, "\ndata_name not specified.\n Invalid input!"

    combinations_stats ={}
    # get combination names
    combinations_names = list(subjects_sim_dict["subject 01"].keys())
    # get similarity methods names
    methods_names = subjects_sim_dict["subject 01"][combinations_names[0]].keys()
    # iterate over combination names
    for name in combinations_names:
        # create an empty dict with empty lists for each method to store all the methods measures for the subjects for each combination\
        measures_across_subjects={}
        for method in methods_names:
            measures_across_subjects[method]=[]
        for i in range(len(subjects_sim_dict)):
            # get similarity measures for each subject for the specific combinations: name
            values=subjects_sim_dict["subject %02d"%(i+1)][name]
            # for each method append the values in its appropriate list in the new dict
            for method in values.keys():
                measures_across_subjects[method].append(values[method])
        # create an empty dict to store the stats for each combination for different similarity methods
        stats_across_subjects={}
        for method in measures_across_subjects.keys():
            # test we have correlation measure (2 values) or cosine/euclidean distance (1 value)
            stats_across_subjects[method] = [get_stats(measures_across_subjects[method])]
            # if len(measures_across_subjects[method])==2:
            #     p, r = get_stats(measures_across_subjects[method]), get_stats(measures_across_subjects[method], 1)
            #     stats_across_subjects[method] = [p, r]
            # else:
            #     p =
            #     stats_across_subjects[method] = [p]

        combinations_stats[name] = stats_across_subjects
    if save:
        save_pickle(combinations_stats, file_name)

    return combinations_stats


def subjects_similarity_score(meg_rdms: np.array, meg_sensors: list, network_rdms: np.array, network_layers: list, save_all: bool = False, file_name: str = None,
                save_subject: bool = False, model_name: str = None, data_name: str = None ):
    """
        Get a dict with subjects mappings of similarity scores

        Parameters:
        ---------------
        meg_rdm             MEG RDMs, size: [n_chls, n_cons, n_cons]
        meg_sensors         List of meg sensors names
        network_rdms        Network RDMs, size: [n_layers, n_cons, n_cons]
        network_layers      List of meg layers names.
        save_all            Save computed similarity scores for all subjects in one file, default: False
        file_name:          The name of the file in which the computed similarity scores for all the subjects will be saved.
        save_subject        Save computed similarity scores for each subjects in different files, default: False
        model_name          Network name, required if save=True. Default: None
        data_name           Data name, required if save=True. Default: None

        returns:
        ---------------
        subjects_sim_dict   a dict with subject_ids as keys and mappings dict of similarity scores between layers and channels as values.
    """
    if save_subject:
        assert model_name != None, "\nmodel_name not specified.\n Invalid input!"
        assert data_name != None, "\ndata_name not specified.\n Invalid input!"

    subjects_sim_dict={}
    for subject_id in range(len(meg_rdms)):
        subj_sim_file=os.path.join(similarity_folder, "%s_subject_%02d_%s_data_sim_scores.pkl"%(model_name, subject_id+1, data_name))
        if os.path.isfile(subj_sim_file):
            subjects_sim_dict["subject %02d"%(subject_id+1)]=load_pickle(subj_sim_file)
        else:
            meg_rdm=meg_rdms[subject_id]
            subjects_sim_dict["subject %02d"%(subject_id+1)]= similarity_score(meg_rdm, meg_sensors,
                    network_rdms, network_layers, save_subject, subj_sim_file)
    if save_all:
        save_pickle(subjects_sim_dict, file_name)
    return subjects_sim_dict


def similarity_score(meg_rdm: np.array, meg_sensors: list, network_rdms: np.array, network_layers: list, save: bool= False, file_name: str = None):
    """
        Get a dict with a mapping of similarity scores: all channels to all layers

        Parameters:
        ---------------
        meg_rdm         MEG RDMs, size: [n_chls, n_cons, n_cons]
        meg_sensors     List of meg sensors names
        network_rdms    Network RDMs, size: [n_layers, n_cons, n_cons]
        network_layers  List of meg layers names.
        save            Save computed similarity scores or no, default: False.
        file_name:      The name of the file in which the computed similarity scores will be saved.

        returns:
        ---------------
        sim_dict        a dict with 'layer_name, sensor_name' as keys and the similarity measure dicts as values.
                            length: n_chls * n_layers
    """

    # To do : all channels vs a single layer
    sim_dict={}
    for i in range(len(meg_rdm)):
        chl_rdm = meg_rdm[i]
        sensor_name = meg_sensors[i]
        for j in range(len(network_rdms)):
            layer_rdm = network_rdms[j]
            layer_name = network_layers[j]
            similarity = score(chl_rdm, layer_rdm)
            sim_dict["%s %s"%(layer_name, sensor_name)]=similarity
    if save:
        save_pickle(sim_dict, file_name)
    return sim_dict


def score(chl_rdm: np.array, layer_rdm: np.array, method: str = "all"):
    """
        Get a dict with similarity scores of two RDMs

        Parameters:
        ---------------
        chl_rdm         MEG channel RDM, size: [n_cons, n_cons]
        layer_rdm       layer RDM, size: [n_cons, n_cons]
        method          method to compute the correlation between the RDMs : expecting:
                            - spearman:     compute spearman correlation
                            - pearson:      compute pearson correlation
                            - kendall:      compute kendall correlation
                            - cosine_sim:   compute cosine similarity
                            - euclidian:    compute euclidian distance
                            - all, default  compute all measures

        returns:
        ---------------
        sim_dict    a dict with [sensor_name, layer_name)] name as key and the measure as value:
                    - correlation method: retrun a dict:  {"method name": [r, p]}, r is similarity measure and p is P-value
                    - cosine and euclidean distances: retrun a dict:  {"method name": [r]}, r is similarity measure
                    - all, returns a dict with all measures.

    """
    assert chl_rdm.shape == layer_rdm.shape, "\nRDM shapes don't match.\n Invalid input!"

    if method == "spearman":
        rsa_spearman=rdm_correlation_spearman(chl_rdm, layer_rdm)
        return {"spearman": rsa_spearman}
    elif method == "pearson":
        rsa_pearson=rdm_correlation_pearson(chl_rdm, layer_rdm)
        return {"pearson": rsa_pearson}
    elif method == "kendall":
        rsa_kendall=rdm_correlation_kendall(chl_rdm, layer_rdm)
        return {"kendall": rsa_kendall}
    elif method == "cosine_sim":
        rsa_sim=rdm_similarity(chl_rdm, layer_rdm)
        return {"cosine_sim": rsa_sim}
    elif method == "euclidian":
        rsa_euc=rdm_distance(chl_rdm, layer_rdm)
        return {"euclidian": rsa_spearman}
    else:
        rsa_spearman=rdm_correlation_spearman(chl_rdm, layer_rdm)
        rsa_pearson=rdm_correlation_pearson(chl_rdm, layer_rdm)
        rsa_kendall=rdm_correlation_kendall(chl_rdm, layer_rdm)
        rsa_sim=rdm_similarity(chl_rdm, layer_rdm)
        rsa_euc=rdm_distance(chl_rdm, layer_rdm)
        return {"spearman": rsa_spearman, "pearson": rsa_pearson,
                "kendall": rsa_kendall, "cosine_sim": rsa_sim, "euclidian": rsa_spearman}

def whole_network_similarity_scores(name: str, meg_rdm: np.array, meg_sensors: list):
    """Get the model similarity results"""
    if os.path.exists(os.path.join(similarity_folder, '%s_sim_model.pkl'%name)):
        sim_dict=load_pickle(os.path.join(similarity_folder, '%s_sim_model.pkl'%name))
        return sim_dict
    else:
        network_rdm=load_npy(os.path.join(rdms_folder, '%s_model_rdm.npy'%name))
        sim_dict={}
        for i in range(len(meg_rdm)):
            chl_rdm = meg_rdm[i]
            sensor_name = meg_sensors[i]
            similarity = score(chl_rdm, network_rdm)
            sim_dict["%s %s"%(layer_name, sensor_name)]=similarity
        if save:
            save_pickle(sim_dict, os.path.join(similarity_folder, '%s_sim_model.pkl'%name))
        return sim_dict
