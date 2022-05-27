import os
import pickle
import numpy as np
from neurora.rdm_corr import rdm_similarity, rdm_distance, rdm_correlation_kendall, rdm_correlation_spearman, rdm_correlation_pearson

similarity_folder = '/home/hamza97/scratch/data/MFRS_data/similarity_scores'

def stats(list):
    return [np.mean(np.array(list)), np.std(np.array(list)), max(list), min(list)]

def get_stats(measures_list):
    """ Get mean, std, max, min of a measure for combination across all subjects """
    if type(measures_list[0])==np.ndarray:
        p, r = [item[0] for item in measures_list], [item[1] for item in measures_list]
        return [stats(p), stats(r)]
    else:
        return stats([item for item in measures_list])



def stats_subjects_similarity_score(subjects_sim_dict: dict, save: bool = False, model_name: str = None, data_name: str = None):
    """
        Get a dict with stats for each combination of similarity measures (layer x sensor) across subjects

        Parameters:
        ---------------
        subjects_sim_dict   a dict with subject_ids as keys and mappings dict of similarity scores between layers and channels as values.
        save                save computed similarity scores stats, default: False
        model_name          network name, required if save=True. Default: None
        data_name           data name, required if save=True. Default: None

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
        file=os.path.join(similarity_folder, "%s_%s_data_stats.pkl"%(model_name, data_name))
        save_similarity_score(combinations_stats, file)

    return combinations_stats


def subjects_similarity_score(meg_rdms: np.array, meg_sensors: list, network_rdms: np.array, network_layers: list, save_subjects: bool = False,
                save_subject: bool = False, single_subj: bool = True, model_name: str = None, data_name: str = None ):
    """
        Get a dict with subjects mappings of similarity scores

        Parameters:
        ---------------
        meg_rdm             MEG RDMs, size: [n_chls, n_cons, n_cons]
        meg_sensors         list of meg sensors names
        network_rdms        network RDMs, size: [n_layers, n_cons, n_cons]
        network_layers      list of meg layers names.
        save_subject       save computed similarity scores for all subjects in one file, default: False
        save_subjects       save computed similarity scores for each subjects in different files, default: False
        single_subj         meg rdms are for a single subject (True) or averaed over subjects (False), required if save=True. Default: True
        model_name          network name, required if save=True. Default: None
        data_name           data name, required if save=True. Default: None

        returns:
        ---------------
        subjects_sim_dict   a dict with subject_ids as keys and mappings dict of similarity scores between layers and channels as values.
    """
    if save_subjects:
        assert model_name != None, "\nmodel_name not specified.\n Invalid input!"
        assert data_name != None, "\ndata_name not specified.\n Invalid input!"

    subjects_sim_dict={}
    for subject_id in range(len(meg_rdms)):
        subj_sim_file=os.path.join(similarity_folder, "%s_subject_%02d_%s_data_sim_scores.pkl"%(model_name, subject_id+1, data_name))
        if os.path.isfile(subj_sim_file):
            subjects_sim_dict["subject %02d"%(subject_id+1)]=load_similarity_score(subj_sim_file)
        else:
            meg_rdm=meg_rdms[subject_id]
            subjects_sim_dict["subject %02d"%(subject_id+1)]= similarity_score(meg_rdm, meg_sensors,
                    network_rdms, network_layers, save_subject, single_subj, subject_id+1, model_name, data_name)
    if save_subjects:
        file=os.path.join(similarity_folder, "%s_%s_data_sim_scores.pkl"%(model_name, data_name))
        save_similarity_score(subjects_sim_dict, file)
    return subjects_sim_dict


def similarity_score(meg_rdm: np.array, meg_sensors: list, network_rdms: np.array, network_layers: list, save: bool= False,
                        single_subj: bool = True, subject_id: int = None, model_name: str = None, data_name: str = None):
    """
        Get a dict with a mapping of similarity scores: all channels to all layers

        Parameters:
        ---------------
        meg_rdm         MEG RDMs, size: [n_chls, n_cons, n_cons]
        meg_sensors     list of meg sensors names
        network_rdms    network RDMs, size: [n_layers, n_cons, n_cons]
        network_layers  list of meg layers names.
        save            save computed similarity scores or no, default: False
        single_subj     meg rdms are for a single subject (True) or averaed over subjects (False), required if save=True. Default: True
        model_name      network name, required if save=True. Default: None
        data_name       data name, required if save=True. Default: None
        subject_id      subject id, required if save=True. Default: None

        returns:
        ---------------
        sim_dict        a dict with 'layer_name, sensor_name' as keys and the similarity measure dicts as values.
                            length: n_chls * n_layers
    """

    if save:
        assert model_name != None, "\nmodel_name not specified.\n Invalid input!"
        assert data_name != None, "\ndata_name not specified.\n Invalid input!"
        if single_subj:
            assert subject_id != None, "\nsubject_id not specified.\n Invalid input!"
    # To do : all channels vs a single layer
    sim_dict={}
    for i in range(len(meg_rdm)):
        chl_rdm = meg_rdm[i]
        sensor_name = meg_sensors[i]
        for j in range(len(network_rdms)):
            layer_rdm=network_rdms[j]
            layer_name= network_layers[j]
            similarity = score(chl_rdm, layer_rdm)
            sim_dict["%s %s"%(layer_name, sensor_name)]=similarity
    if save:
        if single_subj:
            file=os.path.join(similarity_folder, "%s_subject_%02d_%s_data_sim_scores.pkl"%(model_name, subject_id, data_name))
        else :
            # if we have brain rdms computed by avreging over subjects
            file=os.path.join(similarity_folder, "%s_avg_rdm_subjects_%s_data_sim_scores.pkl"%(model_name, data_name))
        save_similarity_score(sim_dict, file)
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

def save_similarity_score(sim_dict, file):
    """ Save similarity score in pickle files """
    with open(file, 'wb') as f:
        pickle.dump(sim_dict, f)
    print('File saved successfully')

def load_similarity_score(file):
    """ Load similarity score from pickle files """
    with open(file, 'rb') as f:
        sim_dict = pickle.load(f)
    print('File loaded successfully')
    return sim_dict
