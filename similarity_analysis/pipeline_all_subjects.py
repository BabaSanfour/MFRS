import os
import sys
import time
import numpy as np

from config_sim_analysis import facenet_layers, activations_folder, rdms_folder, weights_path, similarity_folder, meg_rdm, meg_sensors
from extract_activations import model_activations
from networks_rdm import groupRDMs
from similarity import subjects_similarity_score, stats_subjects_similarity_score
sys.path.append('/home/hamza97/MFRS/')
from models.FaceNet import FaceNet
from utils.load_data import Stimuliloader
from utils.general import load_pickle, load_npy

if __name__ == '__main__':
    start = time.time()

    save = True
    # stimuli_hdf5_list = {"Fam": 150, "Unfam": 150, "Scram": 150,
    #         "FamUnfam": 300, "FamScram": 300, "UnfamScram": 300,
    #         "FamUnfamScram1": 300, "FamUnfamScram2": 300, "FamUnfamScram0": 450}
    stimuli_hdf5_list = {"FamUnfam": 300}

    networks_list = {"FaceNet": [FaceNet, "FaceNet_0.01LR_32Batch_1000_30_final"]}
    for model_name, model_param in networks_list.items():
        model = model_param[0](False, 1000, 1)
        weights = os.path.join(weights_path, model_param[1])
        for stimuli_file_name, cons in stimuli_hdf5_list.items():
            combinations_stats_file = os.path.join(similarity_folder, "%s_%s_data_stats.pkl"%(model_name, stimuli_file_name))
            if os.path.isfile(combinations_stats_file):
                print("combinations_stats file (data: %s) for %s already exists!!!"%(stimuli_file_name, model_name))
                combinations_stats = load_pickle(combinations_stats_file)
            else:
                rdms_file = os.path.join(rdms_dir, "%s_%s_data_rdm.npy"%(model_name, stimuli_file_name))
                if os.path.isfile(rdms_file):
                    print("RDMs file (data: %s) for %s already exists!!!"%(stimuli_file_name, model_name))
                    network_rdms = load_npy(rdms_file)
                else:
                    images=Stimuliloader(cons, stimuli_file_name)
                    images = next(iter(images))
                    activations_file = os.path.join(activations_folder, "%s_%s_activations.pkl"%(stimuli_file_name, model_name))
                    if os.path.isfile(activations_file):
                        print("activations file (data: %s) for %s already exists!!!"%(stimuli_file_name, model_name))
                        activations = load_pickle(activations_file)
                    else:
                        activations=model_activations(model, images, weights, method= "list", list_layers = facenet_layers, save = save, file_name = activations_file)

                    network_rdms=groupRDMs(activations, cons, save = save, file_name = rdms_file)
                subjects_sim_dict_file = os.path.join(similarity_folder, "%s_%s_data_sim_scores.pkl"%(model_name, stimuli_file_name))
                if os.path.isfile(subjects_sim_dict_file):
                    print("subjects sim file (data: %s) for %s already exists!!!"%(stimuli_file_name, model_name))
                    subjects_sim_dict = load_pickle(subjects_sim_dict_file)
                else:
                    out_file = os.path.join(meg_dir, "RDMs_16-subject_0-sub_opt0-chl_opt.npy")
                    meg_rdms = load_npy(out_file)
                    network_layers = [item[0] for item in list(model.named_modules())]
                    meg_sensors=get_sensor("meg")
                    subjects_sim_dict = subjects_similarity_score(meg_rdms, meg_sensors, network_rdms, network_layers, facenet_layers, save_all = save,
                                    file_name=subjects_sim_dict_file, save_subject = save, model_name = model_name, data_name = stimuli_file_name)
                combinations_stats=stats_subjects_similarity_score(subjects_sim_dict, save = save, file_name = combinations_stats_file)
    time_sim = time.time() - start
    print('Computations ended in %s h %s m %s s' % (time_sim // 3600, (time_sim % 3600) // 60, time_sim % 60))
