import os
import sys
import time
import numpy as np

from config_sim_analysis import facenet_layers, cornet_s_layers, activations_folder, rdms_folder, weights_path, similarity_folder, meg_rdm, meg_sensors
from similarity import similarity_score, get_main_network_similarity_scores
from plot_utils import match_layers_sensor, get_network_layers_info
from similarity import similarity_score
sys.path.append('/home/hamza97/MFRS/')
from models.cornet_s import cornet_s
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
    correlation_measure="pearson"
    networks_list = {"FaceNet": [FaceNet, "FaceNet_0.01LR_32Batch_1000_30_final", facenet_layers],
                    "cornet_s": [cornet_s, "cornet_s_0.01LR_32Batch_1000_final", cornet_s_layers]
    }
    for model_name, model_param in networks_list.items():
        model = model_param[0](False, 1000, 1)
        weights = os.path.join(weights_path, model_param[1])
        for stimuli_file_name, cons in stimuli_hdf5_list.items():
            sim_dict_file = os.path.join(similarity_folder, "%s_%s_data_sim_scores_avg.pkl"%(model_name, stimuli_file_name))
            if os.path.isfile(sim_dict_file):
                print("Sim file (data: %s) for %s already exists!!!"%(stimuli_file_name, model_name))
            else:
                rdms_file = os.path.join(rdms_folder, "%s_%s_data_rdm.npy"%(model_name, stimuli_file_name))
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

                network_layers = [item[0] for item in list(model.named_modules())]
                subjects_sim_dict = similarity_score(meg_rdm, meg_sensors, network_rdms, network_layers, model_param[2], save = save,
                                file_name=sim_dict_file, method=correlation_measure)
    time_sim = time.time() - start
    print('Computations ended in %s h %s m %s s' % (time_sim // 3600, (time_sim % 3600) // 60, time_sim % 60))