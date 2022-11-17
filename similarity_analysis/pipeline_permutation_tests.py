import os
import sys
import time
import numpy as np

from config_sim_analysis import meg_rdm, meg_sensors, networks, rdms_folder, similarity_folder, channels_grad1, channels_grad2, channels_mag
from similarity import score, get_main_network_similarity_scores
from plot_utils import match_layers_sensor, get_network_layers_info

sys.path.append('/home/hamza97/MFRS/')
from utils.general import save_pickle, load_pickle, load_npy

from models.FaceNet import FaceNet
from models.inception import inception_v3
from models.mobilenet import mobilenet_v2
from models.SphereFace import SphereFace
from models.resnet import resnet50
from models.cornet_s import cornet_s
from models.vgg import vgg16_bn

def get_index(main_similarity_scores, network_layers, model_layers, channel_list, correlation_measure="pearson"):
    # Get correlations_list, sensors_list
    correlations_list, sensors_list = match_layers_sensor(main_similarity_scores, network_layers, channel_list, correlation_measure)
    # Get correlations_list, sensors_list
    idx, best_layer, sim_chls, extremum, mask = get_network_layers_info(correlations_list,
                                                                   network_layers, main_similarity_scores, sensors_list, channel_list)
    #get index of layer in the whole network layers
    non_clean_idx=model_layers.index(best_layer)
    sensor_channel_idx=list(mask).index(True)
    sensor_meg_idx=meg_sensors.index(channel_list[sensor_channel_idx])
    return non_clean_idx, sensor_meg_idx


if __name__ == '__main__':
    start = time.time()

    results={}
    models= {"inception_v3": inception_v3, "mobilenet": mobilenet_v2, "SphereFace": SphereFace, "resnet50": resnet50, "cornet_s": cornet_s, "FaceNet": FaceNet, "vgg16_bn": vgg16_bn, }

    for i, network in enumerate(networks.keys()):
        # Get clean network layers
        network_layers=networks[network]
        # Get non-clean network layers
        model = models[network](False, 1000, 1)
        model_layers = [item[0] for item in list(model.named_modules())]
        rdms_file = os.path.join(rdms_folder, "%s_FamUnfam_data_rdm.npy"%(network))
        model_rdm=load_npy(rdms_file)
        if model_rdm.shape[0] == len(network_layers):
            model_layers=network_layers
        # Get main similarity scores
        main_similarity_scores=get_main_network_similarity_scores('%s_FamUnfam_data_sim_scores'%network, network_layers)
        whole_model_similarity_scores=load_pickle(os.path.join(similarity_folder, "%s_FamUnfam_sim_model.pkl"%(network)))
        for name, channel_list in {"grad1": channels_grad1, "grad2": channels_grad2, "mag": channels_mag}.items():
            #get the indexs of the layer and its associate sensor that gave the best sim score
            layer_rdm_idx, sensor_rdm_idx= get_index(main_similarity_scores, network_layers, model_layers, channel_list)
            #get model layer rdms
            layer_rdm= model_rdm[layer_rdm_idx]
            #get MEG sensor rdm
            sensor_rdm= meg_rdm[sensor_rdm_idx]
            pr=score(sensor_rdm, layer_rdm, permutation=True, iter=1000)
            results[network+" "+model_layers[layer_rdm_idx]+" "+meg_sensors[sensor_rdm_idx]+" "+name]=pr
            # Do the same for the whole model rdm
            model_rdm_idx, sensor_rdm_idx= get_index(whole_model_similarity_scores, [network], [network], channel_list)
            #get model layer rdms
            # always the same, one rdm per model
            #get MEG sensor rdm
            sensor_rdm= meg_rdm[sensor_rdm_idx]
            pr=score(sensor_rdm, model_rdm, permutation=True, iter=1000)
            results[network+" "+meg_sensors[sensor_rdm_idx]+" "+name]=pr
    save_pickle(results, "FamUnfam_permutation_tests.pkl")
    time_sim = time.time() - start
    print('Computations ended in %s h %s m %s s' % (time_sim // 3600, (time_sim % 3600) // 60, time_sim % 60))
