"""
===============================
Utils file for plotting results
===============================
Util functions for selecting the results that we will plot.
"""

import numpy as np
from config_sim_analysis import channels_mag, channels_grad1, channels_grad2, meg_rdm, meg_sensors
from similarity import whole_network_similarity_scores

def get_chls_similarity(layer: str, sim_dict: dict, channels_list: list, correlation_measure: str = "pearson"):
    """Get specific sensor type similarity results + extremum sim for colorbar"""
    sim_chls, extremum = [], 0
    for sensor in channels_list:
        sim_chls.append(sim_dict[layer + ' ' + sensor][correlation_measure][0])
    extremum=max(max(sim_chls), abs(min(sim_chls)))
    return sim_chls, extremum

def get_layers_similarity(sim_dict: dict, list_layers: list, correlation_measure: str = "pearson"):
    """Divide each layer sim results into 3 lists corresponding to sensor types + Get extremum values"""
    layer_res, extremum_3 ={}, [0,0,0]
    for layer in list_layers:
        sim_chls_3 = []
        for i, channels_list in enumerate([channels_mag, channels_grad1, channels_grad2]):
            sim_chls, extremum = get_chls_similarity(layer, sim_dict,channels_list, correlation_measure)
            sim_chls_3.append(sim_chls)
            extremum_3[i]=max(extremum_3[i], extremum)
        layer_res[layer]= sim_chls_3
    return layer_res, extremum_3

def match_layers_sensor(sim_dict: dict, list_layers: list, channels_list: list, correlation_measure: str = "pearson"):
    """Get a list of the maximum value of similarity for each layer and the sensor that gave the value"""
    correlations_list, sensors_list= [], []
    for layer in list_layers:
        best_corr = 0
        for sensor in channels_list:
            name = layer + ' ' + sensor
            if sim_dict[name][correlation_measure][0]>best_corr:
                best_corr=sim_dict[name][correlation_measure][0]
                best_sensor=sensor
        correlations_list.append(best_corr)
        sensors_list.append(best_sensor)
    return correlations_list, sensors_list

def get_network_layers_info(correlations_list: list, layers: list, similarity_scores: dict,
                                    sensors_list: list, channels_list: list, correlation_measure:str = 'pearson'):
    """Get a list of the maximum value of similarity for each layer and the sensor that gave the value"""
    idx=correlations_list.index(max(correlations_list))
    best_layer=layers[idx]
    sim_chls, extremum = get_chls_similarity(best_layer, similarity_scores, channels_list, correlation_measure)
    mask=np.array([sensor==sensors_list[idx] for sensor in channels_list])
    return idx, best_layer, sim_chls, extremum, mask

def get_networks_results(networks, meg_rdm=meg_rdm, meg_sensors=meg_sensors):
    """Get a list of each network maximum similarity value and the similarity values needed
                    to plot the 3 topomaps for each network + extremum values for color bar"""
    models, max_sim, extremum_3 = {}, [[], [], []], [0, 0, 0]
    for network, network_layers in networks.items():
        sim_dict=whole_network_similarity_scores(network, 'FamUnfam', meg_rdm, meg_sensors)
        model_results=[]
        for i, channels_list in enumerate([channels_mag, channels_grad1, channels_grad2]):
            sil_chls, extremum = get_chls_similarity(network, sim_dict, channels_list, 'pearson')
            max_sim[i].append(max(sil_chls))
            extremum_3[i]=max(extremum_3[i], extremum)
            model_results.append(sil_chls)
        models[network]=model_results
    return models, extremum_3, max_sim
