"""
===============================
Utils file for plotting results
===============================
Util functions for selecting the results that we will plot.
"""
import os
import sys
import numpy as np
from scipy.stats import sem

from config_sim_analysis import channels_mag, channels_grad1, channels_grad2, meg_rdm, meg_sensors, similarity_folder
from similarity import whole_network_similarity_scores
sys.path.append('/home/hamza97/MFRS/')
from utils.general import load_npy

def get_filtered_measures(sim_dict: dict, layer: str, channels_list: list, measure: str = "pearson", epsilon: int = 0.05):
    """
    Extracts values from the given dictionary based on the specified layer-channel conditions.
    
    Args:
    - sim_dict (dict): Dictionary with keys as "layer_name channel_name" and values as {"measure": [r, p]}.
    - layer (str): Layer name to extract values for.
    - channels_list (list): List of channel names to iterate over.
    - measure (str): Name of the measure to extract from the dictionary. Default is "pearson".
    - epsilon (float): Threshold value for the condition p > epsilon. Default is 0.05.
    
    Returns:
    - filtered_values (list): List of values from the "measure" key that satisfy the condition p > epsilon,
                             with 0 added for values that don't meet the condition.
    - extremum (float): Maximum absolute value among the filtered values.
    """
    filtered_values = []
    
    for channel_name in channels_list:
        key = f"{layer} {channel_name}"
        value = sim_dict.get(key, {}).get(measure, [])
        if value[1] > epsilon:
            filtered_values.append(value[0])
        else:
            filtered_values.append(0)
    
    extremum = max(max(filtered_values), abs(min(filtered_values)))
    return filtered_values, extremum


def get_layers_similarity(sim_dict: dict, list_layers: list, correlation_measure: str = "spearman"):
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

def match_layers_sensor(sim_dict: dict, list_layers: list, channels_list: list, correlation_measure: str = "spearman"):
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
                                    sensors_list: list, channels_list: list, correlation_measure:str = 'spearman'):
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
            sil_chls, extremum = get_chls_similarity(network, sim_dict, channels_list, 'spearman')
            max_sim[i].append(max(sil_chls))
            extremum_3[i]=max(extremum_3[i], extremum)
            model_results.append(sil_chls)
        models[network]=model_results
    return models, extremum_3, max_sim

def get_bootstrap_values(network, sensors_list, percentile: int = 5):
    boot_all=load_npy(os.path.join(similarity_folder, "%s_FamUnfam_bootstrap.npy"%network))
    # higher_percentile, lower_percentile = [], []
    boot_sem
    for layer_idx in range(boot_all.shape[0]):
        sensor_idx=meg_sensors.index(sensors_list[layer_idx]) # get sensor idx that got highest correlation
        # with layer
        boot_sensor=boot_all[layer_idx][sensor_idx] # get layer and sensor bootstraps
        boot = sem(boot_sensor)
        boot_sem.append(boot)
        # higher_percentile.append(np.percentile(boot_sensor, 100-percentile))
        # lower_percentile.append(np.percentile(boot_sensor, percentile))
    # return [lower_percentile, higher_percentile]
    return boot_sem 
