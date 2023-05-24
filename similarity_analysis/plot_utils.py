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

from utils.config_sim_analysis import channels_mag, channels_grad1, channels_grad2, meg_rdm, meg_sensors, similarity_folder
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
        if value[1] < epsilon:
            filtered_values.append(value[0])
        else:
            filtered_values.append(0)
    
    extremum = max(max(filtered_values), abs(min(filtered_values)))
    return filtered_values, extremum


def get_layers_similarity(sim_dict, layer_list, correlation_measure="pearson", epsilon=0.05):
    """
    Divide each layer similarity results into 3 lists corresponding to sensor types + get extremum values.
    
    Args:
    - sim_dict (dict): Dictionary with keys as "layer_name channel_name" and values as {"measure": [r, p]}.
    - layer_list (list): List of layer names to iterate over.
    - correlation_measure (str): Name of the correlation measure to use. Default is "spearman".
    - epsilon (float): Threshold value for the condition p > epsilon. Default is 0.05.
    
    Returns:
    - layer_similarities (dict): Dictionary with layer names as keys and corresponding similarity lists for each sensor type as values.
    - extremum_values (list): List containing the maximum absolute value for each sensor type.
    """
    layer_similarities = {}
    extremum_values = [0, 0, 0]
    
    for layer in layer_list:
        sensor_type_similarities = []
        
        for i, channels_list in enumerate([channels_mag, channels_grad1, channels_grad2]):
            filtered_values, extremum = get_filtered_measures(sim_dict, layer, channels_list, measure=correlation_measure, epsilon=epsilon)
            sensor_type_similarities.append(filtered_values)
            extremum_values[i] = max(extremum_values[i], extremum)
        
        layer_similarities[layer] = sensor_type_similarities
    
    return layer_similarities, extremum_values


def extract_layers_max_sim_values(sim_dict: dict, sensor_type: str, channels_list: list):
    """
    Extracts a list of values for a given sensor type from the dictionary and returns the name of the layer with the highest similarity.

    Args:
    - sim_dict (dict): Dictionary containing the similarity values for each layer and sensor type.
    - sensor_type (str): The sensor type to extract values for (e.g., 'grad1', 'grad2', 'mag').
    - channels_list (list): List of all sensor names.

    Returns:
    - values_list (list): List of values corresponding to the given sensor type.
    - max_index (int): Index of the function with the highest similarity.
    - max_layer_name (str): Name of the layer that gave the highest similarity.
    - mask (list): Mask indicating the sensor that gave the highest similarity (1 for the sensor, 0 for others).
    """

    values_list = [values.get(sensor_type, [])[0] for values in sim_dict.values()]
    max_value = max(values_list)
    max_index = values_list.index(max_value)
    max_layer_name = next((key for key, value in sim_dict.items() if value.get(sensor_type, [])[0] == max_value), None)
    mask = [sensor == channels_list[max_index] for sensor in channels_list]

    return values_list, max_index, max_layer_name, mask


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
