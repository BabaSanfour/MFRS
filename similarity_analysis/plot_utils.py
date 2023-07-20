"""
===============================
Utils file for plotting results
===============================
Util functions for selecting the results that we will plot.
"""
from typing import Dict
import sys
from scipy.stats import sem

from utils.config import channels_mag, channels_grad1, channels_grad2, meg_rdm, meg_sensors, similarity_folder
sys.path.append('../../MFRS/')
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
    sensor_type_idx = {"mag":0, "grad1":1, "grad2":2}
    values_list = [max(values[sensor_type_idx[sensor_type]]) for values in sim_dict.values()]
    max_value = max(values_list)
    max_index = values_list.index(max_value)
    max_layer_name = next((key for key, value in sim_dict.items() if max(value[sensor_type_idx[sensor_type]]) == max_value), None)
    max_sensor_idx = sim_dict[max_layer_name][sensor_type_idx[sensor_type]].index(max_value)
    mask = [channels_grad2[max_sensor_idx] == sensor for sensor in channels_grad2]
    return values_list, max_index, max_layer_name, mask


def get_bootstrap_values(bootstrap_data: Dict[str, Dict[str, list]]) -> Dict[str, Dict[str, float]]:
    """
    Calculate the standard error of the mean (SEM) for each layer and sensor type from bootstrap data.

    Args:
    - bootstrap_data (dict): Dictionary containing the bootstrap values for each layer and sensor type.

    Returns:
    - boot_sem (dict): Dictionary containing the SEM for each layer and sensor type.
                      The structure is boot_sem[layer][sensor_type] = SEM.
    """

    boot_sem = {}
    for layer, values in bootstrap_data.items():
        boot_layer = {}
        for sensor_type, bootstrap_values in values.items():
            boot_layer[sensor_type] = sem(bootstrap_values)
        boot_sem[layer] = boot_layer

    return boot_sem
