import os
import torch
import torch.nn as nn
from collections import OrderedDict

import sys
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
from utils.config import weights_path

def state_dict_layer_names(state_dict : OrderedDict) -> list:
    """
    Create a list of layer names from a state_dict.

    Args:
        state_dict (OrderedDict): The state_dict of a model.

    Returns:
        list: A list of unique layer names.
    """
    layer_names = [".".join(k.split('.')[:-1]) for k in state_dict.keys()]
    # Order preserving unique set of names
    return list(OrderedDict.fromkeys(layer_names))

def transfer_weights(model : nn.Module, original_state_dict : OrderedDict) -> OrderedDict:
    """
    Transfer weights from a pre-trained model to the given model.

    Args:
        model (nn.Module): The model to which weights will be transferred.
        original_state_dict (OrderedDict): The state_dict of the pre-trained model.

    Returns:
        OrderedDict: The state_dict of the model with transferred weights.
    """
    output_state_dict = model.state_dict()
    pytorch_layer_names_out = state_dict_layer_names(output_state_dict)
    pytorch_layer_names_original = state_dict_layer_names(original_state_dict)
    
    # Ensure that the original model and the target model have the same number of layers
    assert len(pytorch_layer_names_out) == len(pytorch_layer_names_original), "Models have different architectures."

    # Transfer weights and biases
    for layer_in, layer_out in zip(pytorch_layer_names_original, pytorch_layer_names_out):
        if layer_in[:3] == "Aux":
            continue
        
        weight_key_in, weight_key_out = layer_in + '.weight', layer_out + '.weight'
        bias_key_in, bias_key_out = layer_in + '.bias', layer_out + '.bias'
        running_mean_key_in, running_mean_key_out = layer_in + '.running_mean', layer_out + '.running_mean'
        running_var_key_in, running_var_key_out = layer_in + '.running_var', layer_out + '.running_var'
        
        output_state_dict[weight_key_out] = original_state_dict[weight_key_in]
        if bias_key_in in original_state_dict:
            output_state_dict[bias_key_out] = original_state_dict[bias_key_in]
        if running_mean_key_in in original_state_dict:
            output_state_dict[running_mean_key_out] = original_state_dict[running_mean_key_in]
        if running_var_key_in in original_state_dict:
            output_state_dict[running_var_key_out] = original_state_dict[running_var_key_in]

    # Update the first layer weights for grayscale images
    first_layer_key = pytorch_layer_names_original[0] + '.weight'
    if first_layer_key in output_state_dict:
        conv1_weight = original_state_dict[first_layer_key]
        output_state_dict[first_layer_key] = conv1_weight.sum(dim=1, keepdim=True)

    return output_state_dict

def load_weights(model: nn.Module, transfer: str, weights: str) -> nn.Module:
    """
    Load weights into a model, optionally applying transfer learning.

    Args:
        model (nn.Module): The model to load weights into.
        transfer (str): Indicates whether to apply transfer learning.
        weights (str): The name of the weights file.

    Returns:
        nn.Module: The model with loaded weights.
    """

    weights=os.path.join(weights_path, weights)
    original_state_dict = torch.load(weights)
    if transfer:
        model.load_state_dict(transfer_weights(model, original_state_dict))
        return model
    model.load_state_dict(original_state_dict)
    return model