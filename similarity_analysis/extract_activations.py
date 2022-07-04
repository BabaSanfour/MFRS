import os
import pickle
import torch
import torchvision
import torch.nn as nn
import sys
import numpy as np
sys.path.append('/home/hamza97/MFRS/utils')
from general import save_pickle, load_pickle, save_npy
from config_sim_analysis import activations_folder

activations={}

def get_activation(name: str):
    """ Get the activation of the network without changing the Parameters"""
    def hook(model, input, output):
        # detach() is used to detach a tensor from the current computational graph. It returns a new tensor
        # that doesn't require a gradient. When we don't need a tensor to be traced for the gradient computation,
        # we detach the tensor from the current computational graph
        # transform the tensor to numpy (needed type for rdm)
        # reshape to the proper shape we need to compute rdm
        # Function from https://discuss.pytorch.org/t/how-can-l-load-my-best-model-as-a-feature-extractor-evaluator/17254/24
        activations[name] = output.detach().cpu().numpy().reshape(output.shape[0], -1) # output.shape[0] is batch batch_size which should be equal to N_conditions
    return hook

def model_activations(model: nn.Module, data: torch.Tensor, weights: str, method: str = 'all', list_layers: list = [],
                        save: bool = False, file_name: str = None):
    """
        Get a dict with layers names and their activations for your input data

        Parameters:
        ---------------
        model           your architecture
        data_name       the name of your set of data,
        batch_size      your batch size, preferrably the length of your data
        weights         the path to your trained architecture
        method          method to select the layers that you want to compute their activations
                            - must be index, list, type or all. default is all.
        list_layers     list of layers to get their activtions : expecting:
                            - for 'method'=index: list of layers indexs
                            - for 'method'=list:  list of layers names
                            - for 'method'=type:  list of layers types
                            - for 'method'=all:   nothing, return all layers.
        save            save computed similarity scores stats, default: False

        returns:
        ---------------
        activations     a dict with layers names as keys and their activations as values
            activation (values) size: [n_cons, N_neurons_in_layer]
    """
    if method not in ["all", "index", "list", "type"] and list_layers == []:
        print("\nMethod specified not in [all, index, list, type].\n")
        return "Invalid input!"
    if method in ["index", "list", "type"] and list_layers == []:
        print("\nList of layers to select is empty.\n")
        return "Invalid input!"
    if (method ==  "index") and (False in [type(item)==int for item in list_layers]):
        print("\nFor method index, the list_layers must be of int.\n")
        return "Invalid input!"
    if (method ==  "index") and (max(list_layers) > len(list(model.named_modules()))):
        print("\nOne Provided index (%d) is greated than the number of layers in the model (%d).\n"%(max(list_layers), len(list(model.named_modules()))))
        return "Invalid input!"
    if (method ==  "list") and (False in [type(item)==str for item in list_layers]):
        print("\nFor method list, the list_layers must be of strings.\n")
        return "Invalid input!"
    if (method ==  "list") and not all(elem in [item[0] for item in list(model.named_modules())] for elem in list_layers):
        print("\nOne Provided layer names don't exist in network.\n")
        return "Invalid input!"
    if method ==  "type" and not all(elem in [item[1] for item in list(model.named_modules())] for elem in list_layers):
        print("\nProvided layers types don't match network layers types.\n")
        return "Invalid input!"

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if weights!= 'None':
        model.load_state_dict(torch.load(weights, map_location=torch.device(device)))
    else:
        print("\nFailed to load model weights.\n")
        print("\nContinuing with random weights.\n")


    if method == "index":
        for idx, layer in enumeate(model.named_modules()):
            if idx in list_layers:
                layer.register_forward_hook(get_activation(name))
    elif method=="list":
        for name, layer in model.named_modules():
            if name in list_layers:
                layer.register_forward_hook(get_activation(name))
    elif method=="type":
        for name, layer in model.named_modules():
            if type(layer) in list_layers:
                layer.register_forward_hook(get_activation(name))
    else:
        for name, layer in model.named_modules():
            layer.register_forward_hook(get_activation(name))

    output = model(data)
    if save:
        save_pickle(activations, file_name)

    return activations


def get_main_network_activations(name: str, layers: list, save: bool = True):
    """Get the activations of the main layers of a network"""
    if os.path.exists(os.path.join(activations_folder, '%s_main.pkl'%name)):
        main_activ=load_pickle(os.path.join(activations_folder, '%s_main.pkl'%name))
        return main_activ
    else:
        activations=load_pickle(os.path.join(activations_folder, '%s.pkl'%name))
        main_activ={}
        for layer in layers:
            main_activ[layer] = activations[layer]
        if save:
            save_pickle(main_activ, os.path.join(activations_folder, '%s_main.pkl'%name))
        return main_activ

def get_whole_network_activations(name: str, save: bool = True):
    """Concatinate the main activations of a network into 1 array"""
    if os.path.exists(os.path.join(activations_folder, '%s_model.npy'%name)):
        whole=load_npy(os.path.join(activations_folder, '%s_model.npy'%name))
        return whole
    else:
        activations_main=load_pickle(os.path.join(activations_folder, '%s_main.pkl'%name))
        whole = np.concatenate([activ for activ in activations_main.values()], axis=1)
        if save:
            save_npy(whole, os.path.join(activations_folder, '%s_model.npy'%name))
        return whole
