import os
import pickle
import torch
import torchvision
import torch.nn as nn

activations_folder = '/home/hamza97/scratch/data/MFRS_data/activations'
activations={}
def get_activation(name: str, batch_size: int = 150):
    """ Get the activation of the network without changing the Parameters"""
    def hook(model, input, output):
        # detach() is used to detach a tensor from the current computational graph. It returns a new tensor
        # that doesn't require a gradient. When we don't need a tensor to be traced for the gradient computation,
        # we detach the tensor from the current computational graph
        # transform the tensor to numpy (needed type for rdm)
        # reshape to the proper shape we need to compute rdm
        # Function from https://discuss.pytorch.org/t/how-can-l-load-my-best-model-as-a-feature-extractor-evaluator/17254/24
        activations[name] = output.detach().cpu().numpy().reshape(batch_size, -1) # change to output.shape
    return hook

def model_activations(model: nn.Module, data: torch.Tensor, weights: str, method: str = 'all', list_layers: list = [],
                        save: bool = False, model_name: str = None, data_name: str = None):
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
        model_name      network name, required if save=True. Default: None
        data_name       data name, required if save=True. Default: None

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
    if save:
        assert model_name != None, "\nmodel_name not specified.\n Invalid input!"
        assert data_name != None, "\ndata_name not specified.\n Invalid input!"

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if weights!= 'None':
        model.load_state_dict(torch.load(weights, map_location=torch.device(device)))
    else:
        print("\nFailed to load model weights.\n")
        print("\nContinuing with random weights.\n")


    if method == "index":
        for idx, layer in enumeate(model.named_modules()):
            if idx in list_layers:
                layer.register_forward_hook(get_activation(name, len(data)))
    elif method=="list":
        for name, layer in model.named_modules():
            if name in list_layers:
                layer.register_forward_hook(get_activation(name, len(data)))
    elif method=="type":
        for name, layer in model.named_modules():
            if type(layer) in list_layers:
                layer.register_forward_hook(get_activation(name, len(data)))
    else:
        for name, layer in model.named_modules():
            layer.register_forward_hook(get_activation(name, len(data)))

    output = model(data)
    if save:
        file=os.path.join(activations_folder, "%s_%s_activations.pkl"%(model_name, data_name))
        save_activations(activations, file)

    return activations

def save_activations(activations, file):
    """ Save activations in pickle files """
    with open(file, 'wb') as f:
        pickle.dump(activations, f)
    print('File saved successfully')


def load_activations(file):
    """ Load activations from pickle files """
    with open(file, 'rb') as f:
        activations = pickle.load(f)
    print('File loaded successfully')
    return activations
