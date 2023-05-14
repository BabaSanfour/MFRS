import os
import torch
import torch.nn as nn
import sys
import numpy as np
from typing import Union, List, Dict
sys.path.append('../../MFRS')
from utils.general import save_pickle, load_pickle, load_npy, save_npy
from utils.config_sim_analysis import activations_folder, networks
from utils.load_data import Stimuliloader
from models.inception import inception_v3
from models.cornet_s import cornet_s
from models.mobilenet import mobilenet_v2
from models.resnet import resnet50
from models.vgg import vgg16_bn
from models.FaceNet import FaceNet
from models.SphereFace import SphereFace
from utils.config import get_similarity_parser

activations={}

def get_activation(name: str) -> None:
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

def model_activations(model: nn.Module, data: torch.Tensor, weights: Union[str, None] = None, method: str = 'all', list_layers: List[Union[int, str, type]] = [],
                        save: bool = False, file_name: Union[str, None] = None) -> Dict[str, np.ndarray]:
    """
    Get a dictionary with the activations of selected layers for a given input tensor.

    Parameters:
    -----------
    model: nn.Module
        The neural network architecture.
    data: torch.Tensor
        The input tensor to the network.
    weights: str
        The path to the trained model weights. If set to None, the model will use random weights.
    method: str, optional
        The method to select the layers whose activations to compute. Must be one of 'all', 'index', 'list', or 'type'.
        Default is 'all'.
    list_layers: list, optional
        A list of layers to get their activations. The structure of the list depends on the value of 'method':
        - if 'method' is 'index': a list of layer indices (integers).
        - if 'method' is 'list': a list of layer names (strings).
        - if 'method' is 'type': a list of layer types (nn.Module subclasses).
    save: bool, optional
        Whether to save the computed activations to a file. Default is False.
    file_name: str, optional
        The name of the file to save the activations to. This parameter is only used if 'save' is set to True.

    Returns:
    --------
    activations: dict
        A dictionary with layers names as keys and their activations as values.
        Each activation value is a numpy array of shape [n_cons, N_neurons_in_layer], where 'n_cons' is the number of
        examples in the input tensor and 'N_neurons_in_layer' is the number of neurons in the corresponding layer.
    """

    if method not in ["all", "index", "list", "type"]:
        raise ValueError("Invalid value for 'method'. Must be one of 'all', 'index', 'list', or 'type'.")

    if method in ["index", "list", "type"] and not list_layers:
        raise ValueError("List of layers to select is empty.")

    if method == "index" and not all(isinstance(item, int) for item in list_layers):
        raise ValueError("For method 'index', the list_layers must contain integers only.")

    if method == "index" and max(list_layers) >= len(list(model.named_modules())):
        raise ValueError("One or more of the provided indices exceeds the number of layers in the model.")

    if method == "list" and not all(isinstance(item, str) for item in list_layers):
        raise ValueError("For method 'list', the list_layers must contain strings only.")

    if method == "list" and not all(elem in [item[0] for item in list(model.named_modules())] for elem in list_layers):
        raise ValueError("One or more of the provided layer names do not exist in the network.")

    if method == "type" and not all(isinstance(item, type) and issubclass(item, nn.Module) for item in list_layers):
        raise ValueError("For method 'type', the list_layers must contain nn.Module subclasses only.")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if weights!= 'None':
        model.load_state_dict(torch.load(weights, map_location=torch.device(device)))
    else:
        print("\nFailed to load model weights.\n")
        print("\nContinuing with random weights.\n")
    model.eval()
    model.train(False)

    if method == "index":
        for idx, layer_params in enumerate(model.named_modules()):
            name, layer = layer_params[0], layer_params[1]
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

    ouptput = model(data)
    if save:
        save_pickle(activations, file_name)

    return activations #activation is defined above

def get_main_network_activations(name: str, layers: list, save: bool = True) -> dict:
    """
    Retrieves the activations of the main layers of a network by filtering out any layer that is a replica of a previous
    layer. This helps in obtaining only the activations of the layers that we are interested in.
    
    Parameters:
    -----------
    name : str
        The name of the network whose activations are to be obtained.
    layers : list
        A list of layer names whose activations are to be obtained.
    save : bool, optional
        If True, saves the obtained activations as a pickle file.
    
    Returns:
    --------
    main_activ : dict
        A dictionary containing the activations of the main layers of the network. The keys are layer names and values
        are corresponding activation tensors.
    """
    main_activ_path = os.path.join(activations_folder, f'{name}_main.pkl')
    if os.path.exists(main_activ_path):
        main_activ = load_pickle(main_activ_path)
        return main_activ
    else:
        activations_path = os.path.join(activations_folder, f'{name}.pkl')
        activ = load_pickle(activations_path)
        main_activ = {layer: activ[layer] for layer in layers if 'layer' in layer}
        if save:
            save_pickle(main_activ, main_activ_path)
        return main_activ

def get_whole_network_activations(name: str, save: bool = True) -> np.ndarray:
    """
    Concatenate the main activations of a network into a single array.

    Parameters:
    -----------
    name : str
        Name of the network to retrieve activations for.
    save : bool, optional
        If True, save the concatenated activations as an NPY file. Default is True.

    Returns:
    --------
    whole : np.ndarray
        A 2D array containing the concatenated activations of all main layers of the network.
    """
    activations_file = os.path.join(activations_folder, f'{name}_main.pkl')
    whole_file = os.path.join(activations_folder, f'{name}_model.npy')

    if os.path.exists(whole_file):
        whole = load_npy(whole_file)
    else:
        activations_main = load_pickle(activations_file)
        whole = np.concatenate(list(activations_main.values()), axis=1)
        if save:
            save_npy(whole, whole_file)
    return whole

def extract_activations(num_images: int, stimuli_file: str, model_name: str, model: nn.Module, 
        layers: List[str], weights: str, method: str, save: bool, activ_type: str = "main") -> Dict[str, np.ndarray]:
    """
    Extracts the activations of the specified layers for the given stimuli using the specified model.
    
    Parameters:
    -----------
    num_images : int
        The number of images to be used as stimuli.
    stimuli_file_name : str
        The name of the stimuli file.
    model_name : str
        The name of the model.
    model : torch.nn.Module
        The model.
    layers : list
        The names of the layers to extract the activations from.
    weights : str, optional
        The path to the model weights.
    method : str, optional
        The method for extracting activations.
    save : bool, optional
        Whether or not to save the extracted activations.
        
    Returns:
    --------
    dict
        A dictionary containing the activations of the specified layers for the given stimuli.
    """
    images = Stimuliloader(num_images, stimuli_file)
    images = next(iter(images))
    activations_file = os.path.join(activations_folder, f"{model_name}_{stimuli_file}_activations_{activ_type}.pkl")
    if os.path.isfile(activations_file):
        print(f"Activations file (data: {model_name}) for {stimuli_file} already exists!!!")
        activ = load_pickle(activations_file)
    else:
        activ = model_activations(model, images, weights, method, layers, save, file_name=activations_file)
    return activ

if __name__ == '__main__':
    parser = get_similarity_parser()
    args = parser.parse_args()
    model_cls = { "cornet_s": cornet_s, "resnet50": resnet50, "mobilenet": mobilenet_v2,  "vgg16_bn": vgg16_bn, 
                 "inception_v3": inception_v3, "FaceNet": FaceNet, "SphereFace": SphereFace}[args.model_name]

    model = model_cls(False, 1000, 1)
    list_layers = networks[args.model_name]
    # Call extract_activations
    activs = extract_activations(args.cons, args.stimuli_file_name, args.model_name, model, list_layers, args.weights, args.method, args.save, "main")
    activs = extract_activations(args.cons, args.stimuli_file_name, args.model_name, model, list_layers, args.weights, args.method, args.save, "model")
    activs = extract_activations(args.cons, args.stimuli_file_name, args.model_name, model, list_layers, "None", args.method, args.save, "untrained")