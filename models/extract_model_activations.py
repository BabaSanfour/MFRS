import os
import pickle
import numpy as np
from typing import Union,  Dict

import torch
import torch.nn as nn
import torch.nn.modules as mod

import sys
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
from utils.config import activations_folder
from utils.arg_parser import get_training_config_parser
from utils.load_data import Stimuliloader
from utils.load_weights import load_weights

from inception import inception_v3, InceptionA, InceptionC, InceptionB, Inception3, InceptionE, InceptionD
from cornet_s import cornet_s
from mobilenet import mobilenet_v2
from resnet import resnet50
from vgg import vgg16_bn
from FaceNet import FaceNet, Block35, Mixed_6a, Mixed_7a, Block8, Block17
from SphereFace import SphereFace

import logging  # Import the logging module

# Set up logging configuration
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Global variable to store activations
activations = {}


def get_activation_hook(name: str):
    """
    Create a hook to capture the activation of a specific layer in the network.

    Parameters:
        name (str): A name to identify the activation tensor in the activations dictionary.

    Returns:
        Callable: A hook function that can be registered to capture activations.
    """
    def hook(module, input, output):
        """
        Hook function to capture the activation tensor.

        This function detaches the tensor from the computation graph, converts it to a NumPy array,
        and reshapes it for further analysis.

        Parameters:
            module (torch.nn.Module): The layer to which the hook is attached.
            input (tuple): Input tensors to the layer.
            output (torch.Tensor): The output tensor of the layer.

        Returns:
            None
        """
        # Detach the tensor from the computation graph, convert to NumPy, and reshape
        activation = output.detach().cpu().numpy().reshape(output.size(0), -1)
        activations[name] = activation

    return hook


def model_activations(model: nn.Module,
    data: torch.Tensor,
    weights: Union[str, None] = None,
    file_name: Union[str, None] = None,
    transfer: bool = False
) -> dict:
    """
    Get a dictionary with the activations of selected layers for a given input tensor.

    Args:
        model (nn.Module): The neural network architecture.
        data (torch.Tensor): The input tensor to the network.
        weights (str): The path to the trained model weights. If set to None, the model will use random weights.
        file_name (str, optional): The name of the file to save the activations to.
            This parameter is only used if 'save' is set to True.
        transfer (bool, optional): Indicates whether transfer first layer.

    Returns:
        dict: A dictionary with layer names as keys and their activations as values.
              Each activation value is a numpy array of shape [n_cons, N_neurons_in_layer], where 'n_cons' is the
              number of examples in the input tensor, and 'N_neurons_in_layer' is the number of neurons in the
              corresponding layer.
    """

    # Load model weights if provided
    if weights and weights != 'None':
        model = load_weights(model, transfer, weights)
    else:
        logger.warning("Failed to load model weights. Continuing with random weights.")

    model.eval()
    activations.clear()  # Clear previous activations

    # Register hooks for selected layer types
    for name, layer in model.named_modules():
        if type(layer) in layer_types_to_select:
            layer.register_forward_hook(get_activation_hook(name))

    with torch.no_grad():
        output = model(data)

    with open(file_name, 'wb') as f:
        pickle.dump(activations, f)

    return activations


def extract_activations(
    model_name: str,
    analysis_type: str,
    model: nn.Module,
    weights: str,
    activ_type: str = "trained",
    seed: int = 0,
    transfer: bool = False
) -> Dict[str, np.ndarray]:
    """
    Extracts the activations of the specified layers for the given stimuli using the specified model.

    Args:
        model_name (str): The name of the model.
        analysis_type (str): The type of analysis.
        model (nn.Module): The model.
        weights (str): The path to the model weights.
        activ_type (str, optional): Trained or untrained model.
        seed (int, optional): The seed to used for training.
        transfer (bool, optional): Indicates whether transfer first layer.

    Returns:
        dict: A dictionary containing the activations of the specified layers for the given stimuli.
    """
    images = Stimuliloader(450, "Stimuli")
    images = next(iter(images))
    activations_file = os.path.join(activations_folder, f"{model_name}_{analysis_type}_activations_{activ_type}_{seed}.pkl")

    if os.path.isfile(activations_file):
        logger.info(f"Activations file ({model_name}) for {analysis_type} already exists!!!")
        with open(activations_file, 'rb') as file:
            activations = pickle.load(file)
    else:
        logger.info(f"Activations file ( {model_name}) for {analysis_type} does not exist. Creating it...")
        activations = model_activations(model, images, weights, file_name=activations_file, transfer=transfer)
        logger.info(f"Activations file ({model_name}) for {analysis_type} created.")

    return activations


if __name__ == '__main__':
    parser = get_training_config_parser()
    args = parser.parse_args()

    layer_types_to_select = [ mod.activation.PReLU, mod.batchnorm.BatchNorm1d, mod.conv.Conv2d,
                        mod.linear.Linear, mod.batchnorm.BatchNorm2d, mod.pooling.AdaptiveAvgPool2d,
                        mod.activation.ReLU, mod.pooling.MaxPool2d,  mod.activation.ReLU6, 
                         InceptionA, InceptionC, InceptionB, Inception3,
                         Block35, Mixed_6a, Mixed_7a, Block8, Block17, InceptionE, InceptionD
                        ]

    model_name = args.model
    # Create the model
    model_cls = {
        "cornet_s": cornet_s,
        "resnet50": resnet50,
        "mobilenet": mobilenet_v2,
        "vgg16_bn": vgg16_bn,
        "inception_v3": inception_v3,
        "FaceNet": FaceNet,
        "SphereFace": SphereFace
    }[model_name]

    model = model_cls(False, args.num_classes, args.n_input_channels)

    # Call extract_activations
    activations = extract_activations(model_name, args.analysis_type, model, args.in_weights, args.activ_type, args.seed, args.transfer)