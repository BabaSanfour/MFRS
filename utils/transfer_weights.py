"""
    - Download the weights of the model
    - Adapt the weights of each model to make weight extraction easier
    - Change the weights of first and last layer
"""

import torch
import torch.nn as nn
from collections import OrderedDict

from torch.utils.model_zoo import load_url

model_urls = {
    'alexnet': 'https://download.pytorch.org/models/alexnet-owt-7be5be79.pth',

    'vgg16': 'https://download.pytorch.org/models/vgg16-397923af.pth',
    'vgg19': 'https://download.pytorch.org/models/vgg19-dcbb9e9d.pth',
    'vgg16_bn': 'https://download.pytorch.org/models/vgg16_bn-6c64b313.pth',
    'vgg19_bn': 'https://download.pytorch.org/models/vgg19_bn-c79401a0.pth',

    "resnet18": "https://download.pytorch.org/models/resnet18-f37072fd.pth",
    "resnet34": "https://download.pytorch.org/models/resnet34-b627a593.pth",
    "resnet50": "https://download.pytorch.org/models/resnet50-0676ba61.pth",
    "resnet101": "https://download.pytorch.org/models/resnet101-63fe2227.pth",
    "resnet152": "https://download.pytorch.org/models/resnet152-394f9c45.pth",


}

def state_dict_layer_names(state_dict):
    "create a dict with layer names of model "
    layer_names = [".".join(k.split('.')[:-1]) for k in state_dict.keys()]
    # Order preserving unique set of names
    return list(OrderedDict.fromkeys(layer_names))


def transfer(
    name: str,
    model: nn.Module,
    weights: str
    ) -> OrderedDict:

    # original model weights
    original_state_dict = load_url(model_urls[name])
    # created model weights (vide)
    output_state_dict = model.state_dict()
    # get created model layer names
    pytorch_layer_names_out = state_dict_layer_names(output_state_dict)
    # get original model layer names
    pytorch_layer_names_original = state_dict_layer_names(original_state_dict)

    # Drop first and last layers name from original model
    pytorch_layer_names_original.pop()
    pytorch_layer_names_original.pop(0)

    # match layer names: created and original model
    dictest = {}
    i=0
    for layer in pytorch_layer_names_original:
        dictest[layer] = pytorch_layer_names_out[i+1]
        i+=1
    # match weights with new layers name
    for layer_in in pytorch_layer_names_original:
        layer_out=dictest[layer_in]
        weight_key_in = layer_in + '.weight'
        bias_key_in = layer_in + '.bias'
        running_mean_key_in = layer_in + '.running_mean'
        running_var_key_in = layer_in + '.running_var'
        weight_key_out = layer_out + '.weight'
        bias_key_out = layer_out + '.bias'
        running_mean_key_out = layer_out + '.running_mean'
        running_var_key_out = layer_out + '.running_var'

        output_state_dict[weight_key_out]=original_state_dict[weight_key_in]
        if bias_key_out in original_state_dict:
            output_state_dict[bias_key_out]=original_state_dict[bias_key_in]
        if running_mean_key_out in original_state_dict:
            output_state_dict[running_mean_key_out]=stateoriginal_state_dict_dict[running_mean_key_in]
        if running_var_key_out in original_state_dict:
            output_state_dict[running_var_key_out]=original_state_dict[running_var_key_in]

    # load weights into created model dict
    model.load_state_dict(output_state_dict)
    # save weights
    torch.save(model.state_dict(), weights)
    print("weights transfered and saved")
    # return weights
    return output_state_dict
