"""
    - load the weights of the model
"""
import os
import torch
import torch.nn as nn
from transfer_weights import transfer

path='/home/hamza97/scratch/net_weights/'

def load_weights(name: nn.Module, model: str, n_input_channels: int, weights: str):
    if weights == None:
        weights=os.path.join(path, '%s_weights_%sD_input'%(name, n_input_channels))
        if name in ['vgg16_bn', 'vgg16_bn']:
            weights=weights+'_bn'
    else:
        weights=os.path.join(path, weights)
    if os.path.isfile(weights):
        model.load_state_dict(torch.load(weights))
    else:
        state_dict = transfer(name, model, n_input_channels, weights)
        model.load_state_dict(state_dict)
    return model
