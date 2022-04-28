"""
    transfer:
        - Download/load the weights of the model
        - Adapt the weights of each model to make weight extraction easier
        - Change the weights of first and last layer if needed
        ---------------
        Input :
            name              string,       model/architecture name
            model             nn.torch,     model
            n_input_channels  integer,      number of input channels in the first layer of model
            weights           string,       weights file name
        ---------------
        Output :
            output_state_dict state_dict,   dict with model weights
        ---------------
        ---------------
    load_weights:
        - Depending on the input paramaters, returns model loaded with target weights
        ---------------
        Input :
            name              string,       model/architecture name
            model             nn.torch,     model
            n_input_channels  integer,      number of input channels in the first layer of model
            weights           string,       weights file name
        ---------------
        Output : model loaded with weights


"""
import os
import torch
import torch.nn as nn
from collections import OrderedDict
from torch.utils.model_zoo import load_url

path='/home/hamza97/scratch/net_weights/'

# Original weights trained on ImageNet (vgg, alexnet, resnet mobilenet, cornet_s and inception) or VGGFace2 (FaceNet)
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

    "mobilenet": "https://download.pytorch.org/models/mobilenet_v2-b0353104.pth",

    'cornet_s': 'https://s3.amazonaws.com/cornet-models/cornet_s-1d3f7974.pth',

    "inception": "https://download.pytorch.org/models/inception_v3_google-0cc3c7bd.pth",

    "FaceNet": "https://github.com/timesler/facenet-pytorch/releases/download/v2.2.9/20180402-114759-vggface2.pt",
}

def state_dict_layer_names(state_dict):
    "create a dict with layer names of model "
    layer_names = [".".join(k.split('.')[:-1]) for k in state_dict.keys()]
    # Order preserving unique set of names
    return list(OrderedDict.fromkeys(layer_names))


def transfer(
    name: str,
    model: nn.Module,
    n_input_channels: int,
    weights: str
    ) -> OrderedDict:
    # created model weights (vide)
    output_state_dict = model.state_dict()
    # get created model layer names
    pytorch_layer_names_out = state_dict_layer_names(output_state_dict)
    # original model weights + get original model layer names

    # if the weights file exists then it should be the one after training on VGGFace because for
    # the othe conditions the weight file name doesnt exist. So to avoid erasing the existing file
    # and to create transfer weights and save the file with the correct name we change weights to
    # "architecture"_weights_"n_input"_VGGFace
    if os.path.isfile(weights):
        original_state_dict= torch.load(weights)
        weights=os.path.join(path, '%s_weights_%sD_input_VGGFace'%(name, n_input_channels))
        if os.path.isfile(weights):
            return weights
    else:
        if name == "SphereFace":
            original_state_dict= torch.load(path+'sphere20a_20171020.pth')
        elif name == "LightCNN":
            original_state_dict= torch.load(path+'LightCNN-V4_checkpoint.pth.tar')['state_dict']
        elif name == "cornet_s" :
            original_state_dict = load_url(model_urls[name], map_location=torch.device('cpu') )
            original_state_dict = original_state_dict['state_dict']
        else:
            original_state_dict = load_url(model_urls[name])
    pytorch_layer_names_original = state_dict_layer_names(original_state_dict)

    # to avoid losing the weights of the pretrained first layer when using transfer learning,
    # we sum the 3 channels weights' of the first layer from state_dict of models trained on 3D pictures (RGB)
    # This idea is supported by the fact that the values R+G+B of a pictures gives the greyscale couterpart
    # Foe more details: https://stackoverflow.com/questions/51995977/how-can-i-use-a-pre-trained-neural-network-with-grayscale-images
    if n_input_channels == 1:
        conv1_weight = original_state_dict[pytorch_layer_names_original[0]+'.weight']
        original_state_dict[pytorch_layer_names_original[0]+'.weight'] = conv1_weight.sum(dim=1, keepdim=True)
    print(original_state_dict[pytorch_layer_names_original[0]+'.weight'].shape)
    # models trained on ImageNet have 1000 class in the last layer, so when training on celebA we have no need to change the last layer,
    # however when training these models on VGGFace with using pretrained-ImageNet weights we need to drop the last layer weights.
    # In the case of models pretrained on VGGFace and then we need to finetunned on celebA, we drop the last layer weights.
    if n_input_channels==3 or weights[-7:] == 'VGGFace' or weights[-8:] == 'VGGFace' or name in ["LightCNN", "SphereFace"]:
        pytorch_layer_names_original.pop()
        # in the case of SphereFace we drop the two last layers ( changes made to the model to fit our project )
        if name == "SphereFace":
            pytorch_layer_names_original.pop()
    # match the original layer names with their counterparts from our output model (desired model)
    i=0
    dictest = {}
    # match layer names: created and original model
    for layer in pytorch_layer_names_original:
        if layer[:3] == "Aux":
            continue
        dictest[layer] = pytorch_layer_names_out[i]
        i+=1

    # match weights with new layers name
    for layer_in in pytorch_layer_names_original:
        if layer_in[:3] == "Aux":
            continue

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
            output_state_dict[running_mean_key_out]=original_state_dict[running_mean_key_in]
        if running_var_key_out in original_state_dict:
            output_state_dict[running_var_key_out]=original_state_dict[running_var_key_in]

    # load weights into created model dict
    model.load_state_dict(output_state_dict)
    # save weights
    torch.save(model.state_dict(), weights)
    # return weights
    return weights


def load_weights(name: str, model: nn.Module, n_input_channels: int, weights: str):
    if weights == None:
        weights=os.path.join(path, '%s_weights_%sD_input'%(name, n_input_channels))
        if name in ['LightCNN','SphereFace', 'FaceNet'] :
            weights=weights+ '_VGGFace'
    else:
        weights=os.path.join(path, weights)
    # Similar to explanation in line 102
    if (weights[-7:] == 'VGGFace' or weights[-8:-1] == 'VGGFace') or not os.path.isfile(weights):
        weights=transfer(name, model, n_input_channels, weights)
    model.load_state_dict(torch.load(weights))
    return model
