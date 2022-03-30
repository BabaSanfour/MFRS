import os
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import sys
sys.path.append('/home/hamza97/MFRS/utils')
from transfer_weights import transfer

path='/home/hamza97/scratch/net_weights/'


class mfm(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, padding=1, type=1):
        super(mfm, self).__init__()
        self.out_channels = out_channels
        if type == 1:
            self.filter = nn.Conv2d(in_channels, 2*out_channels, kernel_size=kernel_size, stride=stride, padding=padding)
        else:
            self.filter = nn.Linear(in_channels, 2*out_channels)

    def forward(self, x):
        x = self.filter(x)
        out = torch.split(x, self.out_channels, 1)
        return torch.max(out[0], out[1])

class resblock_v1(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(resblock_v1, self).__init__()
        self.conv1 = mfm(in_channels, out_channels, kernel_size=3, stride=1, padding=1)
        self.conv2 = mfm(out_channels, out_channels, kernel_size=3, stride=1, padding=1)

    def forward(self, x):
        res = x
        out = self.conv1(x)
        out = self.conv2(out)
        out = out + res
        return out


class network(nn.Module):
    def __init__(self, block, layers, num_classes):
        super(network, self).__init__()

        self.conv1 = mfm(1, 48, 3, 1, 1)

        self.block1 = self._make_layer(block, layers[0], 48, 48)
        self.conv2  = mfm(48, 96, 3, 1, 1)
        self.block2 = self._make_layer(block, layers[1], 96, 96)
        self.conv3  = mfm(96, 192, 3, 1, 1)
        self.block3 = self._make_layer(block, layers[2], 192, 192)
        self.conv4  = mfm(192, 128, 3, 1, 1)
        self.block4 = self._make_layer(block, layers[3], 128, 128)
        self.conv5  = mfm(128, 128, 3, 1, 1)

        self.fc = nn.Linear(8*8*128, 256)
        self.fc2 = nn.Linear(256, num_classes)

        nn.init.normal_(self.fc.weight, std=0.001)
        nn.init.normal_(self.fc2.weight, std=0.001)

    def _make_layer(self, block, num_blocks, in_channels, out_channels):
        layers = []
        for i in range(0, num_blocks):
            layers.append(block(in_channels, out_channels))
        return nn.Sequential(*layers)

    def forward(self, x, label=None):
        x = self.conv1(x)
        x = F.max_pool2d(x, 2) + F.avg_pool2d(x, 2)

        x = self.block1(x)
        x = self.conv2(x)
        x = F.max_pool2d(x, 2) + F.avg_pool2d(x, 2)

        x = self.block2(x)
        x = self.conv3(x)
        x = F.max_pool2d(x, 2) + F.avg_pool2d(x, 2)

        x = self.block3(x)
        x = self.conv4(x)
        x = self.block4(x)
        x = self.conv5(x)
        x = F.max_pool2d(x, 2) + F.avg_pool2d(x, 2)

        x = torch.flatten(x, 1)
        fc = self.fc(x)
        out = self.fc2(fc)
        return out

def LightCNN_V4(pretrained: bool = False, num_classes: int =1000):
    model = network(resblock_v1, [1, 2, 3, 4], num_classes)
    if pretrained== True:
        weights=path+'LightCNN_V4'
        if os.path.isfile(weights):
            model.load_state_dict(torch.load(weights))
        else:
            state_dict = transfer('LightCNN_V4', model, weights)
            model.load_state_dict(state_dict)
    return model
