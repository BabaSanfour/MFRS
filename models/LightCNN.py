import os
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import sys
from typing import Type, Any, Callable, Union, List, Optional
sys.path.append('/home/hamza97/MFRS/utils')
from load_weights import load_weights


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
    def __init__(self, block: Type[resblock_v1], layers: List[int], num_classes: int = 1000, n_input_channels: int = 3):
        super(network, self).__init__()

        self.conv1 = mfm(n_input_channels, 48, 3, 1, 1)

        self.block1 = self._make_layer(block, layers[0], 48, 48)
        self.conv2  = mfm(48, 96, 3, 1, 1)
        self.block2 = self._make_layer(block, layers[1], 96, 96)
        self.conv3  = mfm(96, 192, 3, 1, 1)
        self.block3 = self._make_layer(block, layers[2], 192, 192)
        self.conv4  = mfm(192, 128, 3, 1, 1)
        self.block4 = self._make_layer(block, layers[3], 128, 128)
        self.conv5  = mfm(128, 128, 3, 1, 1)

        self.fc = nn.Linear(25088, 256)
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
        x = self.fc(x)
        x = self.fc2(x)
        return x


def LightCNN_V4(pretrained: bool = False, num_classes: int = 1000, n_input_channels: int = 3, weights: str = None) -> network:
    model = network(resblock_v1, [1, 2, 3, 4], num_classes, n_input_channels)
    if pretrained:
        return load_weights('LightCNN', model, n_input_channels, weights)
    return model
