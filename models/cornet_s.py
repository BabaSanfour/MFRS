"""
Cornet-S network
"""
import os
import math
from typing import Any
from collections import OrderedDict
from torch import nn

import sys
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
from utils.load_weights import load_weights

class Flatten(nn.Module):

    """
    Helper module for flattening input tensor to 1-D for the use in Linear modules
    """

    def forward(self, x):
        return x.view(x.size(0), -1)


class Identity(nn.Module):

    """
    Helper module that stores the current tensor. Useful for accessing by name
    """

    def forward(self, x):
        return x


class CORblock_S(nn.Module):

    scale = 4  # scale of the bottleneck convolution channels

    def __init__(self, in_channels, out_channels, times=1):
        super().__init__()

        self.times = times

        self.conv_input = nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False)
        self.skip = nn.Conv2d(out_channels, out_channels,
                              kernel_size=1, stride=2, bias=False)
        self.norm_skip = nn.BatchNorm2d(out_channels)

        self.conv1 = nn.Conv2d(out_channels, out_channels * self.scale,
                               kernel_size=1, bias=False)
        self.nonlin1 = nn.ReLU(inplace=True)

        self.conv2 = nn.Conv2d(out_channels * self.scale, out_channels * self.scale,
                               kernel_size=3, stride=2, padding=1, bias=False)
        self.nonlin2 = nn.ReLU(inplace=True)

        self.conv3 = nn.Conv2d(out_channels * self.scale, out_channels,
                               kernel_size=1, bias=False)
        self.nonlin3 = nn.ReLU(inplace=True)

        self.output = Identity()  # for an easy access to this block's output

        # need BatchNorm for each time step for training to work well
        for t in range(self.times):
            setattr(self, "norm1_%s"%(t), nn.BatchNorm2d(out_channels * self.scale))
            setattr(self, "norm2_%s"%(t), nn.BatchNorm2d(out_channels * self.scale))
            setattr(self, "norm3_%s"%(t), nn.BatchNorm2d(out_channels))

    def forward(self, inp):
        x = self.conv_input(inp)

        for t in range(self.times):
            if t == 0:
                skip = self.norm_skip(self.skip(x)) #conv and norm apply to x for time 0
                self.conv2.stride = (2, 2)
            else:
                skip = x
                self.conv2.stride = (1, 1)

            x = self.conv1(x)
            x = getattr(self, "norm1_%s"%(t))(x)
            x = self.nonlin1(x)

            x = self.conv2(x)
            x = getattr(self, "norm2_%s"%(t))(x)
            x = self.nonlin2(x)

            x = self.conv3(x)
            x = getattr(self, "norm3_%s"%(t))(x)

            x += skip #recurrent step
            x = self.nonlin3(x)
            output = self.output(x)

        return output


class CORnet_S(nn.Module):

    def __init__(self, num_classes: int = 1000, n_input_channels: int = 3) -> None:
        super(CORnet_S ,self).__init__()

        self.model = nn.Sequential(OrderedDict([
            ('V1', nn.Sequential(OrderedDict([  # this one is custom to save GPU memory
                ('conv1', nn.Conv2d(n_input_channels, 64, kernel_size=7, stride=2, padding=3,
                                bias=False)),
                ('norm1', nn.BatchNorm2d(64)),
                ('nonlin1', nn.ReLU(inplace=True)),
                ('pool', nn.MaxPool2d(kernel_size=3, stride=2, padding=1)),
                ('conv2', nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1,
                                bias=False)),
                ('norm2', nn.BatchNorm2d(64)),
                ('nonlin2', nn.ReLU(inplace=True)),
                ('output', Identity())
            ]))),
            ('V2', CORblock_S(64, 128, times=2)),
            ('V4', CORblock_S(128, 256, times=4)),
            ('IT', CORblock_S(256, 512, times=2)),
            ]))

        self.avgpool = nn.AdaptiveAvgPool2d(1)
        self.classifier = nn.Sequential(OrderedDict([
                ('flatten', Flatten()),
                ('linear', nn.Linear(512, num_classes)),
        ]))

        self._initialize_weights()

    def _initialize_weights(self) :
        # weight initialization
        for m in self.model.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
            # nn.Linear is missing here because I originally forgot
            # to add it during the training of this network
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

    def forward(self,input) :
        output = self.model(input)
        output = self.avgpool(output)
        output = self.classifier(output)
        return output


def cornet_s(pretrained: bool = False, num_classes: int = 1000, n_input_channels: int = 3,  transfer: bool = False, weights: str = None) -> CORnet_S:
    model = CORnet_S(num_classes, n_input_channels)
    if pretrained:
        return load_weights(model, transfer, weights)
    return model
