"""
Cornet-Z network
"""
import sys
import math
from typing import Any
from collections import OrderedDict
from torch import nn
sys.path.append('../../MFRS/')
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


class CORblock_Z(nn.Module):

    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size,
                              stride=stride, padding=kernel_size // 2)
        self.nonlin = nn.ReLU(inplace=True)
        self.pool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.output = Identity()  # for an easy access to this block's output

    def forward(self, inp):
        x = self.conv(inp)
        x = self.nonlin(x)
        x = self.pool(x)
        x = self.output(x)  # for an easy access to this block's output
        return x


class CORnet_Z(nn.Module):

    def __init__(self, num_classes: int = 1000, n_input_channels: int = 3) -> None:
        super(CORnet_Z ,self).__init__()

        self.model = nn.Sequential(OrderedDict([
        ('V1', CORblock_Z(n_input_channels, 64, kernel_size=7, stride=2)),
        ('V2', CORblock_Z(64, 128)),
        ('V4', CORblock_Z(128, 256)),
        ('IT', CORblock_Z(256, 512)),
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


def cornet_z(pretrained: bool = False, num_classes: int = 1000, n_input_channels: int = 3,  map_location=None, weights: str = None,**kwargs: Any) -> CORnet_Z:
    model = CORnet_Z(num_classes, n_input_channels)
    if pretrained:
        return load_weights('cornet_z', model, n_input_channels, weights)
    return model
