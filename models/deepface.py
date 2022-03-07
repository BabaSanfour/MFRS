"""
DeepFace network
"""

import math
import torch
from collections import OrderedDict
from torch import nn
from torch.nn.modules.utils import _pair

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

class LocallyConnected2d(nn.Module):

    """
    A conv layer without sharing weights. Used in L4, L5, L6. Wheras keras and tf have this class implemented,
    it doesn't exist in pytorch.
    """
    def __init__(self, in_channels, out_channels, output_size, kernel_size, stride, bias=False):
        super(LocallyConnected2d, self).__init__()
        output_size = _pair(output_size)
        self.weight = nn.Parameter(
            torch.randn(1, out_channels, in_channels, output_size[0], output_size[1], kernel_size**2)
        )
        if bias:
            self.bias = nn.Parameter(
                torch.randn(1, out_channels, output_size[0], output_size[1])
            )
        else:
            self.register_parameter('bias', None)
        self.kernel_size = _pair(kernel_size)
        self.stride = _pair(stride)

    def forward(self, x):
        _, c, h, w = x.size()
        kh, kw = self.kernel_size
        dh, dw = self.stride
        x = x.unfold(2, kh, dh).unfold(3, kw, dw)
        x = x.contiguous().view(*x.size()[:-2], -1)
        # Sum in in_channel and kernel_size dims
        out = (x.unsqueeze(1) * self.weight).sum([2, -1])
        if self.bias is not None:
            out += self.bias
        return out


class DeepFace(nn.Module):

    def __init__(self):
        super(DeepFace, self).__init__()
        self.C1_conv1=nn.Conv2d(in_channels=1,out_channels= 32, kernel_size=11,stride=1,  bias=False)
        self.C1_nonlin1=nn.ReLU(inplace=True)
        self.M2_pool=nn.FractionalMaxPool2d(kernel_size=3, output_size=(71,71))
        self.C3_conv2= nn.Conv2d(32, 16, kernel_size=9)
        self.C3_nonlin2=nn.ReLU(inplace=True)
        self.L4_lc2d=LocallyConnected2d(16, 16, kernel_size=9, output_size=55 , stride=1,  bias=False)
        self.L4_nonlin3=nn.ReLU(inplace=True)
        self.L5_lc2d= LocallyConnected2d(16, 16, kernel_size=7, output_size=25 , stride=2,  bias=False)
        self.L5_nonlin5=nn.ReLU(inplace=True)
        self.L6_lc2d=LocallyConnected2d(16, 16, kernel_size=5, output_size=21 , stride=1,  bias=False)
        self.L6_nonlin1=nn.ReLU(inplace=True)
        self.flatten=Flatten()
        self.linear1=nn.Linear(7056, 4096)
        self.dropout=nn.Dropout(p=0.5, inplace=False)
        self.linear2=nn.Linear(4096, 1000)
        self.softmax=nn.Softmax(1)

        self._initialize_weights()

    def _initialize_weights(self) :
        # weight initialization


        for m in self.modules():
            if isinstance(m, (nn.Conv2d, nn.Linear)):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

    def forward(self,input) :
        x=self.C1_conv1(input)
        x=self.C1_nonlin1(x)
        x=self.M2_pool(x)
        x=self.C3_conv2(x)
        x=self.C3_nonlin2(x)
        x=self.L4_lc2d(x)
        x=self.L4_nonlin3(x)
        x=self.L5_lc2d(x)
        x=self.L5_nonlin5(x)
        x=self.L6_lc2d(x)
        x=self.L6_nonlin1(x)
        x=self.flatten(x)
        x=self.linear1(x)
        x=self.dropout(x)
        x=self.linear2(x)
        x=self.softmax(x)
        return x
