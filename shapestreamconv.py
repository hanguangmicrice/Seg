# -*- coding: utf-8 -*-
"""
Created on Tue Oct 22 17:00:18 2019

@author: william.han
"""

import torch
import torch.nn as nn
import torch.nn.functional as F  
from torch.nn.modules.conv import _ConvNd
from torch.nn.modules.utils import _pair
import numpy as np

class GatedSpatialConv2d(_ConvNd):
    def __init__(self,in_ch,out_ch,kernel_size=1,stride=1,
                    padding=0,dilation=1,groups=1,bias=False):

        kernel_size = _pair(kernel_size)
        stride = _pair(stride)
        padding = _pair(padding)
        dilation = _pair(dilation)

        super(GatedSpatialConv2d,self).__init__(in_ch, out_ch,
                            kernel_size, stride, padding, dilation,
                            False, _pair(0), groups, bias, 'zeros')
        self._gate_conv = nn.Sequential(nn.BatchNorm2d(in_ch+1),
                                        nn.Conv2d(in_ch+1,in_ch+1,1),
                                        nn.ReLU(),
                                        nn.Conv2d(in_ch+1,1,1),
                                        nn.BatchNorm2d(1),
                                        nn.Sigmoid())
    def forward(self, input_features, gating_features):
        """
        input_features is the output from shape stream [n,c,h,w]
        gating_feature is the output from resnet stram and only one channel [n,1,h,w]
        """
        x = torch.cat([input_features, gating_features], dim=1)
        alphas = self._gate_conv(x)
        input_features = (alphas+1)*input_features

        return F.conv2d(input_features,self.weight,self.bias,
                        self.stride,self.padding,self.dilation,self.groups)


test_tensor = torch.rand((1,3,512,514))
test_tensor1 = torch.rand((1,1,512,514))
net = GatedSpatialConv2d(3,1)
net.eval()
print(net(test_tensor,test_tensor1))