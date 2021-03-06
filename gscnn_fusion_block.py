import torch
import torch.nn as nn
import torch.nn.functional as F  
import numpy as np  

def conv1x1(in_channel, out_channel, stride=1,padding=0,dilation=1):
    layers = []
    layers.append(nn.Conv2d(in_channel, out_channel,kernel_size=1,stride=stride,dilation=dilation,bias=False))
    layers.append(nn.BatchNorm2d(out_channel))
    layers.append(nn.ReLU(inplace=True))
    return nn.Sequential(*layers)

def conv3x3(in_channel, out_channel, stride=1,padding=0,dilation=1):
    layers = []
    layers.append(nn.Conv2d(in_channel, out_channel,kernel_size=3,stride=stride,padding=padding,dilation=dilation,bias=False))
    layers.append(nn.BatchNorm2d(out_channel))
    layers.append(nn.ReLU(inplace=True))
    return nn.Sequential(*layers)


class ImagePooling(nn.Module):
    def __init__(self, in_channel, out_channel):
        super(ImagePooling, self).__init__()
        self.avgpol = nn.AdaptiveAvgPool2d((1,1))
        self.conv = nn.Conv2d(in_channel, out_channel,kernel_size=1,stride=1,padding=0,dilation=1) 
        
    def forward(self,x):
        _,_,h,w = x.shape 
        x1 = self.avgpol(x)
        x2 = self.conv(x1)
        x3 = F.interpolate(x2,(h,w),mode='bilinear', align_corners=False)

        return x3

class ASPP(nn.Module):
    def __init__(self, in_channel, out_channel,rates):
        super(ASPP, self).__init__()
        self._aspp = nn.Module()
        self._aspp.add_module('conv1',conv1x1(in_channel,out_channel,1,0,1))
        for key, value in enumerate(rates):
            self._aspp.add_module("c{0}".format(key+1),
                                    conv3x3(in_channel,out_channel,stride=1,padding=value, dilation=value))
        self._aspp.add_module('edge_conv',conv1x1(1,out_channel))
        self._aspp.add_module('imgpool',ImagePooling(in_channel,out_channel))
        self.conv1 = conv1x1(out_channel*6, out_channel, stride=1,padding=0,dilation=1)
    def forward(self,x,edge):
        _,_,h,w = x.shape
        img_pool_features = self._aspp.imgpool(x)
        edge_featues = F.interpolate(edge,(h,w),mode='bilinar',align_corners=True)
        edge_featues = self._aspp.edge_conv(edge_featues)
        img_edge_features = torch.cat((edge_featues,img_pool_features), dim=1)

        result = torch.cat([stage(x) for stage in self._aspp.children()[:-2]], dim=1)
        result = torch.cat((img_edge_features,result), dim=1)
        result = self.conv1(result)
        return result

if __name__ == '__main__':
    print('load_fusion')
else:
    print('load_fusion')
