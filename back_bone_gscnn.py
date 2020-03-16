import torch
import torch.nn as nn
import torch.nn.functional as F  
from torchvision.models import resnet
import numpy as np  

class Backbone(nn.Module):
    def __init__(self,mode_name='resnet50'):
        super(Backbone,self).__init__()
        assert mode_name in ('resnet18','resnet34','resnet50', 'resnet101','resnet152')
        if mode_name =='resnet18':
            self.res_back_bone = resnet.resnet18()
        elif mode_name == 'resnet34':
            self.res_back_bone = resnet.resnet34()
        elif mode_name == 'resnet50':
            self.res_back_bone = resnet.resnet50()
        elif mode_name == 'resnet101':
            self.res_back_bone = resnet.resnet101()
        else:
            self.res_back_bone = resnet.resnet152()
        self.backbone = nn.Module()
        layer0 = nn.Sequential(*list(self.res_back_bone.children())[:4])
        self.backbone.add_module('layer0', layer0)
        self.backbone.add_module('layer1', self.res_back_bone.layer1)
        self.backbone.add_module('layer2', self.res_back_bone.layer2)
        self.backbone.add_module('layer3', self.res_back_bone.layer3)
        self.backbone.add_module('layer4', self.res_back_bone.layer4)
        
    def forward(self, x):
        x0 = self.backbone.layer0(x)
        x1 = self.backbone.layer1(x0)
        x2 = self.backbone.layer2(x1)
        x3 = self.backbone.layer3(x2)
        x4 = self.backbone.layer4(x3)
        
        return [x0,x1,x2,x3,x4]


net = Backbone()
net.eval()
test_tensor = torch.rand((1,3,512,514))
print(net(test_tensor))