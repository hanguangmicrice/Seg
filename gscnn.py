import torch
import torch.nn as nn
import torch.nn.functional as F  
import numpy as np  
from shapestreamconv import GatedSpatialConv2d
from back_bone_gscnn import Backbone
from model.module import ASPP

class GSCNN(nn.Module):
    def __init__(self,cfg):
        super(GSCNN,self).__init__()
        self.backbone = Backbone()
        self.ASPP = ASPP()
        self.
