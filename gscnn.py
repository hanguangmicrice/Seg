import torch
import torch.nn as nn
import torch.nn.functional as F  
import numpy as np  
from gscnn_back_bone import Backbone
from gscnn_fusion_block import ASPP
from gscnn_shape_stream import 


class GSCNN(nn.Module):
    def __init__(self,cfg):
        super(GSCNN,self).__init__()
        self.backbone = Backbone()
        self.ASPP = ASPP()
        self.
