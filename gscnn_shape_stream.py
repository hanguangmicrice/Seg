import torch 
import torch.nn as nn
import torch.nn.functional as F  
import numpy as np 
from torch.nn.modules.conv import _ConvNd
from torch.nn.modules.utils import _pair
from gscnn_back_bone import Backbone
import cv2

def conv3x3(in_planes, out_planes, stride=1, dilation=1):
    """3x3 convolution with padding"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=1 * dilation, dilation=dilation, bias=False)


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, in_chans, out_chans, stride=1, dilation=1, downsample=None):
        super(BasicBlock, self).__init__()
        self.conv1 = conv3x3(in_chans, out_chans, stride, dilation)
        self.bn1 = nn.BatchNorm2d(out_chans)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(out_chans, out_chans)
        self.bn2 = nn.BatchNorm2d(out_chans)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out

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

class ShapeBlock(nn.Module):
    def __init__(self):
        super(ShapeBlock,self).__init__()
        self.res_1 = BasicBlock(64,64)
        self.downsample_1 = nn.Conv2d(64,32,kernel_size=1)
        self.res_2 = BasicBlock(32,32)
        self.downsample_2 = nn.Conv2d(32,16,kernel_size=1)
        self.res_3 = BasicBlock(16,16)
        self.downsample_3 = nn.Conv2d(16,8,kernel_size=1)
        self.d_3 = nn.Conv2d(512,1,kernel_size=1)
        self.d_4 = nn.Conv2d(1024,1,kernel_size=1)
        self.d_5 = nn.Conv2d(2048,1,kernel_size=1)
        self.fusion = nn.Conv2d(8,1,kernel_size=1,padding=0)
        self.cw = nn.Conv2d(2,1,kernel_size=1)
        self.gate_1 = GatedSpatialConv2d(32,32)
        self.gate_2 = GatedSpatialConv2d(16,16)
        self.gate_3 = GatedSpatialConv2d(8,8)

    def forward(self,ori_img,results):
        b,_,h,w = ori_img.shape
        x1,x2,x3,x4,x5 = results
        #change output from layer1 into ori shape
        s1 = F.interpolate(x1,(h,w),mode='bilinear',align_corners=True)
        #from output1 to gate1
        shape_fluid = self.res_1(s1)
        shape_fluid = self.downsample_1(shape_fluid)
        s3 = F.interpolate(self.d_3(x3),(h,w),mode='bilinear',align_corners=True)
        shape_fluid = self.gate_1(shape_fluid,s3)
        #from gate1 to gate2
        shape_fluid = self.res_2(shape_fluid)
        shape_fluid = self.downsample_2(shape_fluid)
        s4 = F.interpolate(self.d_4(x4),(h,w),mode='bilinear',align_corners=True)
        shape_fluid = self.gate_2(shape_fluid,s4)
        
        #from gate2 to gate3
        shape_fluid = self.res_3(shape_fluid)
        shape_fluid = self.downsample_3(shape_fluid)
        s5 = F.interpolate(self.d_5(x5),(h,w),mode='bilinear',align_corners=True)
        shape_fluid = self.gate_3(shape_fluid,s5)
       
        # from gate3 to b,1,w,h fusion feature map
        shape_fluid = self.fusion(shape_fluid)
        #generate the edge boundary by using sigmoid
        edge_result = nn.Sigmoid()(shape_fluid)

        #generate canny img array
        img_arr = ori_img.cpu().numpy().transpose(0,2,3,1).astype(np.uint8)
        canny = np.zeros((b,1,h,w))
        for i in range(b):
            canny[i] = cv2.Canny(img_arr[i],10,100)
        canny = torch.from_numpy(canny).cuda().float()

        #concate traditional canny and  cnn edge 
        edge_result = torch.cat((edge_result,canny), dim=1)
        edge_result = self.cw(edge_result)
        edge_result = nn.Sigmoid()(edge_result)

        return edge_result        

test = torch.rand((1,3,512,512))
test = test.cuda()
net1 = Backbone().cuda()
net = ShapeBlock().cuda()
net.eval()
net1.eval()
results = net1(test)
out = net(test,results)
print(out.shape)