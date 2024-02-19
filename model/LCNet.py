import torch
import math
import torch.nn as nn
import torch.nn.functional as F
from torchsummary import summary

import math
__all__ = ["LCNet"]

class Conv(nn.Module):
    def __init__(self, nIn, nOut, kSize, stride, padding, dilation=(1, 1), groups=1, bn_acti=False, bias=False):
        super().__init__()

        self.bn_acti = bn_acti

        self.conv = nn.Conv2d(nIn, nOut, kernel_size=kSize,
                              stride=stride, padding=padding,
                              dilation=dilation, groups=groups, bias=bias)

        if self.bn_acti:
            self.bn_prelu = BNPReLU(nOut)

    def forward(self, input):
        output = self.conv(input)

        if self.bn_acti:
            output = self.bn_prelu(output)

        return output


class BNPReLU(nn.Module):
    def __init__(self, nIn):
        super().__init__()
        self.bn = nn.BatchNorm2d(nIn, eps=1e-3)
        self.acti = nn.SELU(nIn)

    def forward(self, input):
        output = self.bn(input)
        output = self.acti(output)
        return output


class DownSamplingBlock(nn.Module):
    def __init__(self, nIn, nOut):
        super().__init__()
        self.nIn = nIn
        self.nOut = nOut

        if self.nIn < self.nOut:
            nConv = nOut - nIn
        else:
            nConv = nOut

        self.conv3x3 = Conv(nIn, nConv, kSize=3, stride=2, padding=1)
        self.max_pool = nn.MaxPool2d(2, stride=2, padding=0)
        self.bn_prelu = BNPReLU(nOut)

    def forward(self, input):
        output = self.conv3x3(input)

        if self.nIn < self.nOut:
            max_pool = self.max_pool(input)
            output = torch.cat([output,max_pool], 1)

        output = self.bn_prelu(output)

        return output
        
def Split(x,p):
    c = int(x.size()[1])
    c1 = round(c * (1-p))
    x1 = x[:, :c1, :, :].contiguous()
    x2 = x[:, c1:, :, :].contiguous()
    return x1, x2
    
class TCA(nn.Module):
    def __init__(self, c, d=1, dropout=0, kSize=3, dkSize=3):
        super().__init__()
      
        self.conv3x3 = Conv(c, c, kSize, 1, padding=1, bn_acti=True)
    
        self.dconv3x3=Conv(c, c, (dkSize, dkSize), 1,
                             padding=(1, 1), groups=c, bn_acti=True)

        self.ddconv3x3 = Conv(c, c, (dkSize, dkSize), 1,
                              padding=(1 * d, 1 * d), groups=c, dilation=(d, d), bn_acti=True)
      
        self.bp = BNPReLU(c)

    def forward(self, input):
        br = self.conv3x3(input)
        
        br1 = self.dconv3x3(br)
        br2 = self.ddconv3x3(br)
        br = br + br1 + br2
        
        output = self.bp(br)
        return output
       

class PCT(nn.Module):
    def __init__(self, nIn, d=1, dropout=0, p = 0.5):
        super().__init__()
        self.p = p
        c = int(nIn) - round(int(nIn) * (1-p))
    
        self.TCA = TCA(c,d)
        
        self.conv1x1 = Conv(nIn, nIn, 1, 1, padding=0, bn_acti=True)
      

    def forward(self, input):
        
        output1, output2 = Split(input,self.p)
       
        output2 = self.TCA(output2)
        
        output = torch.cat([output1, output2], dim=1)
        output = self.conv1x1(output)
        return output
class Bottleneck(nn.Module):
    # 前面1x1和3x3卷积的filter个数相等，最后1x1卷积是其expansion倍
    

    def __init__(self, in_planes, planes, stride=1,d=1):
        super(Bottleneck, self).__init__()
        self.conv1 = nn.Conv2d(in_planes, planes//4, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes//4)
        
        self.TCA = TCA(planes//4,2)
        self.conv3 = nn.Conv2d(planes//4, planes,
                               kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(planes)


        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, self.expansion*planes,
                          kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(self.expansion*planes)
            )

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.TCA(out)
        out = self.bn3(self.conv3(out))
        out += self.shortcut(x)
        out = F.relu(out)
        return out
        
class LCNet(nn.Module):
    def __init__(self, classes=19, block_1=3, block_2=7, C = 32, P=0.5):
        super().__init__()
     
        self.Init_Block = nn.Sequential(
            Conv(3, C, 3, 2, padding=1, bn_acti=True),
            Conv(C, C, 3, 1, padding=1, bn_acti=True),
            Conv(C, C, 3, 1, padding=1, bn_acti=True)
        )
        dilation_block_1 = [2, 2, 2,2, 2, 2,2, 2, 2]
        dilation_block_2 = [4, 4, 8, 8,16,16,32,32,32,32,32,32]
       
        #Block 1
        self.LC_Block_1 = nn.Sequential()
        self.LC_Block_1.add_module("downsample", DownSamplingBlock(C, C*2))
        
        for i in range(0, block_1):        
            #self.LC_Block_1.add_module("LC_Module_1_" + str(i), Bottleneck(C*2, C*2,d = dilation_block_1[i]))
            self.LC_Block_1.add_module("LC_Module_1_" + str(i), PCT(nIn = C*2, d = dilation_block_1[i], p = P ))
      
        #Block 2
        self.LC_Block_2 = nn.Sequential()
        self.LC_Block_2.add_module("downsample", DownSamplingBlock(C*2, C*4))
        for i in range(0, block_2):
            #self.LC_Block_2.add_module("LC_Module_2_" + str(i), Bottleneck(C*4, C*4,d = dilation_block_2[i]))
            self.LC_Block_2.add_module("LC_Module_2_" + str(i), PCT(nIn = C*4, d = dilation_block_2[i], p = P ))
        self.DAD = DAD(C*4, C*2, classes)
    
    def forward(self, input):

        output0 = self.Init_Block(input)

        output1 = self.LC_Block_1(output0)
      
        output2 = self.LC_Block_2(output1)
    
        out = self.DAD(output1,output2)

        out = F.interpolate(out, input.size()[2:], mode='bilinear', align_corners=False)
        return out


class DAD(nn.Module):
    def __init__(self,c2, c1, classes):
        super().__init__()
        self.conv1x1_c = Conv(c2, c1, 1, 1, padding=0, bn_acti=True)
        self.conv1x1_neg = Conv(c1, c1, 1, 1, padding=0, bn_acti=True)
        
        self.conv3x3 = Conv(c1, c1, (3, 3), 1, padding=(1, 1), groups=c1, bn_acti=True)
        self.conv1x1 = Conv(c1, classes, 1, 1, padding=0, bn_acti=True)

    def forward(self, X, Y):
        X_map = torch.sigmoid(X)
        F_sg =  X_map
        
        Yc = self.conv1x1_c(Y)
        Yc_map = torch.sigmoid(Yc)
        Neg_map = self.conv1x1_neg(-Yc_map)
        F_rg = Neg_map*Yc_map + Yc
        F_rg =  F.interpolate(F_rg, F_sg.size()[2:], mode='bilinear', align_corners=False)
        
        output =  F_sg * F_rg
        output = self.conv3x3(output)
        output = self.conv1x1(output)
        return output

"""print layers and params of network"""
if __name__ == '__main__':
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = LCNet(classes=19).to(device)
    summary(model, (3, 1024, 1024))
  


    # print(f"macs: {macs}, params: {params}")
