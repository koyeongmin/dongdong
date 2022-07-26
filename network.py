#########################################################################
##
## Structure of network.
##
#########################################################################
import torch
import torch.nn as nn
from parameters import Parameters

p = Parameters()

####################################################################
##
## classification_network
##
####################################################################
class classification_network(nn.Module):
    def __init__(self):
        super(classification_network, self).__init__()
        self.conv1 = nn.Sequential(nn.Conv2d(3, 32, 3, padding=1, stride=2, bias=True),
                                nn.BatchNorm2d(32),
                                nn.ReLU(inplace=True)) # 128*128*32
        self.conv2 = nn.Sequential(nn.Conv2d(32, 64, 3, padding=1, stride=2, bias=True),
                                nn.BatchNorm2d(64),
                                nn.ReLU(inplace=True)) # 64*64*32
        self.conv3 = nn.Sequential(nn.Conv2d(64, 128, 3, padding=1, stride=2, bias=True),
                                nn.BatchNorm2d(128),
                                nn.ReLU(inplace=True)) # 32*32*32
        self.conv4 = nn.Sequential(nn.Conv2d(128, 256, 3, padding=1, stride=2, bias=True),
                                nn.BatchNorm2d(256),
                                nn.ReLU(inplace=True)) # 16*16*32
        self.conv5 = nn.Sequential(nn.Conv2d(256, 512, 3, padding=1, stride=2, bias=True),
                                nn.BatchNorm2d(512),
                                nn.ReLU(inplace=True)) # 8*8*32
        self.conv6 = nn.Sequential(nn.Conv2d(512, 1024, 3, padding=1, stride=2, bias=True),
                                nn.BatchNorm2d(1024),
                                nn.ReLU(inplace=True)) # 4*4*32
        self.conv7 = nn.Conv2d(1024, 1024, 4, padding=0, stride=1, bias=True) # 1*1*32

    def forward(self, inputs):
        batch_size = inputs.shape[0]
        output = self.conv1(inputs)
        output = self.conv2(output)
        output = self.conv3(output)
        output = self.conv4(output)
        output = self.conv5(output)
        output = self.conv6(output)
        output = self.conv7(output)

        return output.view(batch_size, -1)
