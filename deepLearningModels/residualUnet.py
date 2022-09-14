import torch.nn as nn
import torch
from torch import autograd
from PIL import Image
from torchvision import transforms
import numpy as np
#network with 3 conv-and-deconv steps used in paper

class singleBlock(nn.Module):
    def __init__(self, in_ch, out_ch):
        super(singleBlock, self).__init__()
        self.conv1 = nn.Sequential(
            # Level 1
            nn.Conv2d(in_channels=in_ch, out_channels=out_ch, kernel_size=3, padding=1),            
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
            # Level 2            
            nn.Conv2d(in_channels=out_ch, out_channels=out_ch, kernel_size=3, padding=1),            
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.3)   
        )
        self.conv2 = nn.Sequential(
            # Level 1
            nn.Conv2d(in_channels=in_ch, out_channels=out_ch, kernel_size=3, padding=1),            
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),           
        )

    def forward(self, input):        
        x1 = self.conv1(input)        
        x2 = self.conv2(input)        
        x3 = x1 + x2 # Residuals
        return x3

        # x1 = self.conv1(input)                
        # x3 = x1 + input # Residuals
        # return x3


class ResidualCustomUnet(nn.Module):
    def __init__(self, in_ch, out_ch):
        super(ResidualCustomUnet, self).__init__()
        base_filter_num = 64
        
        # Encoder (Downsampling)
        self.conv_down_1 = singleBlock(in_ch, base_filter_num)
        self.pool1 = nn.MaxPool2d(2)
        self.conv_down_2 = singleBlock(base_filter_num, base_filter_num*2)
        self.pool2 = nn.MaxPool2d(2)
        self.conv_down_3 = singleBlock(base_filter_num*2, base_filter_num*4)
        self.pool3 = nn.MaxPool2d(2)
        self.conv_bottom = singleBlock(base_filter_num*4, base_filter_num*8)
        
        # Decoder (Upsampling)
        self.upsample_1 = nn.ConvTranspose2d(base_filter_num*8, base_filter_num*4, kernel_size=2, stride=2)
        self.conv_up_1 = singleBlock(base_filter_num*8, base_filter_num*4)
        self.upsample_2 = nn.ConvTranspose2d(base_filter_num*4, base_filter_num*2, kernel_size=2, stride=2)
        self.conv_up_2 = singleBlock(base_filter_num*4, base_filter_num*2)
        self.upsample_3 = nn.ConvTranspose2d(base_filter_num*2, base_filter_num, kernel_size=2, stride=2)
        self.conv_up_3 = singleBlock(base_filter_num*2, base_filter_num)
        self.conv_out = nn.Conv2d(base_filter_num, out_ch, 1)

    def forward(self, x):
        ###down########         
        down_1 = self.conv_down_1(x)
        #res1 = x + down_1
        # Residuals res = x + down_1
        pool_1 = self.pool1(down_1)

        down_2 = self.conv_down_2(pool_1)
        #res2 = pool_1 + down_2
        # Residuals res = pool1 + down_2
        pool_2 = self.pool2(down_2)

        down_3 = self.conv_down_3(pool_2)
        #res3 = pool_2 + down_3
        # Residuals res = pool2 + down_3
        pool_3 = self.pool3(down_3)

        bottom = self.conv_bottom(pool_3)
        # Residuals res = pool_3 + bottom
        #res4 = pool_3 + bottom
        up_1 = self.upsample_1(bottom)
        merge1 = torch.cat([up_1, down_3], dim=1)

        #print(merge1.shape)
        up_1_out = self.conv_up_1(merge1)
        #res5 = merge1 + up_1_out
        # Add Residual => merge1 + up_1_out
        up_2 = self.upsample_2(up_1_out)

        merge2 = torch.cat([up_2, down_2], dim=1)        
        up_2_out = self.conv_up_2(merge2)
        #res6 = merge2 + up_2_out
        # Add Residual => merge2 + up_2_out
        up_3 = self.upsample_3(up_2_out)

        merge3 = torch.cat([up_3, down_1], dim=1)                
        up_3_out = self.conv_up_3(merge3)
        #res7 = merge3 + up_3_out
        # Add Residual => merge3 + up_3_out
        end_out = self.conv_out(up_3_out)
        # Applying classification in the output layer        
        out = nn.Sigmoid()(end_out)
        return out

 