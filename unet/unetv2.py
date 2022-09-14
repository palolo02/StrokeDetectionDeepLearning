
from turtle import forward
import matplotlib.pyplot as plt
from torch import nn, square, unsqueeze
import torch
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
from torchvision import transforms
from torchvision import models # new
from torchvision.models.detection.mask_rcnn import maskrcnn_resnet50_fpn
import torchvision

# https://huggingface.co/spaces/pytorch/U-NET-for-brain-MRI

import torch
import torchvision.transforms.functional as TF
img_file = "kaggle_3m/train_images/TCGA_CS_4941_19960909_11.tif"

class DoubleConv(nn.Module):
    """ Class that defines the basic operations for each level of the network"""
    def __init__(self, in_channels, out_channels):
        super(DoubleConv, self).__init__()
        # Adding Batch Normalization and/or Dropout to avoid overfitting
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=3, stride=1, padding=1, bias=False),            
            nn.BatchNorm2d(out_channels),
            #nn.Dropout2d(0.25),
            nn.ReLU(inplace=True),     
        )
    
    def forward(self, x):        
        return self.conv(x)

class Unetv2(nn.Module):
    """Class that represents the UNet Architecture """
    def __init__(self, in_channels=3, out_channels=1, features =[64,128,256,512]) -> None:
        super(Unetv2, self).__init__()
        self.ups = nn.ModuleList()
        self.downs = nn.ModuleList()
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)

        # Downsampling in UNET
        for feature in features:
            self.downs.append(DoubleConv(in_channels=in_channels, out_channels=feature))
            in_channels = feature
        
        # Upsampling in UNET
        for feature in reversed(features):
            self.ups.append(nn.ConvTranspose2d(in_channels=feature*2, out_channels=feature, kernel_size=2, stride=2))
            self.ups.append(DoubleConv(in_channels=feature*2, out_channels=feature))

        # Bottle neck of the network
        self.bottleneck = DoubleConv(in_channels=features[-1], out_channels=features[-1]*2)        
        self.final_conv = nn.Conv2d(in_channels=features[0], out_channels=out_channels, kernel_size=1)
    
    def forward(self, x):
        skip_connections = []

        # Apply downsampling
        for down in self.downs:
            x = down(x)
            skip_connections.append(x)
            x = self.pool(x)
        
        x = self.bottleneck(x)
        skip_connections = skip_connections[::-1]

        for idx in range(0,len(self.ups),2):
            x = self.ups[idx](x)
            skip_connection = skip_connections[idx//2]

            if x.shape != skip_connection.shape:
                x = TF.resize(x,size=skip_connection.shape[2:])
            
            concat_skip = torch.cat((skip_connection,x), dim=1)
            x = self.ups[idx+1](concat_skip)
        
        #print(x.shape)
        return self.final_conv(x)
