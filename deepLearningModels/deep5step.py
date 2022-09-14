import torch.nn as nn
import torch.nn as nn
import torch
from torch import autograd
#network with 3 conv-and-deconv steps used in paper



class one_step_conv(nn.Module):
    """ Network with 5 layers of depth appying 1 convolution """
    def __init__(self, in_ch, out_ch):
        super(one_step_conv, self).__init__()
        self.conv = nn.Sequential(
            # Level 1
            nn.Conv2d(in_channels=in_ch, out_channels=out_ch, kernel_size=5, padding=2),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
            #nn.Dropout(p = 0.3),      
            
            # level 2
            nn.Conv2d(in_channels=out_ch, out_channels=out_ch, kernel_size=5, padding=2),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),                        
        )

    def forward(self, x):
        return self.conv(x)

# class one_step_conv(nn.Module):
#     def __init__(self, in_ch, out_ch):
#         super(one_step_conv, self).__init__()
#         self.conv = nn.Sequential(            
#             nn.Conv2d(in_channels=in_ch, out_channels=out_ch, kernel_size=3, padding=1),
#             nn.ReLU(inplace=True),
#             nn.BatchNorm2d(out_ch),
#             nn.Dropout(p = 0.1)
            
#         )

#     def forward(self, input):
#         return self.conv(input)

class Custom5LayersUnet(nn.Module):
    def __init__(self, in_ch, out_ch):
        super(Custom5LayersUnet, self).__init__()
        # Downsampling        
        self.conv_down_1 = one_step_conv(1, 64)
        self.pool1 = nn.MaxPool2d(2)
        self.conv_down_2 = one_step_conv(64, 128)
        self.pool2 = nn.MaxPool2d(2)
        self.conv_down_3 = one_step_conv(128, 256)
        self.pool3 = nn.MaxPool2d(2)
        self.conv_down_4 = one_step_conv(256, 512)
        self.pool4 = nn.MaxPool2d(2)
        self.conv_down_5 = one_step_conv(512, 1024)
        self.pool5 = nn.MaxPool2d(2)
        
        self.conv_bottom = one_step_conv(1024, 2048)
        
        # Upsampling        
        self.upsample_1 = nn.ConvTranspose2d(2048, 1024, 2, stride=2)
        self.conv_up_1 = one_step_conv(2048, 1024)
        self.upsample_2 = nn.ConvTranspose2d(1024, 512, 2, stride=2)
        self.conv_up_2 = one_step_conv(1024, 512)
        self.upsample_3 = nn.ConvTranspose2d(512, 256, 2, stride=2)
        self.conv_up_3 = one_step_conv(512, 256)
        self.upsample_4 = nn.ConvTranspose2d(256, 128, 2, stride=2)
        self.conv_up_4 = one_step_conv(256, 128)
        self.upsample_5 = nn.ConvTranspose2d(128, 64, 2, stride=2)
        self.conv_up_5 = one_step_conv(128, 64)

        self.conv_out = nn.Conv2d(64, 1, 1)
        

    def forward(self, x):
        ###down########
        down_1 = self.conv_down_1(x)
        pool_1 = self.pool1(down_1)
        #print(f"down_1 : {down_1.shape} pool_1: {pool_1.shape}")

        down_2 = self.conv_down_2(pool_1)
        pool_2 = self.pool2(down_2)
        #print(f"down_2 : {down_2.shape} pool_2: {pool_2.shape}")

        down_3 = self.conv_down_3(pool_2)
        pool_3 = self.pool3(down_3)
        #print(f"down_3 : {down_3.shape} pool_3: {pool_3.shape}")

        down_4 = self.conv_down_4(pool_3)
        pool_4 = self.pool4(down_4)
        #print(f"down_4 : {down_4.shape} pool_4: {pool_4.shape}")

        down_5 = self.conv_down_5(pool_4)
        pool_5 = self.pool5(down_5)
        #print(f"down_5 : {down_5.shape} pool_5: {pool_5.shape}")

        bottom = self.conv_bottom(pool_5)
        #print(f"bottom : {bottom.shape}")

        up_1 = self.upsample_1(bottom)
        merge1 = torch.cat([up_1, down_5], dim=1)
        up_1_out = self.conv_up_1(merge1)
        #print(f"up_1 : {up_1.shape} merge1: {merge1.shape} up_1_out: {up_1_out.shape}")

        up_2 = self.upsample_2(up_1_out)
        merge2 = torch.cat([up_2, down_4], dim=1)
        up_2_out = self.conv_up_2(merge2)
        #print(f"up_2 : {up_2.shape} merge2: {merge2.shape} up_2_out: {up_2_out.shape}")

        up_3 = self.upsample_3(up_2_out)
        merge3 = torch.cat([up_3, down_3], dim=1)
        up_3_out = self.conv_up_3(merge3)
        #print(f"up_3 : {up_3.shape} merge3: {merge3.shape} up_3_out: {up_3_out.shape}")

        up_4 = self.upsample_4(up_3_out)        
        merge4 = torch.cat([up_4, down_2], dim=1)        
        up_4_out = self.conv_up_4(merge4)
        #print(f"up_4 : {up_4.shape} merge4: {merge4.shape} up_4_out: {up_4_out.shape}")

        up_5 = self.upsample_5(up_4_out)
        merge5 = torch.cat([up_5, down_1], dim=1)
        up_5_out = self.conv_up_5(merge5)
        #print(f"up_5 : {up_5.shape} merge5: {merge5.shape} up_5_out: {up_5_out.shape}")

        end_out = self.conv_out(up_5_out)
        out = nn.Sigmoid()(end_out)      

        return out
