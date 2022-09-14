from turtle import down, forward
import torch.nn as nn
import torch
from torch import autograd
from PIL import Image
from torchvision import transforms
import numpy as np
import matplotlib.pyplot as plt
#network with 3 conv-and-deconv steps used in paper

class SingleConvolution(nn.Module):

    def __init__(self, in_channels, out_channels):
        super(SingleConvolution, self).__init__()

        # Defining the hidden layers of the net
        # First hiddden layer
        self.conv = nn.Sequential(
            # convolution
            nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=3, padding=1, stride=1),
            # batch normalization
            nn.BatchNorm2d(num_features=out_channels),
            # activation function
            nn.ReLU(inplace=True),
            # convolution
            nn.Conv2d(in_channels=out_channels, out_channels=out_channels, kernel_size=3, padding=1, stride=1),
            # batch normalization
            nn.BatchNorm2d(num_features=out_channels),
            # activation function
            nn.ReLU(inplace=True),
            # regularization
            nn.Dropout2d(p=0.3) 
        )

    def forward(self, input):
        x = self.conv(input)
        return x

class AttentionBlock(nn.Module):

    def __init__(self, g, x_l, out_channels):
        super(AttentionBlock, self).__init__()
        self.input_g = nn.Sequential(
            nn.Conv2d(in_channels=g, out_channels=out_channels, kernel_size=1, stride=1, padding=0),
            nn.BatchNorm2d(out_channels)
        )
        self.input_x  = nn.Sequential(
            nn.Conv2d(in_channels=x_l, out_channels=out_channels, kernel_size=1, stride=1, padding=0),
            nn.BatchNorm2d(num_features=out_channels)            
        )
        self.psi = nn.Sequential(
            nn.Conv2d(in_channels=out_channels, out_channels=1, kernel_size=1, stride=1, padding=0),
            nn.BatchNorm2d(num_features=1),
            nn.Sigmoid()
        )
        self.relu = nn.ReLU(inplace=True)
    
    def forward(self, g, x):
        g1 = self.input_g(g) # [1, 512, 32, 32])    => [1, 256, 32, 32])
        x1 = self.input_x(x) # [1, 512, 32, 32]     => [1, 256, 32, 32])
        res = g1 + x1
        res = self.relu(res)
        res = self.psi(res) # [1, 1, 32, 32]
        res = x * res
        return res
    
class CustomAttentionUnetv2(nn.Module):

    def __init__(self, in_channels=1, out_channels=1):
        super(CustomAttentionUnetv2, self).__init__()

        base_filter_num = 32
        
        # Encoder
        self.conv_down_1 = SingleConvolution(in_channels=in_channels, out_channels=base_filter_num) # 1 => 64
        self.pool_1 = nn.MaxPool2d(2)

        self.conv_down_2 = SingleConvolution(in_channels=base_filter_num, out_channels=base_filter_num*2) # 64 => 128
        self.pool_2 = nn.MaxPool2d(2)

        self.conv_down_3 = SingleConvolution(in_channels=base_filter_num*2, out_channels=base_filter_num*4) # 128 => 256
        self.pool_3 = nn.MaxPool2d(2)

        self.conv_down_4 = SingleConvolution(in_channels=base_filter_num*4, out_channels=base_filter_num*8) # 256 => 512
        self.pool_4 = nn.MaxPool2d(2)

        self.conv_down_5 = SingleConvolution(in_channels=base_filter_num*8, out_channels=base_filter_num*16) # 512 => 1024
        #self.pool_5 = nn.MaxPool2d(2)

        #self.conv_down_6 = SingleConvolution(in_channels=base_filter_num*16, out_channels=base_filter_num*8) # 512 => 1024

        # Backbone
        #self.conv_bottom = SingleConvolution(base_filter_num*16, base_filter_num*16)   # 1024 => 1024

        # Decoder
        self.upsample_1 = nn.ConvTranspose2d(base_filter_num*16, base_filter_num*8, kernel_size=2, stride=2)    # 1024 => 512
        self.attention_1 = AttentionBlock(g=base_filter_num*8, x_l= base_filter_num*8, out_channels=base_filter_num*4) # 512, 512 => 256
        self.conv_up_1 = SingleConvolution(base_filter_num*16, base_filter_num*8)

        self.upsample_2 = nn.ConvTranspose2d(base_filter_num*8, base_filter_num*4, kernel_size=2, stride=2)
        self.attention_2 = AttentionBlock(g=base_filter_num*4, x_l= base_filter_num*4, out_channels=base_filter_num*2)
        self.conv_up_2 = SingleConvolution(base_filter_num*8, base_filter_num*4)

        self.upsample_3 = nn.ConvTranspose2d(base_filter_num*4, base_filter_num*2, kernel_size=2, stride=2)
        self.attention_3 = AttentionBlock(g=base_filter_num*2, x_l= base_filter_num*2, out_channels=base_filter_num)
        self.conv_up_3 = SingleConvolution(base_filter_num*4, base_filter_num*2)

        self.upsample_4 = nn.ConvTranspose2d(base_filter_num*2, base_filter_num, kernel_size=2, stride=2)
        self.attention_4 = AttentionBlock(g=base_filter_num, x_l= base_filter_num, out_channels=base_filter_num//2)
        self.conv_up_4 = SingleConvolution(base_filter_num*2, base_filter_num)

        # Last layer
        self.conv_out = nn.Conv2d(in_channels=base_filter_num, out_channels=out_channels, kernel_size=1)
    
    def forward(self, input): 
        # Encoder
        down_1 = self.conv_down_1(input)        # [1, 1, 256, 256]      => [1, 64, 256, 256]
        pool_1 = self.pool_1(down_1)            # [1, 64, 256, 256]     => [1, 64, 128, 128]
        down_2 = self.conv_down_2(pool_1)       # [1, 64, 128, 128]     => [1, 128, 128, 128]
        pool_2 = self.pool_2(down_2)            # [1, 128, 128, 128]    => [1, 128, 64, 64]    
        down_3 = self.conv_down_3(pool_2)       # [1, 128, 64, 64]      => [1, 256, 64, 64]    
        pool_3 = self.pool_3(down_3)            # [1, 256, 64, 64]      => [1, 256, 32, 32]    
        down_4 = self.conv_down_4(pool_3)       # [1, 256, 32, 32]      => [1, 512, 32, 32] 
        pool_4 = self.pool_4(down_4)            # [1, 512, 32, 32]      => [1, 512, 16, 16] 
        down_5 = self.conv_down_5(pool_4)       # [1, 512, 16, 16]      => [1, 1024, 16, 16] 
        #pool_5 = self.pool_5(down_5)           # [1, 1024, 16, 16]     => [1, 1024, 8, 8] 

        # Backbone
        #bottom = self.conv_bottom(down_5)           # [1, 1024, 16, 16]       => [1, 1024, 16, 16]

        # Decoder
        up_1 = self.upsample_1(down_5)              # [1, 1024, 16, 16]     => [1, 512, 32, 32]  
        at_1 = self.attention_1(up_1, down_4)       # [1, 512, 32, 32]
        up_1 = torch.cat((at_1, down_4), dim=1)     # [1, 1024, 32, 32] 
        up_1 = self.conv_up_1(up_1)                 # [1, 1024, 32, 32]     => [1, 512, 32, 32]

        up_2 = self.upsample_2(up_1)                # [1, 512, 32, 32]      => [1, 256, 64, 64]
        at_2 = self.attention_2(up_2, down_3)       # [1, 256, 64, 64]      
        up_2 = torch.cat((at_2, down_3), dim=1)     # [1, 512, 64, 64]
        up_2 = self.conv_up_2(up_2)                 # [1, 512, 64, 64]      => [1, 256, 64, 64]

        up_3 = self.upsample_3(up_2)                # [1, 256, 64, 64]      => [1, 128, 128, 128]
        at_3 = self.attention_3(up_3, down_2)       # [1, 128, 128, 128]
        up_3 = torch.cat((at_3, down_2), dim=1)     # [1, 256, 128, 128]
        up_3 = self.conv_up_3(up_3)                 # [1, 256, 128, 128]    => [1, 128, 128, 128]

        up_4 = self.upsample_4(up_3)                # [1, 128, 128, 128] => [1, 64, 256, 256]
        at_4 = self.attention_4(up_4, down_1)       # [1, 64, 256, 256]
        up_4 = torch.cat((at_4, down_1), dim=1)     # [1, 128, 256, 256]
        up_4 = self.conv_up_4(up_4)                 # [1, 128, 256, 256] => [1, 64, 256, 256]

        end_out = self.conv_out(up_4)               # [1, 64, 256, 256] => [1, 1, 256, 256]
        # Applying classification in the output layer        
        out = nn.Sigmoid()(end_out)
        return out


   
import matplotlib.pyplot as plt

def testModel3Channels():
    x = torch.rand((1,1,256,256))            
    #print(x.shape)
    model = CustomAttentionUnetv2(in_channels=1, out_channels=1)
    pred = model(x)
    print(pred.shape)
    print(sum(p.numel() for p in model.parameters() if p.requires_grad))

def tryoutNumbers():
    numbers = []
    for n in range(256,0,-1):
        try:
            x = torch.rand((1,1,n,n))            
            print(f"Input shape: {x.shape}")
            #model = UnetDeep4(in_ch=1, out_ch=1)
            model = SingleConvolution(in_channels=1, out_channels=1)            
            pred = model(x)
            print(f"Prediction shape: {pred.shape}")
            #print(pred.shape)
            numbers.append(n)
        except:
            pass
    print(numbers)
    # plt.plot(pred[0,0,:,:].detach().numpy())
    # plt.show()
    # plt.close()

def testSingleCNN():    
    # Prediction with one image
    img_file = "dataset/validation/images/sub-r004s001_ses/0085.png"
    with Image.open(img_file).convert("L") as input_image:
        input_image.show()
        input_image = np.array(input_image)
    #x = torch.randn((1, 1, 360, 360))    
    #plt.plot(x[0,0,:,:])
    #plt.show()    
    val_transformations = transforms.Compose(
        [
            transforms.ToPILImage(),
            transforms.Grayscale(),
            transforms.Resize(size=(224,224)),
            transforms.ToTensor()
        ]
    )
    final_input_img = val_transformations(input_image)
    print(type(final_input_img))
    print(final_input_img.shape)
    # # Adding batch of 1
    final_input_img = torch.unsqueeze(final_input_img, dim=0)
    print(final_input_img.shape)
    with torch.no_grad():
        #model = one_step_conv(in_ch=1, out_ch=1)
        model = CustomAttentionUnetv2(in_channels=1, out_channels=1)
        pred = model(final_input_img)        
        print(pred.shape)
        print(torch.max(pred))
        print(torch.min(pred))
        pred = (pred > 0.5).float()
        print(torch.max(pred))
        print(torch.min(pred))
        pred = torch.squeeze(pred, dim=1)
        pred = pred.numpy()
        pred = np.transpose(pred, (1,2,0))
        print(f"{pred.shape} and {type(pred)}")
        #print(final_input_img.shape)
        #assert pred.shape == final_input_img.shape
        plt.imshow(pred, cmap="gray")
        plt.show()
        plt.close()
    print(sum(p.numel() for p in model.parameters() if p.requires_grad))


if __name__=="__main__":
    #testModel3Channels()
    testSingleCNN()