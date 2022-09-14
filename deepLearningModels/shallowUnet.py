import torch.nn as nn
import torch
from torch import autograd
from PIL import Image
from torchvision import transforms
import numpy as np
#network with 3 conv-and-deconv steps used in paper

class pair_convolutions(nn.Module):
    def __init__(self, in_ch, out_ch):
        super(pair_convolutions, self).__init__()
        self.conv = nn.Sequential(
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

    def forward(self, input):
        return self.conv(input)

class final_convolutions(nn.Module):
    def __init__(self, in_ch, out_ch):
        super(final_convolutions, self).__init__()
        self.conv = nn.Sequential(
            # Level 1
            nn.Conv2d(in_channels=in_ch, out_channels=out_ch, kernel_size=3, padding=1),            
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.2)            
        )

    def forward(self, input):
        return self.conv(input)

class ShallowUNet(nn.Module):
    def __init__(self, in_ch, out_ch):
        super(ShallowUNet, self).__init__()
        base_filter_num = 32
        self.conv_down_1 = pair_convolutions(in_ch, base_filter_num) # 1, 32
        self.pool1 = nn.MaxPool2d(2)

        self.conv_down_2 = pair_convolutions(base_filter_num, base_filter_num*2) # 32, 64
        self.pool2 = nn.MaxPool2d(2)

        self.conv_down_3 = pair_convolutions(base_filter_num*2, base_filter_num*4) # 64, 128        
        self.conv_bottom = pair_convolutions(base_filter_num*4, base_filter_num*8) # 128, 256
        
        self.conv_down_4 = pair_convolutions(base_filter_num*4, base_filter_num) # 128, 32

        self.same_conv_up = nn.ConvTranspose2d(base_filter_num*4, base_filter_num*4, kernel_size=2, stride=2) # 128, 128
        self.same_conv_up_32 = nn.ConvTranspose2d(base_filter_num, base_filter_num, kernel_size=2, stride=2) # 32, 32
        self.conv_down_5 = pair_convolutions(base_filter_num*2, base_filter_num) # 64, 32
        self.last_conv = final_convolutions(base_filter_num, 1) # 64, 32
        
        self.upsample_2 = nn.ConvTranspose2d(base_filter_num*4, base_filter_num*2, kernel_size=2, stride=2) # 128, 64
        
        self.conv_out = nn.Conv2d(base_filter_num, out_ch, 1) # 32, 1

    def forward(self, x):
        ###down########        
        down_1 = self.conv_down_1(x)                # [1, 1, 256, 256]  -> [1, 32, 256, 256]
        pool_1 = self.pool1(down_1)                 # [1, 32, 256, 256] -> [1, 32, 128, 128]

        down_2 = self.conv_down_2(pool_1)           # [1, 32, 128, 128] -> [1, 64, 128, 128] 
        pool_2 = self.pool2(down_2)                 # [1, 64, 128, 128] -> [1, 64, 64, 64]

        down_3 = self.conv_down_3(pool_2)           # [1, 64, 64, 64] ->   [1, 128, 64, 64]        
        down_3 = self.upsample_2(down_3)            # [1, 128, 64, 64] ->  [1, 64, 128, 128]
        
        merge2 = torch.cat([down_2, down_3], dim=1) # [1, 64, 128, 128], [1, 64, 128, 128] = [1, 128, 128, 128]
        
        up_2_out = self.conv_down_4(merge2)         # [1, 128, 128, 128] -> [1, 32, 128, 128]
        up_3 = self.same_conv_up_32(up_2_out)       # [1, 32, 128, 128] -> [1, 32, 256, 256]

        merge3 = torch.cat([up_3, down_1], dim=1)   # [1, 32, 256, 256], [1, 32, 256, 256] = [1, 64, 256, 256]
                
        up_3_out = self.conv_down_5(merge3)         # [1, 64, 256, 256] -> [1, 32, 256, 256]
        end_out = self.conv_out(up_3_out)          # [1, 32, 256, 256] -> [1, 1, 256, 256]        

        return end_out

    
import matplotlib.pyplot as plt

def testModel3Channels():
    x = torch.rand((1,1,256,256))        
    #print(x.shape)
    model = ShallowUNet(in_ch=1, out_ch=1)
    pred = model(x)
    print(pred.shape)

def tryoutNumbers():
    numbers = []
    for n in range(256,0,-1):
        try:
            x = torch.rand((1,1,n,n))            
            print(f"Input shape: {x.shape}")
            #model = UnetDeep3(in_ch=1, out_ch=1)
            model = pair_convolutions(in_ch=1, out_ch=1)            
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
        model = ShallowUNet(in_ch=1, out_ch=1)
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
    


if __name__=="__main__":
    testModel3Channels()
    #testSingleCNN()