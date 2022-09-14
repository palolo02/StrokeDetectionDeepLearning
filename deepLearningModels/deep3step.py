import torch.nn as nn
import torch
from torch import autograd
from PIL import Image
from torchvision import transforms
import numpy as np
#network with 3 conv-and-deconv steps used in paper

class one_step_conv(nn.Module):
    def __init__(self, in_ch, out_ch):
        super(one_step_conv, self).__init__()
        self.conv = nn.Sequential(
            # Level 1
            nn.Conv2d(in_channels=in_ch, out_channels=out_ch, kernel_size=3, padding=1),            
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
            # Level 2            
            nn.Conv2d(in_channels=out_ch, out_channels=out_ch, kernel_size=3, padding=1),            
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.2)   
        )

    def forward(self, input):
        return self.conv(input)


class UnetDeep3(nn.Module):
    def __init__(self, in_ch, out_ch):
        super(UnetDeep3, self).__init__()
        base_filter_num = 64
        self.conv_down_1 = one_step_conv(in_ch, base_filter_num)
        self.pool1 = nn.MaxPool2d(2)
        self.conv_down_2 = one_step_conv(base_filter_num, base_filter_num*2)
        self.pool2 = nn.MaxPool2d(2)
        self.conv_down_3 = one_step_conv(base_filter_num*2, base_filter_num*4)
        self.pool3 = nn.MaxPool2d(2)
        self.conv_bottom = one_step_conv(base_filter_num*4, base_filter_num*8)
        
        
        self.upsample_1 = nn.ConvTranspose2d(base_filter_num*8, base_filter_num*4, kernel_size=2, stride=2)
        self.conv_up_1 = one_step_conv(base_filter_num*8, base_filter_num*4)
        self.upsample_2 = nn.ConvTranspose2d(base_filter_num*4, base_filter_num*2, kernel_size=2, stride=2)
        self.conv_up_2 = one_step_conv(base_filter_num*4, base_filter_num*2)
        self.upsample_3 = nn.ConvTranspose2d(base_filter_num*2, base_filter_num, kernel_size=2, stride=2)
        self.conv_up_3 = one_step_conv(base_filter_num*2, base_filter_num)
        self.conv_out = nn.Conv2d(base_filter_num, out_ch, 1)

    def forward(self, x):
        ###down########        
        down_1 = self.conv_down_1(x)
        pool_1 = self.pool1(down_1)
        down_2 = self.conv_down_2(pool_1)
        pool_2 = self.pool2(down_2)
        down_3 = self.conv_down_3(pool_2)
        pool_3 = self.pool3(down_3)
        bottom = self.conv_bottom(pool_3)

        up_1 = self.upsample_1(bottom)
        merge1 = torch.cat([up_1, down_3], dim=1)
        #print(merge1.shape)
        up_1_out = self.conv_up_1(merge1)
        up_2 = self.upsample_2(up_1_out)
        merge2 = torch.cat([up_2, down_2], dim=1)
        #print(merge2.shape)
        up_2_out = self.conv_up_2(merge2)
        up_3 = self.upsample_3(up_2_out)
        merge3 = torch.cat([up_3, down_1], dim=1)
        #print(merge3.shape)
        up_3_out = self.conv_up_3(merge3)
        end_out = self.conv_out(up_3_out)
        # Applying classification in the output layer        
        out = nn.Sigmoid()(end_out)
        return out

    
import matplotlib.pyplot as plt

def testModel3Channels():
    x = torch.rand((1,3,256,256))            
    #print(x.shape)
    model = UnetDeep3(in_ch=3, out_ch=1)
    pred = model(x)
    print(pred.shape)

def tryoutNumbers():
    numbers = []
    for n in range(256,0,-1):
        try:
            x = torch.rand((1,1,n,n))            
            print(f"Input shape: {x.shape}")
            #model = UnetDeep3(in_ch=1, out_ch=1)
            model = one_step_conv(in_ch=1, out_ch=1)            
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
        model = UnetDeep3(in_ch=1, out_ch=1)
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
    #testModel3Channels()
    testSingleCNN()