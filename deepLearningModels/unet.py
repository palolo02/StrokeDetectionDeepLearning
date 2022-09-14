
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
            nn.Conv2d(in_channels=out_channels, out_channels=out_channels, kernel_size=3,stride=1,padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            #nn.Dropout2d(0.25),
            nn.ReLU(inplace=True),            
        )
    
    def forward(self, x):        
        return self.conv(x)

class UNET(nn.Module):
    """Class that represents the UNet Architecture """
    def __init__(self, in_channels=1, out_channels=1, features =[64,128,256,512]) -> None:
        super(UNET, self).__init__()
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
        x = self.final_conv(x)
        #x = nn.Sigmoid()(x)

        return x

class TestNet(nn.Module):
    def __init__(self, in_channels=3, out_channels=1):
        super(TestNet,self).__init__()
        self.layers  = nn.Sequential(
            nn.Conv2d(in_channels, 16, kernel_size=(11, 11), stride=(4, 4), padding=(2, 2)),
            nn.BatchNorm2d(16),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1, dilation=1, ceil_mode=False),
            nn.Conv2d(16, 32, kernel_size=(5, 5), stride=(1, 1), padding=(2, 2)),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1, dilation=1, ceil_mode=False),
            nn.Conv2d(32, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 128, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1, dilation=1, ceil_mode=False),            
            nn.Conv2d(256, out_channels, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)),
            nn.BatchNorm2d(out_channels),            
            nn.ReLU(inplace=True),
        )
    
    def forward(self, x):
        x = self.layers(x)
        return x



# Run test
def test():
    with Image.open(img_file) as input_image:
        input_image.show()
    #x = torch.randn((1, 1, 360, 360))    
    #plt.plot(x[0,0,:,:])
    #plt.show()    
    preprocess = transforms.Compose([
        #transforms.Grayscale(),
        transforms.ToTensor(),
        transforms.Normalize(mean=0, std=1),
    ])
    final_input_img = preprocess(input_image)
    # Adding batch of 1
    final_input_img = unsqueeze(final_input_img, dim=0)
    print(final_input_img.shape)
    with torch.no_grad():
        model = UNET(in_channels=3, out_channels=1)
        pred = model(final_input_img)
        print(pred.shape)
        pred = torch.squeeze(pred, dim=1)
        pred = pred.numpy()
        pred = np.transpose(pred, (1,2,0))
        print(f"{pred.shape} and {type(pred)}")
        #print(final_input_img.shape)
        #assert pred.shape == final_input_img.shape
        plt.imshow(pred, cmap="gray")
        plt.show()

def Alexnet():
    model = models.alexnet(pretrained=True)    
    # Keep pretrained weights
    for param in model.parameters():
        param.requires_grad = False
    # change classifier
    #torch.manual_seed(100)
    
    # nn.Conv2d(in_channels, out_channels, 3,1,1, bias=False),
    # nn.BatchNorm2d(out_channels),
    # nn.ReLU(inplace=True),
    model.avgpool = nn.Sequential(
        nn.Conv2d(256, 1024, 3,1,1, bias=False),
        nn.BatchNorm2d(1024)
        # nn.ReLU(inplace=True),
        # nn.Conv2d(2, 1024, 3,1,1, bias=False),
        # nn.BatchNorm2d(1024), # Optimize the training process of the network by reducing 
        # nn.ReLU(inplace=True),  
        # nn.Conv2d(1024, 1, 3,1,1, bias=False),
        # nn.BatchNorm2d(1),
        # nn.ReLU(inplace=True)  
    )
    model.classifier = nn.Sequential(
        nn.Conv2d(1024, 1, 3,1,1, bias=False)
        # nn.ReLU(inplace=True),
        # nn.Conv2d(2, 1024, 3,1,1, bias=False),
        # nn.BatchNorm2d(1024), # Optimize the training process of the network by reducing 
        # nn.ReLU(inplace=True),  
        # nn.Conv2d(1024, 1, 3,1,1, bias=False),
        # nn.BatchNorm2d(1),
        # nn.ReLU(inplace=True)  
    )
    
    print(model)
    print(f"=====\n{model}")
    
    with Image.open(img_file) as input_image:
        input_image.show()
    #x = torch.randn((1, 1, 360, 360))    
    #plt.plot(x[0,0,:,:])
    #plt.show()    
    preprocess = transforms.Compose([
        #transforms.Grayscale(),
        transforms.ToTensor(),
        transforms.Normalize(mean=0, std=1),
    ])
    final_input_img = preprocess(input_image)
    # Adding batch of 1
    final_input_img = unsqueeze(final_input_img, dim=0)
    print(final_input_img.shape)
    with torch.no_grad():
        #model = UNET(in_channels=3, out_channels=1)
        pred = model(final_input_img)
        print(pred.shape)
        # pred = torch.squeeze(pred, dim=1)
        # pred = pred.numpy()
        # pred = np.transpose(pred, (1,2,0))
        # print(f"{pred.shape} and {type(pred)}")
        # #print(final_input_img.shape)
        # #assert pred.shape == final_input_img.shape
        plt.imshow(pred, cmap="gray")  
        plt.show()


def testCNN():
    with Image.open(img_file) as input_image:
        input_image.show()
    
    preprocess = transforms.Compose([
        #transforms.Grayscale(),
        transforms.ToTensor(),
        transforms.Normalize(mean=0, std=1),
    ])
    
    final_input_img = preprocess(input_image)
    # Adding batch of 1
    final_input_img = unsqueeze(final_input_img, dim=0)
    print(final_input_img.shape)
    
    with torch.no_grad():
        model = TestNet(3,1)
        #UNET(in_channels=3, out_channels=1)
        pred = model(final_input_img)
        print(pred.shape)
        pred = torch.squeeze(pred, dim=1)
        pred = pred.numpy()
        pred = np.transpose(pred, (1,2,0))
        #print(f"{pred.shape} and {type(pred)}")
        #print(final_input_img.shape)
        #assert pred.shape == final_input_img.shape
        plt.imshow(pred, cmap="gray")
        plt.show()

def maskrcnn():
    # load model
    model = maskrcnn_resnet50_fpn(pretrained=True)
    # set to evaluation mode
    model.eval()
    # load COCO category names
    classes = [
        '__background__', 'strke'
    ]
    print(model)
    with Image.open(img_file) as input_image:
        input_image.show()
    #x = torch.randn((1, 1, 360, 360))    
    #plt.plot(x[0,0,:,:])
    #plt.show()    
    preprocess = transforms.Compose([
        #transforms.Grayscale(),
        transforms.ToTensor(),
        transforms.Normalize(mean=0, std=1),
    ])
    final_input_img = preprocess(input_image)
    # Adding batch of 1
    final_input_img = unsqueeze(final_input_img, dim=0)
    print(final_input_img.shape)
    with torch.no_grad():
        #model = UNET(in_channels=3, out_channels=1)
        pred = model(final_input_img)
        print(pred["masks"])
        # pred = torch.squeeze(pred, dim=1)
        # pred = pred.numpy()
        # pred = np.transpose(pred, (1,2,0))
        # print(f"{pred.shape} and {type(pred)}")
        # #print(final_input_img.shape)
        # #assert pred.shape == final_input_img.shape
        plt.imshow(pred, cmap="gray")  
        plt.show()


def plot_three_imgs():
    with Image.open(img_file) as input_image:
            input_image.show()
        #x = torch.randn((1, 1, 360, 360))    
        #plt.plot(x[0,0,:,:])
        #plt.show()    
    preprocess = transforms.Compose([
            #transforms.Grayscale(),
            transforms.ToTensor(),
            transforms.Normalize(mean=0, std=1),
        ])
    img_1 = preprocess(input_image)
    img_2 = preprocess(input_image)
    img_3 = preprocess(input_image)
    result = torch.cat((img_1,img_2,img_3), dim=1)
    print(result.shape)
    # def show(img):
    #         npimg = img.numpy()
    #         plt.imshow(np.transpose(npimg, (1,2,0)), interpolation='nearest')

    # w = torch.randn(3,3,640,640)
    # print(w.shape)
    grid = torchvision.utils.make_grid(result, nrow=2, padding=100)
    # show(grid)
    torchvision.utils.save_image(grid, f"test.png")
    # torch.cat((x, x, x), 0)
   
    # grid_img = torchvision.utils.make_grid(batch_tensor, nrow=5)
    # torchvision.utils.save_image(final_input_img, f"{self.folder_output}/pred_{batch_idx}_{date.today()}.png") 
    # grid_img.shape
    # torch.Size([3, 518, 1292])


if __name__=="__main__":
    #test()
    #Alexnet()
    #testCNN()
    #maskrcnn()
    plot_three_imgs()
