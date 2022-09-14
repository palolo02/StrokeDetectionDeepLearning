# https://github.com/mmfrxx/X-net/tree/master/Xnet



import torch.nn as nn
import torch
from torch import autograd
from PIL import Image
from torchvision import transforms
import numpy as np


class SingleUnitXblock(nn.Module):
    """Basic network structure for X block"""
    def __init__(self, in_ch, out_ch):
        super(SingleUnitXblock, self).__init__()
        # self.selv_step_1 = nn.Conv2d(in_channels=in_ch, out_channels=out_ch, kernel_size=3, padding=1, groups=in_ch)
        # self.selv_step_2 = nn.BatchNorm2d(out_ch)
        # self.selv_step_3 = nn.ReLU()
        # self.selv_step_4 = nn.Conv2d(in_channels=out_ch, out_channels=out_ch, kernel_size=1, padding=0, groups=in_ch)
        # self.selv_step_5 = nn.BatchNorm2d(out_ch)
        # self.selv_step_6 = nn.ReLU()
        # DepthWiseSeparableConvolution
        self.conv = nn.Sequential(
            # DepthWise           
            nn.Conv2d(in_channels=in_ch, out_channels=out_ch, kernel_size=3, padding=1, groups=in_ch),            
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),            
            # PointWise
            nn.Conv2d(in_channels=out_ch, out_channels=out_ch, kernel_size=1, padding=0),            
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
        )

    def forward(self, input):
        # print(f"Net execution!\n {input.shape}")    
        # result = self.selv_step_1(input)
        # print(f"Step 1: {result.shape}")
        # result = self.selv_step_2(result)
        # print(f"Step 2: {result.shape}")
        # result = self.selv_step_3(result)
        # print(f"Step 3: {result.shape}")        
        # result = self.selv_step_4(result)
        # print(f"Step 4: {result.shape}")
        # result = self.selv_step_5(result)
        # print(f"Step 5: {result.shape}")
        # result = self.selv_step_6(result)
        # print(f"Step 6: {result.shape}")
        

        result = self.conv(input)
        print(f"Single Unit {input.shape} | after convolutions: {result.shape}")
        return result



class GroupSingleUnitXblock(nn.Module):
    """ 3 instances of single units of the X block """
    def __init__(self, in_ch, out_ch):
        super(GroupSingleUnitXblock, self).__init__()
        self.num_channels = 64
        self.block_1 = SingleUnitXblock(in_ch=in_ch, out_ch=self.num_channels)
        self.block_2 = SingleUnitXblock(in_ch=self.num_channels, out_ch=self.num_channels*2)            
        self.block_3 = SingleUnitXblock(in_ch=self.num_channels*2, out_ch=out_ch)            

    def forward(self, x):
        # inner part x block
        print("===== Group of x blocks =====")
        x = self.block_1(x)
        x = self.block_2(x)
        x = self.block_3(x)
        # for _ in range(3):
        #     print(x.shape)
        #     x = self.block(x)
        return x


class Xnet(nn.Module):
    def __init__(self, in_ch, out_ch):
        super(Xnet, self).__init__()
        base_filter_num = 64

        # ====== X NET ===========                (1)         (64)
        self.conv_down_1 = GroupSingleUnitXblock(in_ch, out_ch)
        self.pool1 = nn.MaxPool2d(2)
        # self.conv_down_2 = xblock(base_filter_num, base_filter_num*2)
        # self.pool2 = nn.MaxPool2d(2)
        # self.conv_down_3 = xblock(base_filter_num*2, base_filter_num*4)
        # self.pool3 = nn.MaxPool2d(2)
        # self.conv_bottom = xblock(base_filter_num*4, base_filter_num*8)
                
        # self.upsample_1 = nn.ConvTranspose2d(base_filter_num*8, base_filter_num*4, kernel_size=2, stride=2)
        # self.skip_conn_1 = skipConnection(base_filter_num*8, base_filter_num*4)
        # self.upsample_2 = nn.ConvTranspose2d(base_filter_num*4, base_filter_num*2, kernel_size=2, stride=2)
        # self.skip_conn_2 = skipConnection(base_filter_num*4, base_filter_num*2)
        # self.upsample_3 = nn.ConvTranspose2d(base_filter_num*2, base_filter_num, kernel_size=2, stride=2)
        # self.skip_conn_3 = skipConnection(base_filter_num*2, base_filter_num)
        # self.conv_out = nn.Conv2d(base_filter_num, out_ch, 1)

    def forward(self, x):
        ###down########   
        print("X Net execution")     
        down_1 = self.conv_down_1(x)
        print(down_1.shape)
        pool_1 = self.pool1(down_1)
        print(pool_1.shape)
        # down_2 = self.conv_down_2(pool_1)
        # pool_2 = self.pool2(down_2)
        # down_3 = self.conv_down_3(pool_2)
        # pool_3 = self.pool3(down_3)
        # bottom = self.conv_bottom(pool_3)

        # up_1 = self.upsample_1(bottom)
        # merge1 = torch.cat([up_1, down_3], dim=1)
        # #print(merge1.shape)
        # up_1_out = self.skip_conn_1(merge1)
        # up_2 = self.upsample_2(up_1_out)
        # merge2 = torch.cat([up_2, down_2], dim=1)
        # #print(merge2.shape)
        # up_2_out = self.skip_conn_2(merge2)
        # up_3 = self.upsample_3(up_2_out)
        # merge3 = torch.cat([up_3, down_1], dim=1)
        # #print(merge3.shape)
        # up_3_out = self.skip_conn_3(merge3)
        # end_out = self.conv_out(up_3_out)
        #print(end_out.shape)
        #out = nn.Sigmoid()(end_out)
        #print(out.shape)
        #out = nn.Sigmoid(end_out)

        return pool_1

    
import matplotlib.pyplot as plt

def testDepthWiseConv():
    #from torch.nn import Conv2d

    conv = nn.Conv2d(in_channels=1, out_channels=1, kernel_size=3)
    params = sum(p.numel() for p in conv.parameters() if p.requires_grad)
    n = 10
    x = torch.rand(1, 1, 224, 224)
    out = conv(x)

    depth_conv = nn.Conv2d(in_channels=1, out_channels=1, kernel_size=3, groups=1)
    point_conv = nn.Conv2d(in_channels=1, out_channels=1, kernel_size=1)

    depthwise_separable_conv = torch.nn.Sequential(depth_conv, point_conv)
    params_depthwise = sum(p.numel() for p in depthwise_separable_conv.parameters() if p.requires_grad)

    out_depthwise = depthwise_separable_conv(x)

    print(f"The standard convolution uses {params} parameters.")
    print(f"The depthwise separable convolution uses {params_depthwise} parameters.")

    assert out.shape == out_depthwise.shape, "Size mismatch"


def testModel3Channels():
    x = torch.rand((1,3,256,256))            
    #print(x.shape)
    model = Xnet(in_ch=3, out_ch=1)
    pred = model(x)
    print(pred.shape)

def tryoutNumbers():
    numbers = []
    for n in range(256,0,-1):
        try:
            x = torch.rand((1,1,n,n))            
            print(f"Input shape: {x.shape}")
            #model = UnetDeep3(in_ch=1, out_ch=1)
            model = xblock(in_ch=1, out_ch=1)            
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
    with Image.open(img_file).convert("RGB") as input_image:
        input_image.show()
        input_image = np.array(input_image)
        print(input_image.shape)
    #x = torch.randn((1, 1, 360, 360))    
    #plt.plot(x[0,0,:,:])
    #plt.show()    
    val_transformations = transforms.Compose(
        [
            transforms.ToPILImage(),
            #transforms.Grayscale(),
            transforms.Resize(size=(224,224)),
            transforms.ToTensor()
        ]
    )
    final_input_img = val_transformations(input_image)
    print(type(final_input_img))
    print(final_input_img.shape)
    # # Adding batch of 1
    final_input_img = torch.unsqueeze(final_input_img, dim=0)
    print(f"Shape: {final_input_img.shape} Initial mage Sum: {torch.sum(final_input_img)} ")
    with torch.no_grad():
        #model = SingleUnitXblock(in_ch=1, out_ch=1)
        #model = GroupSingleUnitXblock(in_ch=3, out_ch=3)
        model = Xnet(in_ch=3, out_ch=3)
        #print(model.shape)
        pred = model(final_input_img)
        print(pred.shape)
        pred = torch.squeeze(pred, dim=0)
        pred = pred.numpy()
        pred = np.transpose(pred, (1,2,0))
        print(f"{pred.shape} and {type(pred)}")
        print(f"Max:{np.max(pred)} Min: {np.min(pred)}: Sum:{np.sum(pred)}")
        #print(final_input_img.shape)
        #assert pred.shape == final_input_img.shape
        plt.imshow(pred, cmap="gray")
        plt.show()
        plt.close()

if __name__=="__main__":
    #testModel3Channels()
    #testSingleCNN()
    testDepthWiseConv()