from PIL import Image
import torch
from torchvision.transforms import transforms
from readConfig import configuration
import torchvision.transforms.functional as TF
import random
import numpy as np
from readConfig import IMAGE_WIDTH, IMAGE_HEIGHT, DEPTH

from monai.transforms import (        
    EnsureChannelFirstd,
    Compose,
    CropForegroundd,
    LoadImaged,
    RandSpatialCropd,
    NormalizeIntensityd,        
    ScaleIntensityRanged,
    EnsureTyped,
    Resized,
    RandRotate90d,
    ToTensord
)

def  stroke_train_img_mask_transforms(image,segmentation):
    
    # # Transform to Images
    # if type(image) == Image.Image:        
    #     image = transforms.ToPILImage()(image)
    #     segmentation = transforms.ToPILImage()(segmentation)
    #     # Apply Gray Scale
    #     image = transforms.Grayscale()(image)
    #     segmentation = transforms.Grayscale()(segmentation)      
    
    # Change to tensor   and add channel if there is no any in the image
    if type(image) == np.ndarray:
        image = torch.from_numpy(image)
        segmentation = torch.from_numpy(segmentation)
        if len(image.shape) == 2:
            image = torch.unsqueeze(image, dim=0)
            segmentation = torch.unsqueeze(segmentation, dim=0)

    # Adjust intensity
    # image = ScaleIntensity()(image)
    # segmentation = ScaleIntensity()(segmentation)
    #image = transforms.Grayscale()(image)
    #segmentation = transforms.Grayscale()(segmentation)

    # Only for image    
    image = TF.adjust_gamma(img=image, gamma=1)
    image = TF.adjust_contrast(image,2)

    # random crop appplied to both image and segmentation mask
    if type(image) == Image.Image: 
        w, h = image.size
    else:
        w, h = image.shape[1], image.shape[2]    
    x1 = random.randint(0, w)
    y1 = random.randint(0, h)
    image = TF.crop(image,x1,y1,x1 + IMAGE_WIDTH,y1 + IMAGE_HEIGHT)
    segmentation = TF.crop(segmentation,x1,y1,x1 + IMAGE_WIDTH,y1 + IMAGE_HEIGHT)
    
    # Resize
    image = transforms.Resize((IMAGE_WIDTH,IMAGE_HEIGHT))(image)
    segmentation = transforms.Resize((IMAGE_WIDTH,IMAGE_HEIGHT))(segmentation)

    # Rotation

    
    # to tensor
    if type(image) == Image.Image: 
        image = transforms.ToTensor()(image)
        segmentation = transforms.ToTensor()(segmentation)

    return image, segmentation


stroke_train_transformations = transforms.Compose(
        [
            transforms.ToPILImage(),
            transforms.Grayscale(),
            #transforms.GaussianBlur(kernel_size=(3, 3), sigma=(0.1, 5)), 
            transforms.Resize(size=(IMAGE_WIDTH,IMAGE_HEIGHT)),            
            transforms.ToTensor(),
        ]
    )

stroke_val_transformations = transforms.Compose(
    [
        transforms.ToPILImage(),
        transforms.Grayscale(),
        transforms.Resize(size=(IMAGE_WIDTH,IMAGE_HEIGHT)),
        transforms.ToTensor()
    ]
)

stroke_mask_transformations = transforms.Compose([
            transforms.ToPILImage(),
            transforms.Grayscale(),
            transforms.Resize(size=(IMAGE_WIDTH,IMAGE_HEIGHT)),
            transforms.ToTensor(),
        ])



monai_train_transforms = Compose(
    [
        LoadImaged(keys=["image", "mask"], image_only=True),        
        #EnsureChannelFirstd(keys=["image","mask"]),        
        # Orientationd(keys=["image", "mask"], axcodes="RAS"),
        # Spacingd(keys=["image", "mask"], pixdim=(
        #     1.5, 1.5, 2.0), mode=("bilinear", "nearest")),
        # NormalizeIntensityd(keys=["image"]),
        # ScaleIntensityRanged(
        #     keys=["image"], a_min=-57, a_max=164,
        #     b_min=0.0, b_max=1.0, clip=True,
        # ),
        #RandSpatialCropd(keys=["image", "mask"], roi_size=(IMAGE_WIDTH,IMAGE_HEIGHT)),
        RandRotate90d(keys=["image", "mask"], prob=0.5, spatial_axes=[0, 1]),
        # CropForegroundd(keys=["image", "mask"], source_key="image"),
        Resized(keys=["image", "mask"], spatial_size=(IMAGE_WIDTH,IMAGE_HEIGHT)),        
        EnsureTyped(keys=["image", "mask"]),
        #ToTensord(keys=["image", "mask"])   
    ]
)
monai_val_transforms = Compose(
    [
        LoadImaged(keys=["image", "mask"]),
        EnsureChannelFirstd(keys=["image","mask"]),
        # Orientationd(keys=["image", "mask"], axcodes="RAS"),
        # Spacingd(keys=["image", "mask"], pixdim=(
        #     1.5, 1.5, 2.0), mode=("bilinear", "nearest")),
        ScaleIntensityRanged(
            keys=["image"], a_min=-57, a_max=164,
            b_min=0.0, b_max=1.0, clip=True,
        ),
        CropForegroundd(keys=["image", "mask"], source_key="image"),
        Resized(keys=["image", "mask"], spatial_size=(IMAGE_WIDTH,IMAGE_HEIGHT,DEPTH)),
        EnsureTyped(keys=["image", "mask"]),
    ]
)


class CustomTrainingStrokeTrans(torch.nn.Module):
    def __init__(self):
       super().__init__()

    def __call__(self, img, mask):

        #t = random.choice(self.transforms)
        image = transforms.Grayscale()(img)
        seg = transforms.Grayscale()(mask)

        #t = random.choice(self.transforms)
        # image = [t(img) for t in self.transforms][-1]
        # seg = [t(mask) for t in self.transforms][-1]

        # Adjusting gamma
        image = TF.adjust_gamma(img=image, gamma=1)
        image = TF.adjust_contrast(image,2)

        # Rotating image
        angle_ = random.randint(0, 180)    
        image = TF.rotate(image, angle=angle_)
        seg = TF.rotate(seg, angle=angle_)

        #Random crop
        # if type(image) == Image.Image: 
        #     w, h = image.size
        # else:
        #     w, h = image.shape[1], image.shape[2]
        # x1 = random.randint(0, w)
        # y1 = random.randint(0, h)
        # image = TF.crop(image,x1,y1,x1 + IMAGE_WIDTH,y1 + IMAGE_HEIGHT)
        # seg = TF.crop(seg,x1,y1,x1 + IMAGE_WIDTH,y1 + IMAGE_HEIGHT)

        
        # Gaussian blur
        #random_sigma =  random.randint(1, 5)
        #print(random_sigma)     
        #blurrer = transforms.GaussianBlur(kernel_size=(5, 9), sigma=(0.1, random_sigma))
        #image = blurrer(image)

        # Resize
        image = transforms.Resize((IMAGE_WIDTH,IMAGE_HEIGHT))(image)
        seg = transforms.Resize((IMAGE_WIDTH,IMAGE_HEIGHT))(seg)

        # To tensor
        image =  transforms.ToTensor()(image)
        seg =  transforms.ToTensor()(seg)
        
        return image, seg

class Custom3DTrainingStrokeTrans(torch.nn.Module):
    def __init__(self):
       super().__init__()

    def __call__(self, img, mask):
        # 
        #CustomTrainingStrokeTrans
        return


class CustomTestStrokTrans(torch.nn.Module):
    def __init__(self):
       super().__init__()

    def __call__(self, img, mask=None):

        
        #t = random.choice(self.transforms)
        image = transforms.Grayscale()(img)
        if mask is not None:
            seg = transforms.Grayscale()(mask)
        
        # Adjusting gamma
        image = TF.adjust_gamma(img=image, gamma=1)
        image = TF.adjust_contrast(image,2)               
            
        # Resize
        image = transforms.Resize((IMAGE_WIDTH,IMAGE_HEIGHT))(image)
        if mask is not None:
            seg = transforms.Resize((IMAGE_WIDTH,IMAGE_HEIGHT))(seg)

        # To tensor
        image =  transforms.ToTensor()(image)
        if mask is not None:
            seg =  transforms.ToTensor()(seg)
            
        return image, (seg if mask is not None else None)



class CustomComparisonStrokTrans(torch.nn.Module):
    def __init__(self):
       super().__init__()

    def __call__(self, img):
        
        # Resize
        image = transforms.Resize((IMAGE_WIDTH,IMAGE_HEIGHT))(img)
        image =  transforms.ToTensor()(image)
            
        return image
