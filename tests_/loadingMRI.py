
from unicodedata import name
import matplotlib.pyplot as plt
from sqlalchemy import false
from torch import nn, unsqueeze
import torch
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
from torchvision import transforms
from torchvision.models.detection.mask_rcnn import maskrcnn_resnet50_fpn
import torchvision
import nibabel as nib
import napari
import os
from vnet.vnet import VNet
from vnet.transformations import ResizeImage
import random
from datetime import date


#torch.manual_seed(100)

def loadSingleImage(index, input, masks, viewer):
    """Loading input and masks"""
    # The img is loaded as d,h,w
    input_load =  torch.from_numpy(np.transpose(nib.load(input[index]).get_fdata(), axes=[2,0,1])).float()
    maks_load = torch.from_numpy(np.transpose(nib.load(masks[index]).get_fdata(), axes=[2,0,1])).float()
    #print(f"Input: {input_load.shape} & Mask = {maks_load.shape} Axis in pytorch: [0,1,2] [h, w, d]")

    # Adding original images
    #print(f"Axis in napari: [2,0,1] [d, h, w] => {np.transpose(input_load, axes=[2,0,1]).shape}")
    viewer.add_image(input_load, name=f"Original{index}", visible=False)
    #viewer.add_image(maks_load, name=f"Mask_{index}",visible=False)
    
    # the img as tensor is d, h, w
    #print(f"After first transform Input: {input_load.shape} & Mask = {maks_load.shape}")
    #viewer.add_image(input_load[:,:,:], name="FirstTrans", visible=False) # Good position
    # resize images through depths
    new_img = []
    mask_new = []
    
    list = range(0,input_load.shape[0]) 
    n = 64#input_load.shape[0]
    list = sorted(random.sample(list, n))    
    input_load = torch.unsqueeze(input_load, dim=1)
    maks_load = torch.unsqueeze(maks_load, dim=1)
    #print(f"Shape: {input_load.shape}")
    trans_test = ResizeImage(128)
    
    # Apply transformations        
    for i in list:        
        new_img.append(trans_test(input_load[i,:,:,:]))
        mask_new.append(trans_test(maks_load[i,:,:,:]))          


    # Concatenate all images and add batch dimension
    final_input_img = torch.concat(new_img, dim=0)
    final_maks_img = torch.concat(mask_new, dim=0)
    #print(f"last transform: {final_input_img.shape} and final array {final_input_img.shape} | Mask last transform: {final_maks_img.shape} and final mask array {final_maks_img.shape} ")
    final_input_img = torch.squeeze(final_input_img, dim=1).float()
    final_maks_img = torch.squeeze(final_maks_img, dim=1).float()
    #print(f"adding more dimensions: {final_input_img.shape} and final array {final_input_img.shape} | Mask last transform: {final_maks_img.shape} and final mask array {final_maks_img.shape} ")
    viewer.add_image(final_input_img, name=f"ResizedInput{index}", visible=False) 
    viewer.add_image(final_maks_img, name=f"ResizedMasks{index}", visible=False)

    return {"img": final_input_img, "masks": final_maks_img}

def loadSequence(dir="kaggle_3m/nni/"):
    """ Load all the sequence of images"""
    dir = "kaggle_3m/nni/R045/"
    input_files = []
    masks_files = []
    images = []
    for dirpath, dirnames, filenames in os.walk(dir, topdown=False):
        for name in filenames:       
            if "nii.gz" in name:
                if "mask" in name:
                    masks_files.append(f"{dirpath}/{name}")
                else:
                    input_files.append(f"{dirpath}/{name}")
                #print(f"{dirpath}{name}")
    
    # Adding imges    
    viewer = napari.Viewer()
    print(f"Loading napari to display images of patient 1: {len(input_files)}")
    for i in range(0,len(input_files)):
        pairwise = loadSingleImage(i,input_files, masks_files, viewer)
        images.append(pairwise)
        if i == 1: break
    #napari.run()

    # Running the model
    with torch.no_grad():
        # gray scale imges
        model = VNet()
        #model = VNET(in_channels=1, out_channels=1)
        #input = torch.unsqueeze(images[0]["img"], dim=0)
        #input = torch.rand((1,1,128,128,64), dtype=torch.float32)
        input = torch.rand((1,1,32,128,128), dtype=torch.float32)
        # (N, Cin, D, H, W)
        pred = model(input)
        #pred = torch.squeeze(pred, dim=0)
        print(f"Shape of the prediction: {pred.shape}")
        # Display the image
        viewer.add_image(pred, name=f"PredFirstImg", visible=True)
        nib.save(nib.Nifti1Image(torch.squeeze(pred, dim=0).numpy(), None, None), f'predictions/result_{0}_{date.today()}.nii')                            
        #viewer = napari.view_image(pred, rgb=False)
    
    
    napari.run()


def runSingleImage():
    
    viewer = napari.Viewer()

    # Display the original image
    #img_file = "kaggle_3m/nni/sub-r001s001_ses-1_T1w.nii.gz"
    #img_file = "kaggle_3m/nni/sub-r001s002_ses-1_T1w.nii.gz"
    img_file = "kaggle_3m/nni/sub-r001s003_ses-1_T1w.nii.gz"
    test_load = nib.load(img_file)
    input_image = test_load.get_fdata() # last dimension is depth
    viewer.add_image(np.transpose(input_image, axes=[2,0,1]), name="Original")
    input_image = np.transpose(input_image, (0,1,2))
    #input_image = np.transpose(input_image, (1,0,2))
    print(f"Shape of the original image: {input_image.shape}") # 181 x 213 x 173
    
    
    #viewer = napari.view_image(input_image, rgb=False)
    

    # # shape comes in format x (width), y (height), z (depth) {test_load.header.get_data_shape()}
    # print(f"Shape after transpose: {type(input_image)} {input_image.shape} {test_load.header.get_data_dtype()}")
    # # Display the image
    # viewer = napari.view_image(input_image, rgb=False)
    # napari.run()

    pre_img = transforms.Compose([
        transforms.ToPILImage(),
        transforms.Resize((128,128)),
        #transforms.GaussianBlur(sigma=5.0),
        #transforms.CenterCrop((128,128)),
        transforms.ToTensor(),
        transforms.Normalize(mean=0, std=1)
    ])

    # Apply transformations to the image
    preprocess = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=0, std=1)
    ])

    input_image = preprocess(input_image).float()
    #input_image = torch.transpose(input_image, 1,2)
    viewer.add_image(input_image, name="Transformation") # Good shape and position
    print(f"Shape after transformation: {input_image.shape}")
    #viewer = napari.view_image(input_image, rgb=False)
    #napari.run()
    # resize images through depth
    new_img = []
    n = 64 #input_image.shape[0]
    for i in range(0,n):
        #new_img.append(pre_img(input_image[i,:,:]))
        new_img.append(input_image[i,:,:])

    final_input_img = torch.concat(new_img)
    print(f"last transform: {final_input_img.shape} and final array {final_input_img.shape}")
    viewer.add_image(input_image, name="Resized") # Good position
    # # viewer = napari.view_image(final_input_img, rgb=False)
    # # napari.run()
    # print(f"Shape after transformation: {final_input_img.shape}")

    # final_input_img = unsqueeze(final_input_img, dim=0) # adding 1 channel
    # final_input_img = unsqueeze(final_input_img, dim=0) # adding batch dimension


    # Show the corresponding mask
    #img_file = "kaggle_3m/nni/sub-r001s001_ses-1_mask.nii.gz"
    #img_file = "kaggle_3m/nni/sub-r001s002_ses-1_mask.nii.gz"
    # img_file = "kaggle_3m/nni/sub-r001s003_ses-1_mask.nii.gz"
    # mask = nib.load(img_file)
    # mask = mask.get_fdata()
    # mask = np.transpose(mask, (2,0,1)) # 2,0,1
    # print(f"Shape of the original image: {mask.shape}")
    # viewer.add_image(mask, name="Mask") # Good position
    # # Display the image
    #viewer = napari.view_image(input_image, rgb=False)
    #napari.run()

    # with torch.no_grad():
    #     # gray scale imges
    #     model = VNet(in_channels=1, out_channels=1)
    #     #model = VNET(in_channels=1, out_channels=1)
    #     pred = model(final_input_img)
    #     #pred = torch.squeeze(pred, dim=0)
    #     print(f"Shape of the prediction: {pred.shape}")
    #     # Display the image
    #     viewer = napari.view_image(pred, rgb=False)
        #napari.run()

    napari.run()

def loadPrediction(original=None, mask=None, prediction=None): 
    viewer = napari.Viewer()
    
    original = nib.load(original)
    original = torch.from_numpy(original.get_fdata()) # last dimension is depth # 2,0,1
    
    mask = nib.load(mask)
    mask = torch.from_numpy(mask.get_fdata()) # last dimension is depth 2,0,1

    prediction = nib.load(prediction)
    prediction = torch.from_numpy(prediction.get_fdata()).squeeze(dim=0) # last dimension is depth # 2,0,1

    print(original.shape)
    # resize images through depths
    new_img = []
    new_maks = []
    #list = range(0,image.shape[0]) 
    n = 32
    #list = sorted(random.sample(list, n))    
    #input_load = torch.unsqueeze(original, dim=1)  
    trans_test = ResizeImage(128)
    # Apply transformations
    start = original.shape[0] // 2 - 30
    for i in range(0,n):
    #for i in list:        
        new_img.append(trans_test(torch.unsqueeze(original[:,:,start + i], dim=0))) 
        new_maks.append(trans_test(torch.unsqueeze(mask[:,:,start + i], dim=0))) 
        #new_img.append(torch.unsqueeze(original[start + i,:,:], dim=0))    
    # Concatenate all images and add batch dimension
    print(f"Start: {start} - end {start + i} ")
    print(f"First element: {new_img[0].shape} I guess its dimension 2")
    final_input_img = torch.concat(new_img, dim=0)
    final_mask = torch.concat(new_maks, dim=0)
    print(f"Original: {original.shape} \n Original Processed: {final_input_img.shape} \n Mask: {mask.shape} \n Mask Processed: {final_mask.shape} \n Prediction {prediction.shape}")
    viewer.add_image(np.transpose(original.numpy(), axes=(2,0,1)), name="Original")
    viewer.add_image(final_input_img, name="OriginalP")
    viewer.add_image(np.transpose(mask.numpy(), axes=(2,0,1)), name="Mask")
    viewer.add_image(final_mask, name="MaskP")
    
   
    viewer.add_image(np.transpose(prediction, axes=(0,1,2)), name="Prediction")
    
    napari.run()

if __name__ == "__main__": 
    #loadSequence()
    loadPrediction(
                    original = "dataset/train/images/sub-r001s001_ses-1_T1w.nii.gz",
                    mask="dataset/train/masks/sub-r001s001_ses-1_space-orig_label-L_desc-T1lesion_mask.nii.gz", 
    prediction="predictions/result_1_2022-04-26.nii"
                )
    
    original = "dataset/train/images/sub-r001s001_ses-1_T1w.nii.gz",
    # mask="dataset/train/masks/sub-r001s001_ses-1_space-orig_label-L_desc-T1lesion_mask.nii.gz", 
    # prediction="predictions/result_0_2022-04-26.nii"

    original = "dataset/train/images/sub-r001s002_ses-1_T1w.nii.gz", 
    # mask="dataset/train/masks/sub-r001s002_ses-1_space-orig_label-L_desc-T1lesion_mask.nii.gz",
    # prediction="predictions/result_1_2022-04-26.nii"

    original = "dataset/train/images/sub-r001s003_ses-1_T1w.nii.gz", 
    # mask="dataset/train/masks/sub-r001s003_ses-1_space-orig_label-L_desc-T1lesion_mask.nii.gz", 
    # prediction="predictions/result_2_2022-04-26.nii"

    original = "dataset/train/images/sub-r001s004_ses-1_T1w.nii.gz", 
    # mask="dataset/train/masks/sub-r001s004_ses-1_space-orig_label-L_desc-T1lesion_mask.nii.gz", 
    # prediction="predictions/result_3_2022-04-26.nii"

    # mask="dataset/train/masks/sub-r001s005_ses-1_space-orig_label-L_desc-T1lesion_mask.nii.gz", 
    # prediction="predictions/result_4_2022-04-26.nii"

    # mask="dataset/train/masks/sub-r001s006_ses-1_space-orig_label-L_desc-T1lesion_mask.nii.gz",
    # prediction="predictions/result_5_2022-04-26.nii"

    
    
    
    
    
    #runSingleImage()