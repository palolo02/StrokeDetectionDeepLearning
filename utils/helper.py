from random import shuffle
import numpy as np
from PIL import Image
import os
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
from datetime import date
import cv2
from readConfig import DEPTH, DEVICE, IMAGE_HEIGHT, IMAGE_WIDTH,MODELS_FOLDER, OUTPUT_FOLDER, PLOTS_FOLDER, PARALLEL_GPUS
import shutil
import nibabel as nib
import torch.optim
from deepLearningModels.deep4step import UnetDeep4
from deepLearningModels.deep3step import UnetDeep3
from deepLearningModels.residualUnet import ResidualCustomUnet
from deepLearningModels.deep5step import Custom5LayersUnet
from deepLearningModels.xnet import Xnet
from deepLearningModels.unet import UNET
from deepLearningModels.shallowUnet import ShallowUNet
from deepLearningModels.attentionUnet import CustomAttentionUnet
from deepLearningModels.attentionUnetv2 import CustomAttentionUnetv2
from deepLearningModels.attentionUnetv3 import CustomAttentionUnetv3
from deepLearningModels.ResUnet import ResUnet
from deepLearningModels.ResUNetPlusPLus import ResUnetPlusPlus
import glob
from torch.optim.lr_scheduler import ReduceLROnPlateau, ExponentialLR, StepLR, MultiStepLR, CyclicLR
from monai.transforms import (Compose, EnsureTyped,
                LoadImaged, AddChanneld, NormalizeIntensityd, Resized, RandSpatialCropd)
from monai.networks.nets.unet import UNet
from monai.networks.nets.vnet import VNet
from monai.networks.nets.attentionunet import AttentionUnet
from monai.networks.nets.swin_unetr import SwinUNETR


def getLRScheduler(typeScheduler, optimizer, options):
    """Retrieve a different scheduler depending on the configuration file of the main process"""
    if typeScheduler == "ReduceLROnPlateau":
        return ReduceLROnPlateau(optimizer, mode=options["mode"], patience=options["patience"], factor=options["factor"], verbose=True, cooldown=options["cooldown"], min_lr=1e-9,)
    elif typeScheduler == "MultiStepLR":
        return MultiStepLR(optimizer, milestones=options["milestones"], gamma=options["gamma"], verbose=True)
    elif typeScheduler == "CyclicLR":
        return CyclicLR(optimizer, base_lr=options["base_lr"], max_lr=options["max_lr"], 
                        step_size_up=options["step_size_up"], base_momentum=options["base_momentum"], 
                        max_momentum=options["max_momentum"])

def getOptimizer(typeOptimizer, model, options):
    """Retrieve a different optimizer depending on the configuration file of the main process"""
    if options is not None:
        if typeOptimizer == "Adam":
            return torch.optim.Adam(model.parameters(), lr=options["lr"], weight_decay=options["weight_decay"]) # weight decay = L2 Penalty
        elif typeOptimizer == "AdamW":
            return torch.optim.AdamW(model.parameters(), lr=options["lr"], betas=(0.9, 0.999), eps=1e-08, weight_decay=options["weight_decay"], amsgrad=False, maximize=False)
        elif typeOptimizer == "SGD":
            return torch.optim.SGD(model.parameters(), lr=options["lr"], momentum=options["momentum"], weight_decay=options["weight_decay"])
        elif typeOptimizer == "RMSprop":
            return torch.optim.RMSprop(model.parameters(), lr=options["lr"])
        elif typeOptimizer == "Adagrad":
            return torch.optim.Adagrad(model.parameters(), lr=options["lr"])
    else:
        if typeOptimizer == "Adam":
            return torch.optim.Adam(model.parameters()) # weight decay = L2 Penalty
        elif typeOptimizer == "AdamW":
            return torch.optim.AdamW(model.parameters())
        elif typeOptimizer == "SGD":
            return torch.optim.SGD(model.parameters(), lr=0.0001)
        elif typeOptimizer == "RMSprop":
            return torch.optim.RMSprop(model.parameters())
        elif typeOptimizer == "Adagrad":
            return torch.optim.Adagrad(model.parameters())


def isSigmoid(typeModel):
    """ Determine which models need a Sigmoid function for computing the loss  """
    if typeModel == "Vnet" or \
        typeModel == "Unet" or typeModel == "UNet" or \
        typeModel == "AttentionUnet" or \
        typeModel == "SwinUNETR" or typeModel == "3DSwinUNETR" or \
        typeModel == "3DUnet":
        return True
    return False
    

def getModel(typeModel, DEVICE, noFilters=32):
    """Retrieve a different model depending on the configuration file of the main process"""
    if typeModel == "3LayersUnet" or typeModel == "UnetDeep3":
        model = UnetDeep3(in_ch=1, out_ch=1)
    if typeModel == "UnetDeep4":
        model = UnetDeep4(in_ch=1, out_ch=1)
    elif typeModel == "5LayersUnet" or typeModel == "Custom5LayersUnet":
        model = Custom5LayersUnet(in_ch=1, out_ch=1)
    elif typeModel == "CustomResUnet" or typeModel == "ResidualCustomUnet":
        model = ResidualCustomUnet(in_ch=1, out_ch=1)
    elif typeModel == "CustomUnet":
        model = UNET(in_channels=1, out_channels=1)
    elif typeModel == "CustomAttentionUnet":
        model = CustomAttentionUnet(in_channels=1, out_channels=1)    
    elif typeModel == "CustomAttentionUnetv2":
        model = CustomAttentionUnetv2(in_channels=1, out_channels=1)   
    elif typeModel == "CustomAttentionUnetv3":
        model = CustomAttentionUnetv3(in_channels=1, out_channels=1, base_filter_num=noFilters)    
    elif typeModel == "Xnet":
        model = Xnet(in_ch=1, out_ch=1)
    elif typeModel == "Vnet":
        model = VNet(
            spatial_dims=3, 
            in_channels=1, 
            out_channels=1, 
            #act=('elu', {'inplace': True}), 
            dropout_prob=0.5, 
            dropout_dim=3, 
            #bias=False
        )
    elif typeModel == "Unet" or typeModel == "UNet":
        model = UNet(
            spatial_dims=2,
            in_channels=1,
            out_channels=1,
            channels=(64, 128, 256),
            strides=(2, 2),
            num_res_units=2
        )
    elif typeModel == "3DUnet":
        model = UNet(
            spatial_dims=3,
            in_channels=1,
            out_channels=1,
            channels=(16, 32, 64, 128, 256),
            strides=(2, 2, 2, 2),
            num_res_units=2,
        )
    elif typeModel == "ShallowUnet":
        model = ShallowUNet(in_ch=1, out_ch=1)
    elif typeModel == "ResUnet" or typeModel == "ResNet":
        model = ResUnet(channel=1)    
    elif typeModel == "ResUnetPlusPlus" or typeModel == "ResNet++":
        model = ResUnetPlusPlus(channel=1)    
    elif typeModel == "AttentionUnet":
        model = AttentionUnet(spatial_dims=2,
                            in_channels=1, 
                            out_channels=1,
                            channels=(64, 128, 256),
                            strides=(2, 2)
                            )
    elif typeModel == "SwinUNETR":
        model = SwinUNETR(img_size=(IMAGE_WIDTH,IMAGE_HEIGHT), 
                            in_channels=1, 
                            out_channels=1, 
                            use_checkpoint=True, 
                            spatial_dims=2)
    elif typeModel == "3DSwinUNETR":
        model = SwinUNETR(img_size=(IMAGE_WIDTH,IMAGE_HEIGHT, DEPTH), 
                            in_channels=1, 
                            out_channels=1, 
                            use_checkpoint=True, 
                            spatial_dims=3)
    
    # Initialization method. init.kaiming_uniform_(self.weight, a=math.sqrt(5)) by default in Pytorch
    #model.apply(init_xavier_method)

    # If there are more than one GPU
    if PARALLEL_GPUS:
        if torch.cuda.device_count() > 1:
            print(f"{torch.cuda.device_count()} GPUs available")
            model = nn.DataParallel(model)
        
    model = model.to(DEVICE)

    return model

def _getAccuracyModel(path_models):    
    #import random
    val_dsc = []
    #shuffle(path_models)
    for model in path_models:
        checkpoint = torch.load(model)
        val_dice = checkpoint["val_dice"]
        max_value = np.max(val_dice)
        val_dsc.append(max_value)
        #print(f"Model: {checkpoint['model']}, Max DSC: {max_value}")
    
    best_model = np.argmax(val_dsc)
    #print(f"Best Model: {path_models[best_model]}, Max DSC: {val_dsc[path_models]}")
    return  path_models[best_model]

def _createFolderForModel(modelName):
    folder = "static/data/validation/"
    foldermodelName = modelName.split("\\")[-1].split("_")[0]
    if not os.path.exists(f"{folder}{foldermodelName}"):
        os.makedirs(f"{folder}{foldermodelName}")   
        os.makedirs(f"{folder}{foldermodelName}/input/")   
        os.makedirs(f"{folder}{foldermodelName}/comparison/")   
        os.makedirs(f"{folder}{foldermodelName}/mask/")   
        os.makedirs(f"{folder}{foldermodelName}/prediction/")
    return  modelName
    

def getLoadedModel(folder):
    """ Retrieves the latest pytorch model within the model location folder for the specified experiment in the config file"""                
    # Change model to select the one with highest accuracy
    # Read model configuration variables
    print(f"Loading model inside {folder}")    
    list_of_files = glob.glob(f"{folder}*.pt")
    list_of_files = [_createFolderForModel(file) for file in list_of_files if "_testing" not in file]    
    return _getAccuracyModel(list_of_files)
    #latest_file = max(list_of_files, key=os.path.getmtime)
    

def getNumberOfPaddingImages(input,n):
    """Returns the multiple of N to add padding images and fit the result into the model"""    
    result = 0    
    if input % n > 0:
        result = ((input//n)*n+n) - input    
    #print(f"Need to create {result} slices")
    return result

def init_xavier_method(m):
    if isinstance(m, torch.nn.Conv2d):
        print("Initializing network")
        torch.nn.init.xavier_uniform_(m.weight)
        #torch.nn.init.kaiming_uniform_(m.weight) # This is implemented by default in pytorch
    elif isinstance(m, torch.nn.Conv3d):
        print("Initializing network")
        torch.nn.init.xavier_uniform_(m.weight)
        #torch.nn.init.kaiming_uniform_(m.weight) # This is implemented by default in pytorch


def network_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

def applyTransformsToNiftiFile(imagePath, maskPath, size, n):

    pairwise = {"image": imagePath, 
                "label": maskPath}

    transf = Compose([
        LoadImaged(keys=["image","label"]),
        AddChanneld(keys=["image","label"]),
        NormalizeIntensityd(keys=["image"]),
        RandSpatialCropd(keys=["image","label"], roi_size=(size,size,n), random_center=False, random_size=False),
        Resized(keys=["image","label"], spatial_size=(size,size,n)),
        EnsureTyped(keys=["image","label"])
        # 1, 128, 128, 32
    ])
    res = transf(pairwise)
    image = res["image"]
    mask = res["label"]

    # Change order as tensor in pytorch B, C, H, W
    #image = image.permute(0,3,1,2)
    #mask = mask.permute(0,3,1,2)

    # temp_img = torch.unsqueeze(image[:, 10, :, :],dim=0)
    # temp_mask = torch.unsqueeze(mask[:, 10, :, :],dim=0)
    # temporary = torch.concat((temp_img,temp_mask), dim = 0)
    # grid = torchvision.utils.make_grid(temporary, nrow=2, padding=100)
    # torchvision.utils.save_image(grid, 'dataProcessing/Results/grid2.png')
    
    return image, mask


def resize_nifti_file(data_input, size):  
    data = np.copy(data_input)
    new_imgs = []    
    for i in range(data.shape[2]):        
        original_image = cv2.resize(data[:,:,i], dsize=(size,size), interpolation=cv2.INTER_NEAREST)     
        new_imgs.append(torch.unsqueeze(torch.from_numpy(original_image), dim=0))
    result = torch.concat(new_imgs, dim=0)
    result = result.numpy()   
    return result


def get_nii_gz_resized_file(pathlibpath, width, height, n):
    """ get all depth images in a .nii.gz file and resize them with the specified size """
    data = nib.load(pathlibpath).get_fdata()    
    new_imgs = []    
    for i in range(data.shape[2]):        
        original_image = cv2.resize(data[:,:,i], dsize=(width,height), interpolation=cv2.INTER_NEAREST)     
        new_imgs.append(torch.unsqueeze(torch.from_numpy(original_image), dim=0))
    result = torch.concat(new_imgs, dim=0)
    # Create number of padding images, if needed
    number_of_padding = getNumberOfPaddingImages(data.shape[2], n )
    if number_of_padding > 0:
        padding = torch.zeros((number_of_padding, width, height))
        result = torch.cat((result,padding), dim=0)
    return result

def napari_read_tiff_3d(pathlibpath,start=0, folder=False, nframes='all', resize=128, mask=False):
    '''
    Read a multiframe tif image and loads it in a numpy array
    '''
    stack = []
    if not folder:
        with Image.open(str(pathlibpath), mode='r') as img:            
            if nframes == 'all':
                try:
                    for i in range(start,img.n_frames):
                        img.seek(i)
                        stack.append(np.array(img))
                except:
                    print(f"Error in {i} frame")
                    pass       
    # D, H, W
    print("")
    return torch.concat(stack, dim=0)


def save_imagen(array, pathlibpath,tipo='float'):
  # Imagej has different coordinates base than python #     array= np.transpose(array,(1,0,2))
  # Saving 2d array while looping in "z"
    #print('Saving array as pic in path')
    if (tipo == 'uint8'):
        #skimage.img_as_ubyte(skimage.exposure.rescale_intensity(array))
        imagen = Image.fromarray(array)
    else:
        imagen = Image.fromarray(array.astype(tipo))
    imagen.save(pathlibpath)
    #print('Proceso Terminado')




def napari_read_tiff(pathlibpath,start=0, nframes='all'):
    '''
    Read a multiframe tif image and loads it in a numpy array
    '''

    with Image.open(str(pathlibpath), mode='r') as img:
        stack = []
        if nframes == 'all':
            for i in range(start,img.n_frames):
                img.seek(i)
                stack.append(np.array(img))
        else:
            for i in range(start,nframes):
                img.seek(i)
                stack.append(np.array(img))    

    return torch.from_numpy(np.array(stack))




def save_imagen3d(array, pathlibpath,tipo='float',axis=2, size=256, mask=False, viewer=None):
    import napari
    
    if viewer is None:
        viewer = napari.Viewer()
    # Saving 2d array while looping in "z"
    #print(f'Saving array as pic in path {array.shape}')
    if not os.path.exists(str(pathlibpath)):
        os.makedirs(str(pathlibpath))    
    # Add image 
    viewer.add_image(np.transpose(array, axes=[2,0,1]), name="Original")
    #input_image = np.transpose(input_image, (0,1,2))

    # Starts the image on napari
    viewer.dims.set_current_step(0,0)
    print(array.shape[2])
    for i in range(array.shape[2]):
        nombre = str(i).zfill(4) + '.png'        
        ruta_guardado= f"{pathlibpath}/{nombre}" 
        # save screenshot and update the scroll
        viewer.screenshot(path=f"{ruta_guardado}", canvas_only=True, flash=False)       
        viewer.dims.set_current_step(0,i)
        #plt.close()
    viewer.layers.pop()
   

def f1Score(input,target,noGrad=True):     
    """ Returns the value of the F1-Score metric""" 
    from torchmetrics import F1Score    
    f1 = F1Score().to(DEVICE)
    if type(input) is not int and type(target) is not int:
        result = f1(torch.flatten(input.int()), torch.flatten(target.int()))
        #result = f1(input.int().view(-1), target.int().view(-1))
    else:
        result = f1(torch.flatten(input), torch.flatten(target))        
        #result = f1(input.view(-1), target.view(-1))
    if noGrad:
        result = result.item()
    return result

def precisionAndRecall(input,target,num_classes=2,noGrad=True):
    """ Returns a tuple containing the Precision and Recall metrics"""
    from torchmetrics.functional import precision_recall
    if type(input) is not int and type(target) is not int:
        precision, recall = precision_recall(torch.flatten(input.int()), torch.flatten(target.int()), average='weighted', num_classes=num_classes)
    else:
        precision, recall = precision_recall(torch.flatten(input), torch.flatten(target), average='weighted', num_classes=num_classes)
    if noGrad:
        precision = precision.item()
        recall = recall.item()
    return precision, recall


def diceScore(input, target, noGrad=True):
    """ Calculate the dice score"""
    smooth = 1e-9
    iflat = input.reshape(-1)
    tflat = target.reshape(-1)
    intersection = (iflat * tflat).sum()
    union =  (iflat.sum() + tflat.sum())
    result = (2. * intersection) / max(union, smooth)
    if noGrad:
        result = result.item()
    return result

def intersectionOverUnion(prediction, ground_truth, noGrad=True):
    """ Calculate the IoU metric """        
    pflat = prediction.reshape(-1)
    gflat = ground_truth.reshape(-1)        
    smooth = 1e-9
    intersection = (gflat * pflat).sum()
    union = gflat.sum() + pflat.sum() - intersection         
    iou = intersection / max(union, smooth)    
    if noGrad:
        iou = iou.item()
    return iou
    

def returnCPUList(values):
    """ Return the values of a list as cpu values """
    copy = []        
    for v in values:
        if isinstance(v, torch.Tensor):
            if v.is_cuda:
                copy.append(v.detach().cpu())
            else:
                copy.append(v.detach())
        else:
            copy.append(v)
    return copy

def saveplot(values, name, sample, folder, decimals=2):
    """ Save one graph that corresponds to one metric"""
    values = returnCPUList(values)
    
    if len(values) == 0:
        print(f"No values to plot for {name}")
        return
    title = f"{name} per epoch | Epochs: {len(values)} | (n={sample}) "
    step = 10 if len(values) > 40 else 1
    fig, ax = plt.subplots()
    ax.plot(range(1,len(values)+1), values)
    ax.set(xlabel='epochs', ylabel=name,title=title)

    plt.xticks(range(0, len(values),step))
    ax.set_ylim(ymin=0)
    ax.xaxis.label.set_color('gray')        #setting up X-axis label color to yellow
    ax.yaxis.label.set_color('gray')          #setting up Y-axis label color to blue
    ax.tick_params(axis='x', colors='gray', labelsize=8)    #setting up X-axis tick color to red
    ax.tick_params(axis='y', colors='gray')  #setting up Y-axis tick color to black
    ax.spines['left'].set_color('gray')        # setting up Y-axis tick color to red
    ax.spines['top'].set_color('gray')  
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['bottom'].set_visible(True)
    ax.spines['left'].set_visible(True)
    plt.annotate(f"{values[-1]:.{decimals}f}",(len(values),values[-1]), color='black', size=8)        
    plt.savefig(f'{folder}/{name}_{date.today()}.png', format="png")
    plt.close(fig)

def saveComparisonPlot(values, name, sample, folder, lr):
    """ Save a comparison graph between 2 metrics """
    if len(values) == 1:
        saveplot(values[0],name="LearningRate", sample=sample, folder=folder, decimals=10)
        return
    cpu_values = [returnCPUList(cpu_value) for cpu_value in values]
    values_1 = returnCPUList(values[0])
    values_2 = returnCPUList(values[1])
    step = 10 if len(values_1) > 40 else 1
    #print(f"Values to plot. {values}")
    if len(values_1) == 0:
        print(f"No values to plot for {name}")
        return

    title = f"{name} | Epochs: {len(cpu_values[0])} |  (n={sample}) | LR: {lr:5f} "
    #print(values)
    fig, ax = plt.subplots()
    colors = ["#5E2378","#A3A2AB"]
    for i in range(len(cpu_values)):
        ax.plot(range(1,len(cpu_values[i])+1), cpu_values[i], color=colors[i])
    #ax.plot(range(1,len(values_1)+1), values_2, color="#A3A2AB")
    ax.set(xlabel='epochs', ylabel=name,title=title)
    ax.legend(["training", "testing"], loc=0, frameon=True)
    plt.xticks(range(0, len(cpu_values[0]),step))
    ax.set_ylim(ymin=0)
    ax.xaxis.label.set_color('gray')        #setting up X-axis label color to yellow
    ax.yaxis.label.set_color('gray')          #setting up Y-axis label color to blue
    ax.tick_params(axis='x', colors='gray', labelsize=8)    #setting up X-axis tick color to red
    ax.tick_params(axis='y', colors='gray')  #setting up Y-axis tick color to black
    ax.spines['left'].set_color('gray')        # setting up Y-axis tick color to red
    ax.spines['top'].set_color('gray')  
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['bottom'].set_visible(True)
    ax.spines['left'].set_visible(True)
    for i in range(len(cpu_values)):
        #if "Loss" in name:
        plt.annotate(f"{(cpu_values[i][-1]):.2f}",(len(cpu_values[i]),cpu_values[i][-1]), color='black', size=8)
        #else:
        #    plt.annotate(f"{np.mean(cpu_values[i]):.2f}",(len(cpu_values[i]),cpu_values[i][-1]), color='black', size=8)
    #plt.annotate(f"{np.mean(values_2):.2f}",(len(values_2),values_2[-1]), color='black', size=8)        
    plt.savefig(f'{folder}{name}_{date.today()}.png', format="png")
    plt.close(fig)

def createFoldersForExperiment(name):
    """ Creates the folders needed for saving the plots, the model and the output images"""    
    # Creating folder for model
    model_path = f'{MODELS_FOLDER}{name}'
    plots_path = f'{PLOTS_FOLDER}{name}'
    saved_images_path = f'{OUTPUT_FOLDER}{name}'

    # try:
    #     shutil.rmtree(model_path)
    #     print("Existing files removed!")        
    # except:
    if os.path.exists(model_path):
        shutil.rmtree(model_path)        
    os.makedirs(model_path)
    print(f"Creating folder for model: {model_path}")
        
    # try:
    #     shutil.rmtree(plots_path)
    #     print("Existing files removed!")
    # except:
    if os.path.exists(plots_path):
        shutil.rmtree(plots_path)           
    os.makedirs(plots_path)
    print(f"Creating folder for model: {plots_path}")
        
    # try:
    #     shutil.rmtree(saved_images_path)
    #     print("Existing files removed!")
    # except:
    if os.path.exists(saved_images_path):
        shutil.rmtree(saved_images_path)            
    os.makedirs(saved_images_path)
    print(f"Creating folder for ouput images: {saved_images_path}")
        
    # Move config file to keep a record of the initializing variables
    shutil.copy(src="config.json",dst=f'{MODELS_FOLDER}{name}')

def ConcatLossDiceIoULR(folder=None):
    """ Function to concatenate four images in one for better readiability """
    # locate the images within this folder    
    #folder = "plots/Stroke/TestMetricsFocalLoss/"
    print("Concatenating final results")
    loss_values = glob.glob(f"{folder}Loss*")
    dice_scores = glob.glob(f"{folder}Dice*")
    iou_scores = glob.glob(f"{folder}IoU*")
    lr_values = glob.glob(f"{folder}LearningRate*")
    
    for i in range(len(loss_values)):
        imgs = []
        imgs.append(cv2.imread(loss_values[i]))
        imgs.append(cv2.imread(dice_scores[i]))
        imgs.append(cv2.imread(iou_scores[i]))
        date_value = dice_scores[i].split("_")[-1].split(".")[0]
        try:
            imgs.append(cv2.imread(lr_values[i]))
            print("Image generated")
        except:
            print("Error in generating results image")
            pass
        # show the output image
        cv2.imwrite(f'{folder}Result_{date_value}.jpg', cv2.hconcat(imgs))

