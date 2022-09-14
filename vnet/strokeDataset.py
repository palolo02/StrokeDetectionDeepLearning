#from asyncio.windows_events import NULL
from importlib.resources import path
import os
from PIL import Image
import torch
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
import torchvision.transforms.functional as F
import numpy as np
import matplotlib.pyplot as plt
import logging
from datetime import date
#from .transformations import ResizeImage
import nibabel as nib
import random
from utils.helper import get_nii_gz_resized_file, applyTransformsToNiftiFile
from utils.transformations import monai_train_transforms, monai_val_transforms
from readConfig import configuration
from torchvision.utils import save_image
from monai.data import CacheDataset, DataLoader, Dataset, decollate_batch
from monai.utils import first, set_determinism


class StrokeDataset(Dataset):
    """ Stroke Dataset: Includes the sequence of images for several patients as well as the identifed brain tissues where the stroke appeared. """
    def __init__(self, image_dir, mask_dir, transform=None, maskTransform=None, max_number_sample=5000, groups=False):
        self.image_dir = image_dir
        self.mask_dir = mask_dir
        self.transform = transform
        self.maskTransform = maskTransform
        self.max_number_sample = max_number_sample
        # Consider only folders
        self.images = []
        self.masks = []
        self.data = []
        # Populate images and masks
        self._readFilesNiiGZ(groups=groups)
        self.logFile = f'logs/stroke-dataset-{date.today()}.log'
        logging.basicConfig(filename=self.logFile, level=logging.INFO,format='%(asctime)s:%(levelname)s:%(message)s')
        

    def _readFilesNiiGZ(self, groups=False):
        """ Read the files associated with the information of the images. Assuming Images and Masks are named similarly """
        if groups:
            self._readWholeNiiGzFiles()
        else:
            self._readGroupNiiGzFiles()

    def _readGroupNiiGzFiles(self):
        # Read files inside folders
        for dirpath, dirnames, filenames in os.walk(self.image_dir):
            for dir in dirnames:
                dir_path = f"{self.image_dir}{dir}/"
                mask_dir_path = f"{self.mask_dir}{dir}_mask/"
                for file in os.listdir(dir_path):
                    if ".nii.gz" in file:
                        self.images.append(f"{dir_path}{file}")
                        self.masks.append(f"{mask_dir_path}{file}")

        print(f"Total files: {len(self.images)} \n Total masks {len(self.masks)}")     
        if len(self.images) > self.max_number_sample:
            self.images = self.images[:self.max_number_sample]
            self.masks = self.masks[:self.max_number_sample]
        print(f"Total files: {len(self.images)} \n Total masks {len(self.masks)}")  
        logging.info(f"Total files: {len(self.images)} \n Total masks {len(self.masks)}")
        # Generate a Dictionary with the image and corresponding mask
        self.data = [{"image": image_name, "mask": label_name} for image_name, label_name in zip(self.images, self.masks)]
      
    def _readWholeNiiGzFiles(self):

        for file in os.listdir(self.image_dir):
            if ".nii.gz" in file:
                # Add image
                dir_path = f"{self.image_dir}{file}"
                self.images.append(dir_path)
                mask_sufix = f"{file[0:18]}_space-orig_label-L_desc-T1lesion_mask.nii.gz"
                mask_path = f"{self.mask_dir}{mask_sufix}"
                self.masks.append(mask_path)
        
        print(f"Total files: {len(self.images)} \n Total masks {len(self.masks)}")     
        if len(self.images) > self.max_number_sample:
            self.images = self.images[:self.max_number_sample]
            self.masks = self.masks[:self.max_number_sample]
        print(f"Total files: {len(self.images)} \n Total masks {len(self.masks)}")  
        logging.info(f"Total files: {len(self.images)} \n Total masks {len(self.masks)}")
        # Generate a Dictionary with the image and corresponding mask
        self.data = [{"image": image_name, "mask": label_name} for image_name, label_name in zip(self.images, self.masks)]


    def __len__(self):
        """" Return the number of images of the specified dataset """        
        return len(self.images)
    
    def __getitem__(self, index):
        """ Returns a tuple of image and mask within the dataset """         
        # img_path = os.path.join(self.image_dir, self.images[index])
        # mask_path = os.path.join(self.mask_dir, self.masks[index])
        # img_path = self.images[index]
        # mask_path = self.masks[index]
        img_path = self.data[index]["image"]
        mask_path = self.data[index]["mask"]

        logging.info(f"Reading image {self.images[index]} and mask: {self.masks[index]}")
        
        # We ensure that we're reading the pair-wise element for image and masks
        assert self.images[index].split("/")[-1].split(".")[0] == self.masks[index].split("/")[-1].split(".")[0]
                
        image, mask = applyTransformsToNiftiFile(img_path, 
                                                mask_path, 
                                                configuration["input_image"]["image_widths"],
                                                configuration["hyper_parameters"]["depth"])
        
        # Return pairwise elments as B, D, H, W
        return {"image" : image, "mask" : mask }
        

    
class StrokeModelDataset():
    """ Populates all images with folders """

    def __init__(self, batch_size, num_workers, max_num_sample_train,max_num_sample_test, train_transform, mask_transform, val_transform) -> None:
        self.train_img_dir  = configuration["folders_brain_stroke"]["train_img_dir"]
        self.train_mask_dir = configuration["folders_brain_stroke"]["train_mask_dir"]
        self.test_img_dir   = configuration["folders_brain_stroke"]["test_img_dir"]
        self.test_mask_dir  = configuration["folders_brain_stroke"]["test_mask_dir"]
        self.val_img_dir    = configuration["folders_brain_stroke"]["val_img_dir"]
        self.val_mask_dir   = configuration["folders_brain_stroke"]["val_mask_dir"]
        self.output_folder  = f'{configuration["folders"]["output_folder"]}{configuration["experiment"]["name"]}/'
        self.pin_memory = True
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.max_num_sample_train = max_num_sample_train
        self.max_num_sample_test = max_num_sample_test
        self.train_transform = train_transform
        self.mask_transform = mask_transform
        self.val_transform = val_transform
        self.train_loader = None
        self.test_loader = None
        self.val_loader = None
        
            
    def getTrainAndValDataLoaders(self):
        """Return data loaders for training, testing and validation """
        if self.val_loader is None:
            #torch.manual_seed(100)
            traind_ds = StrokeDataset(image_dir=self.train_img_dir, 
                                        mask_dir=self.train_mask_dir, 
                                        transform=self.train_transform, 
                                        maskTransform=self.mask_transform, 
                                        max_number_sample=self.max_num_sample_train)
            self.train_loader = DataLoader(traind_ds, batch_size=self.batch_size, 
                                        num_workers=self.num_workers, 
                                        pin_memory=self.pin_memory, 
                                        shuffle=True)            
            val_ds = StrokeDataset(image_dir=self.val_img_dir, 
                                    mask_dir=self.val_mask_dir, 
                                    transform=self.val_transform, 
                                    maskTransform=self.mask_transform, 
                                    max_number_sample=self.max_num_sample_test)
            self.val_loader = DataLoader(val_ds, batch_size=self.batch_size, 
                                    num_workers=self.num_workers, 
                                    pin_memory=self.pin_memory, 
                                    shuffle=True)
            print(f"Loaded files: Training : {len(self.train_loader)} Testing: {len(self.val_loader)}")

        return self.train_loader, self.val_loader
    
    def getTestDataloader(self):
        """ Return validation dataloader to test model's performance """
        if self.test_loader is None:
            test_ds = StrokeDataset(image_dir=self.test_img_dir, 
                                    mask_dir=self.test_mask_dir, 
                                    transform=self.val_transform, 
                                    max_number_sample=self.max_num_sample_test)
            self.test_loader = DataLoader(test_ds, batch_size=self.batch_size, 
                                    num_workers=self.num_workers, 
                                    pin_memory=self.pin_memory, 
                                    shuffle=False)        
        return self.test_loader
        
    def getTrainAndValData(self):
        """Return data loaders for training, testing and validation """
        if self.val_loader is None:
            set_determinism(seed=100)
            #torch.manual_seed(100)
            traind_ds = StrokeDataset(image_dir=self.train_img_dir, mask_dir=self.train_mask_dir, transform=self.train_transform, maskTransform=self.mask_transform, max_number_sample=self.max_num_sample_train, groups=False)
            #self.train_loader = DataLoader(traind_ds, batch_size=self.batch_size, num_workers=self.num_workers, pin_memory=self.pin_memory, shuffle=True)            
            cache_train_dataset = CacheDataset(data=traind_ds.data, transform=monai_train_transforms, cache_rate=configuration["samples"]["cache"], num_workers=0)
            self.train_loader = DataLoader(cache_train_dataset, batch_size=self.batch_size, shuffle=True, num_workers=0)

            val_ds = StrokeDataset(image_dir=self.val_img_dir, mask_dir=self.val_mask_dir, transform=self.val_transform, maskTransform=self.mask_transform, max_number_sample=self.max_num_sample_test, groups=False)
            cache_val_dataset = CacheDataset(data=val_ds.data, transform=monai_val_transforms, cache_rate=configuration["samples"]["cache"], num_workers=0)
            #self.val_loader = DataLoader(val_ds, batch_size=self.batch_size, num_workers=self.num_workers, pin_memory=self.pin_memory, shuffle=True)
            self.val_loader = DataLoader(cache_val_dataset, batch_size=self.batch_size, num_workers=0)
            #print(f"Loaded files: Training : {len(self.train_loader)} Testing: {len(self.val_loader)}")

        #return traind_ds.data, val_ds.data
        return self.train_loader, self.val_loader