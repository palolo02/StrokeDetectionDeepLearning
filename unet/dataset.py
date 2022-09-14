#from asyncio.windows_events import NULL
import os
from PIL import Image
from sqlalchemy import false
import torch
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
import numpy as np
import matplotlib.pyplot as plt
import logging
from datetime import date
from torchvision.transforms import transforms
import napari
from readConfig import configuration
from torchvision.utils import save_image
from utils.transformations import CustomTestStrokTrans, monai_train_transforms

class BrainTumorDataset(Dataset):
    """ Brain Tumor Dataset. Includes the sequence of images for several patients as well as the identifed regions where their tumor
    is located.
    """
    def __init__(self, image_dir, mask_dir, transform=None, maskTransform=None, max_number_sample=100):
        self.image_dir = image_dir
        self.mask_dir = mask_dir
        self.transform = transform
        self.maskTransform = maskTransform        
        self.max_number_sample = max_number_sample
        self.logFile = f'{configuration["folders"]["logs_dir"]}dataset-{date.today()}.log'
        logging.basicConfig(filename=self.logFile, level=logging.INFO,format='%(asctime)s:%(levelname)s:%(message)s')       
        
        self.images = []
        self.masks = []
        # Populate images and masks
        self._readFiles()


    def _readFiles(self):
        """ Read the files associated with the information of the images. Assuming Images and Masks are named similarly """                
        
        for file in os.listdir(self.mask_dir):                    
            if ".tif" in file:
                masks_folder = f"{self.mask_dir}{file}"
                img_folder = f"{self.image_dir}{file}".replace("_mask.tif",".tif")
                # Open the mask to see whether it is not empty. Otherwise, just add the files
                if configuration["samples"]["ignore_empty_masks"]:
                    with Image.open(masks_folder).convert("L") as mask:
                        if np.sum(np.asanyarray(mask)) > 0:
                            self.images.append(img_folder)
                            self.masks.append(masks_folder)
                        else: continue
                else:
                    self.images.append(img_folder)
                    self.masks.append(masks_folder)


        print(f"Total files: {len(self.images)} \n Total masks {len(self.masks)}")     
        if len(self.images) > self.max_number_sample:
            self.images = self.images[:self.max_number_sample]
            self.masks = self.masks[:self.max_number_sample]
        print(f"Total files: {len(self.images)} \n Total masks {len(self.masks)}")  
        logging.info(f"Total files: {len(self.images)} \n Total masks {len(self.masks)}")          


    def __len__(self):
        return len(self.images)
    
    def __getitem__(self, index):
        """ Returns a tuple of image and mask within the dataset """        
        img_path = self.images[index]
        mask_path = self.masks[index]
        image = np.array(Image.open(img_path).convert("RGB"))#, dtype=np.float32)
        mask = np.array(Image.open(mask_path).convert("L"))#, dtype=np.float32)
        #mask[mask == 255.0] = 1.0

        if self.transform is not None:
            # augmentations = self.transform(image=image, mask=mask)
            # image = augmentations["image"]
            # mask = augmentations["mask"]
            
            #image = self.transform(image)
            #mask = self.maskTransform(mask)
            image, mask = monai_train_transforms(image,mask,configuration["input_image"]["image_height"] )
            return image, mask
            #return image, mask

class BrainTumorModelDataset():
    """ Populates all images with folders """

    def __init__(self, batch_size, num_workers, max_num_sample_train,max_num_sample_test, train_transform, val_transform) -> None:
        self.train_img_dir  = configuration["folders_brain_tumor"]["train_img_dir"]
        self.train_mask_dir = configuration["folders_brain_tumor"]["train_mask_dir"]
        self.test_img_dir   = configuration["folders_brain_tumor"]["test_img_dir"]
        self.test_mask_dir  = configuration["folders_brain_tumor"]["test_mask_dir"]
        self.val_img_dir    = configuration["folders_brain_tumor"]["val_img_dir"]
        self.val_mask_dir   = configuration["folders_brain_tumor"]["val_mask_dir"]
        self.output_folder  = f'{configuration["folders"]["output_folder"]}{configuration["experiment"]["name"]}/'
        self.pin_memory = True
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.max_num_sample_train = max_num_sample_train
        self.max_num_sample_test = max_num_sample_test
        self.train_transform = train_transform        
        self.val_transform = val_transform
        self.train_loader = None
        self.test_loader = None
        self.val_loader = None       
            
    def getTrainAndValDataLoaders(self):
        """Return data loaders for training, testing and validation """
        if self.val_loader is None:
            #torch.manual_seed(100)
            traind_ds = BrainTumorDataset(image_dir=self.train_img_dir, mask_dir=self.train_mask_dir, transform=self.train_transform, max_number_sample=self.max_num_sample_train)
            self.train_loader = DataLoader(traind_ds, batch_size=self.batch_size, num_workers=self.num_workers, pin_memory=self.pin_memory, shuffle=True)        
            val_ds = BrainTumorDataset(image_dir=self.val_img_dir, mask_dir=self.val_mask_dir, transform=self.val_transform, max_number_sample=self.max_num_sample_test)
            self.val_loader = DataLoader(val_ds, batch_size=self.batch_size, num_workers=self.num_workers, pin_memory=self.pin_memory, shuffle=False)
            
        return self.train_loader, self.val_loader
    
    # def getTestDataloader(self):
    #     """ Return validation dataloader to test model's performance """
    #     if self.test_loader is None:
    #         test_ds = BrainTumorDataset(image_dir=self.test_img_dir, mask_dir=self.test_mask_dir, transform=self.val_transform, max_number_sample=self.max_num_sample_test)
    #         self.test_loader = DataLoader(test_ds, batch_size=self.batch_size, num_workers=self.num_workers, pin_memory=self.pin_memory, shuffle=False)        
    #     return self.test_loader
        
class StrokeDataset(Dataset):
    """ Brain Tumor Dataset. Includes the sequence of images for several patients as well as the identifed regions where their tumor
    is located.
    """
    def __init__(self, image_dir, mask_dir, transform=None, maskTransform=None, max_number_sample=10000):
        self.image_dir = image_dir
        self.mask_dir = mask_dir
        self.transform = transform
        self.maskTransform = maskTransform        
        self.max_number_sample = max_number_sample
        self.logFile = f'{configuration["folders"]["logs_dir"]}dataset-{date.today()}.log'
        logging.basicConfig(filename=self.logFile, level=logging.INFO,format='%(asctime)s:%(levelname)s:%(message)s')       
        
        self.images = []
        self.masks = []
        # Populate images and masks
        self._readFiles()
        
        
    
    def _readFiles(self):
        """ Read the files associated with the information of the images. Assuming Images and Masks are named similarly """                
        for dirpath, dirnames, filenames in os.walk(self.image_dir):
            for dir in dirnames:             
                for file in os.listdir(f"{dirpath}{dir}"):
                    if ".png" in file:
                    #if ".tiff" in file:
                        masks_folder = f"{dirpath}{dir}_mask/{file}".replace("images","masks")
                        img_folder = f"{dirpath}{dir}/{file}"
                        # Open the mask to see whether it is not empty. Otherwise, just add the files
                        if configuration["samples"]["ignore_empty_masks"]:
                            with Image.open(masks_folder).convert("L") as mask:
                                if np.sum(np.asanyarray(mask)) > 0:
                                    self.images.append(img_folder)
                                    self.masks.append(masks_folder)
                                else: continue
                        else:
                            self.images.append(img_folder)
                            self.masks.append(masks_folder)
        
        
        print(f"Total files: {len(self.images)} \n Total masks {len(self.masks)}")     
        if len(self.images) > self.max_number_sample:
            self.images = self.images[:self.max_number_sample]
            self.masks = self.masks[:self.max_number_sample]
        print(f"Total files: {len(self.images)} \n Total masks {len(self.masks)}")     

    def __len__(self):
        return len(self.images)
    
    def __getitem__(self, index):
        """ Returns a tuple of image and mask within the dataset """        
        img_path = self.images[index]
        mask_path = self.masks[index]
        # image = np.array(Image.open(img_path).convert("L"))
        # mask = np.array(Image.open(mask_path).convert("L"))        
        image = Image.open(img_path)
        mask = Image.open(mask_path)
        #mask[mask == 255.0] = 1.0
        #logging.info(f"Reading files... Image: {img_path} \n Mask: {mask_path}")        
        if self.transform is not None:
            image, mask = self.transform()(image, mask)
            #image, mask = stroke_train_img_mask_transforms(image, mask)
            
            #image = self.transform(image)
            # #mask = self.maskTransform(mask)
            # dict = {                
            #     "image" : img_path,                
            #     "mask": mask_path
            # }

            #result = monai_train_transforms(dict)
            #result = monai_train_transforms(dict)
            # result = torch.cat((result["image"][], result["mask"]), dim=0)
            # grid = torchvision.utils.make_grid(result, nrow=BATCH_SIZE, padding=100)
            # torchvision.utils.save_image(grid, f'{self.folder_output}training_{date.today()}.png')                                             
           #image, mask = stroke_train_img_mask_transforms(image,mask,configuration["input_image"]["image_height"])  
        
        #return result["image"], result["mask"]
        return image, mask

class StrokeModelDataset():
    """ Populates all images with folders """

    def __init__(self, batch_size, num_workers, max_num_sample_train,
                max_num_sample_test,max_num_sample_val, train_transform, val_transform) -> None:       
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
        self.max_num_sample_val = max_num_sample_val
        self.max_num_sample_test = max_num_sample_test
        self.train_transform = train_transform
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
                                      max_number_sample=self.max_num_sample_train)
            self.train_loader = DataLoader(traind_ds, 
                                            batch_size=self.batch_size, 
                                            num_workers=self.num_workers, 
                                            pin_memory=self.pin_memory, 
                                            shuffle=True)        
            val_ds = StrokeDataset(image_dir=self.val_img_dir, 
                                    mask_dir=self.val_mask_dir, 
                                    transform=self.val_transform,                                     
                                    max_number_sample=self.max_num_sample_val)
            self.val_loader = DataLoader(val_ds, 
                                        batch_size=self.batch_size, 
                                        num_workers=self.num_workers, 
                                        pin_memory=self.pin_memory, 
                                        shuffle=True)
            
        return self.train_loader, self.val_loader        

    def getTestDataLoader(self):
        if self.test_loader is None:
            #torch.manual_seed(100)
            test_ds = StrokeDataset(image_dir=self.test_img_dir, 
                                    mask_dir=self.test_mask_dir, 
                                    transform=self.val_transform,
                                    max_number_sample=self.max_num_sample_test)
                                    
            self.test_loader = DataLoader(test_ds, 
                                            batch_size=self.batch_size, 
                                            num_workers=self.num_workers, 
                                            pin_memory=self.pin_memory, 
                                            shuffle=True)        
          
        return self.test_loader   


class StrokeTestingModelDataset():
    """Class to generate a data loader for testing the performance of a model """    

    def __init__(self, batch_size, max_num_samples):                
        self.test_img_dir   = configuration["folders_brain_stroke"]["test_img_dir"]
        self.test_mask_dir  = configuration["folders_brain_stroke"]["test_mask_dir"]
        self.batch_size = batch_size        
        self.max_num_samples = max_num_samples
        self.num_workers = 0
        self.pin_memory = False

    def getTestDataloaderFromPNGImages(self):
        """ Return a testing dataloader from a set of png images"""        
        test_ds = StrokeDataset(image_dir=self.test_img_dir, 
                                mask_dir=self.test_mask_dir, 
                                transform=CustomTestStrokTrans,                                 
                                max_number_sample=self.max_num_samples)
        
        self.test_loader = DataLoader(test_ds, 
                                    batch_size=self.batch_size, 
                                    num_workers=self.num_workers, 
                                    pin_memory=self.pin_memory, 
                                    shuffle=True)        
        return self.test_loader

