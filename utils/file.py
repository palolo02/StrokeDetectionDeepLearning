import os
import numpy as np
import torch
from PIL import Image
import nibabel as nib
from utils.helper import save_imagen3d, save_imagen, napari_read_tiff, napari_read_tiff_3d
import napari

class DataProcessor():
    """ Read .nii files and save them into folders """

    def __init__(self):
        self.size = 256
    
    def readPatientsFileAndSaveTIFF(self, folder_name):
        """ Read the main folder of .nii.gz files to save the folder """                
        print(f"====== Reading and saving files from {folder_name} ======")
        mask_flag = True if "mask" in folder_name else False        
        viewer = napari.Viewer()
        for  dirpath, dirnames, filenames in os.walk(folder_name):            
            for file in filenames:
                if ".nii.gz" in file:
                    pathlib = f"{folder_name}{file}"
                    test_load = nib.load(pathlib)
                    input_image = test_load.get_fdata() # last dimension is depth
                    name = file.replace('.nii.gz','')
                    save_imagen3d(array=input_image, pathlibpath=f"{folder_name}{name}", axis=2, size=self.size, mask=mask_flag, viewer=viewer)
                # Just one folder
                #break

       
        viewer.close()
    
    def addingPaddingOfImages(self, number, main_folder):
        """ Adds certain number of images to server as input for the model. In this case, a number divisible by 32."""
        for dirpath, dirnames, filenames in os.walk(main_folder):
            for dir in dirnames:                
                total_files = len(os.listdir(f"{dirpath}{dir}"))                
                if (total_files % number) > 0:
                    last_file_index = int(sorted(os.listdir(f"{dirpath}{dir}"))[-1].replace(".tiff",""))
                    # Add empty files 
                    [save_imagen(array=torch.zeros((self.size,self.size)).numpy(),pathlibpath=f"{dirpath}{dir}/{str(last_file_index+i+1).zfill(4)}.tiff") 
                        for i in range(number - (total_files % number))]
                

    def renameImagesAndMasks(self, main_folder):
        """ Rename folder for images and masks to have common naming convention"""
        # Rename folders
        for dirpath, dirnames, filenames in os.walk(main_folder):
            for dir in dirnames:
                print(f"{dirpath} {dir}")
                if "mask" in dir and "T1lesion" in dir:
                    os.rename(f"{dirpath}/{dir}",f"{dirpath}/{dir.replace('-1_space-orig_label-L_desc-T1lesion','')}")
                elif "1_T1w" in dir:
                    os.rename(f"{dirpath}/{dir}",f"{dirpath}/{dir.replace('-1_T1w','')}")

    
    def readPatientsTIFFFile(self, foldername):
        """ Read the main folder """                
        viewer = napari.Viewer()
        sequence = napari_read_tiff_3d(pathlibpath=foldername, start=0, folder=True, nframes="all")
        viewer.add_image(sequence, name="TIFF")
        napari.run()
   
