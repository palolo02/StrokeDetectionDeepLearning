import shutil
import nibabel as nib
import numpy as np 
from scipy import ndimage, misc
import matplotlib.pyplot as plt
import torch
import os

class GenerateNiftiGroups():

    def __init__(self, slices=32) -> None:
        #self.result_pathlib = "dataset/train/images/sub-r001s002_ses-1_T1w.nii.gz"
        self.no_slices = slices

    def loadFilesFromFolder(self, folder_pathlib):
        """ load all nifti files under the specified directory and generate subgroups inside subfolders with the name of the original nifti file """
        self.folder_pathlib = folder_pathlib
        for file in os.listdir(self.folder_pathlib):
            if ".nii.gz" in file:
                self.generateGroups(file)
    
    def removeFiles(self, folder_pathlib):
        """ Remove all directories and files """        
        for folder in os.listdir(folder_pathlib):
            if os.path.isdir(f"{folder_pathlib}{folder}"):
                for file in os.listdir(f"{folder_pathlib}{folder}"):
                    if ".nii.gz" in file:
                        os.remove(f"{folder_pathlib}{folder}/{file}")


    def generateGroups(self, filename):
        """Read individual nifti files and create a subset of nitfi files each with the specified slice number """
        # Load data to get number of slices
        img  = nib.load(f"{self.folder_pathlib}{filename}")
        img_data = torch.from_numpy(img.get_fdata())
        
        original_shape = img.header.get_data_shape()
        original_affine = img.affine
        original_header = img.header
        
        # generate groups
        if original_shape[2] % self.no_slices > 0:
            groups = (original_shape[2]//self.no_slices) + 1
        else:
            groups = (original_shape[2]//self.no_slices)

        start_offset = 0
        end_offset = 0
        
        
        # Create a nifti file for each group of slices
        folder_patient = filename[:16]
        if "mask" in filename:
            folder_patient += "_mask"
        if not os.path.exists(f"{self.folder_pathlib}{folder_patient}"):
            os.makedirs(f"{self.folder_pathlib}{folder_patient}")
            #print(f"Creating folder for plot results: {self.folder_pathlib}{folder_patient}")

        #print(f"Generating files for {folder_patient}")
        for g in range(0,groups):
            result_pathlib =  f"{self.folder_pathlib}{folder_patient}/{filename[:16]}_{self.no_slices}_{g}.nii.gz"

            start_offset = self.no_slices * g
            end_offset = start_offset + (original_shape[2] - start_offset) if (start_offset + self.no_slices) > original_shape[2] else start_offset + self.no_slices
            final_size = original_shape[2] - start_offset if (start_offset + self.no_slices) > original_shape[2] else self.no_slices
            result = torch.zeros((original_shape[0],original_shape[1],final_size))            
            # Start reading elements            
            result[:, :, :] = torch.clone(img_data[:, :, start_offset:end_offset])            
            # Save Result:
            nifti_result = nib.Nifti1Image(result, affine=original_affine, header=original_header)
            nib.save(nifti_result,result_pathlib)

