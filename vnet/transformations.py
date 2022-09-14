import torch
import matplotlib.pyplot as plt
import numpy as np
from torchvision import transforms
import nibabel as nib
from PIL import Image
import cv2


class ResizeImage(object):
    """ Resizes the image to fit in the input data for the model """

    def __init__(self, output_size):        
        self.output_size = output_size
        #self.device = "cuda" if torch.cuda.is_available() else "cpu"
    
    def __call__(self, image):
        """ Resize the image (H,W) and returns a (H,W) tensor """        
        # from D, H, W => W, H,
        image = image.numpy()
        # Resize the array applying one interpolaation
        image = np.reshape(image, (image.shape[1],image.shape[2]))
        image = cv2.resize(image, dsize=(self.output_size,self.output_size), interpolation=cv2.INTER_CUBIC)
        # from W, H, D => D, H, W
        tensors = torch.from_numpy(image).float()
        tensors = torch.unsqueeze(tensors, dim=0)        
        return tensors
       
