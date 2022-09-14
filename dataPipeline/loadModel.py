import torch
import matplotlib.pyplot as plt
from torch import nn
import numpy as np
import napari
from deepLearningModels.unet import test
from utils.helper import *
from utils.losses import *
from utils.transformations import stroke_val_transformations
from readConfig import model_configuration
from utils.helper import save_imagen3d
import torch
import torchvision.transforms.functional as TF


class LoadPredictionModel():

    def __init__(self):
        # Load model configuration and its parameters
        folder_model_uri = f'{model_configuration["folders"]["models_folder"]}{model_configuration["experiment"]["name"]}/'        
        self.model_uri = getLoadedModel(folder_model_uri)
        print(self.model_uri)
        self.temp_folder = None
        self._loadModelParameters()

    def _loadModelParameters(self):        
        """ Load all configturation parameters from model """
        self.experiment = model_configuration["experiment"]["name"]
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.batch_size = model_configuration["hyper_parameters"]["batch_size"]
        self.image_height = model_configuration["input_image"]["image_height"]
        self.image_width = model_configuration["input_image"]["image_widths"]
        self.num_input_channels = 1
        
        # Load model configuration       
        checkpoint = torch.load(self.model_uri)
        self.learning_rate = checkpoint['learning_rate']
        self.threshold = model_configuration["hyper_parameters"]["threshold"]

        
        self.model = getModel(model_configuration["hyper_parameters"]["model"], self.device	)
        self.loss_function = getLossFunction(model_configuration["hyper_parameters"]["loss_function"])    
        type_optimizer = model_configuration["hyper_parameters"]["optimizer"]
        self.optimizer = getOptimizer(type_optimizer,self.model,model_configuration["optimizer_config"][type_optimizer] )
        
        # ================== Load Model ====================
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.epoch = checkpoint['epochs']
        self.name_loss_function = checkpoint['loss_function']

    def getTestingFolders(self):
        """Return the folder location of the tesitng dataset """
        return model_configuration["folders_brain_stroke"]["test_img_dir"], model_configuration["folders_brain_stroke"]["test_mask_dir"]