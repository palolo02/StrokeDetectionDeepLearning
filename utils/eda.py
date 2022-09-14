from tqdm import tqdm
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from PIL import Image
from readConfig import configuration
import torch
from datetime import datetime, date
from torchvision.transforms import transforms
from unet.dataset import StrokeModelDataset

# Hyperparameters
LEARNING_RATE = configuration["hyper_parameters"]["learning_rate"]
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
BATCH_SIZE = configuration["hyper_parameters"]["batch_size"]
NUM_EPOCHS = configuration["hyper_parameters"]["num_epochs"]
NUM_WORKERS = configuration["hyper_parameters"]["num_workers"]
IMAGE_HEIGHT = configuration["input_image"]["image_height"]
IMAGE_WIDTH = configuration["input_image"]["image_widths"]

PIN_MEMORY = True # Flag to add dataloaders to cuda
LOAD_MODEL = True
MAX_NUM_SAMPLE_TRAIN = int(configuration["samples"]["total_samples"] * configuration["samples"]["training_percentage"])
MAX_NUM_SAMPLE_TEST = int(configuration["samples"]["total_samples"] * configuration["samples"]["validation_percentage"])
METADATA = configuration["folders"]["metadata"]

class ExploratoryDataAnalysis():
    """ Analyze MRI images """
    
    def __init__(self) -> None:
        self.metadata = None

    def _loadMetadata(self):
        """ Load the dataloaders in train, validation and test to display statistical analysis of the information of the model """
        data = pd.read_csv(METADATA, delimiter=";")
        #dat

    def runStatisticalAnalysis(self):
        """  """
        torch.cuda.empty_cache()
        #BrainTumorImages("kaggle_3m").loadImagesForTrainValTest()
        start_time = datetime.now()

        #
       