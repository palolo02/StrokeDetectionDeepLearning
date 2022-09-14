from vnet.modelSteps import ModelPerformanceSteps
from utils.losses import DiceLoss, DiceBCELoss
from vnet.vnet import VNet
from vnet.strokeImages import StrokeImages
from vnet.strokeDataset import StrokeDataset, StrokeModelDataset
import torch.nn as nn
import torch.optim as optim
from torchvision.transforms import transforms
import torch.nn.functional as F
import torch
from datetime import datetime, date
import logging
from readConfig import LOGS_DIR, MODEL, configuration
from utils.helper import saveplot, saveComparisonPlot, diceScore, intersectionOverUnion
from utils.helper import createFoldersForExperiment, getModel, getOptimizer
from utils.losses import getLossFunction
from monai.utils import first, set_determinism

#from monai.handlers.utils import from_engine
from monai.networks.nets import UNet
from monai.networks.nets.vnet import VNet
from monai.networks.layers import Norm
from monai.metrics import DiceMetric
from monai.losses import DiceLoss
from monai.losses.focal_loss import FocalLoss
from monai.inferers import sliding_window_inference
from monai.data import CacheDataset, DataLoader, Dataset, decollate_batch
from monai.config import print_config
from readConfig import (
    EXPERIMENT,LEARNING_RATE,DEVICE, BATCH_SIZE, MAX_NUM_SAMPLE_TRAIN, IMAGE_WIDTH, IMAGE_HEIGHT, 
    NUM_EPOCHS, MAX_NUM_SAMPLE_TEST, NUM_WORKERS, optimizer_type, configuration, LOGS_DIR, MODEL,
    LOSS_FUNCTION, OPTIMIZER, OPTIMIZER_CONFIG
)
torch.cuda.empty_cache()


logFile = f'{LOGS_DIR}_{EXPERIMENT}_{date.today()}.log'
logging.basicConfig(filename=logFile, level=logging.INFO,format='%(asctime)s:%(levelname)s:%(message)s')

#print(DEVICE)
torch.cuda.empty_cache()

logging.info(f'Parameters:\n Learning_rate: {LEARNING_RATE}\n Device: {DEVICE}\n Batch size: {BATCH_SIZE}\n Epochs: {NUM_EPOCHS}\n Workers: {NUM_WORKERS}\n')
logging.info(f'Image dimensions: {IMAGE_HEIGHT}x{IMAGE_WIDTH}\n Training samples: {MAX_NUM_SAMPLE_TRAIN}\n Testing samples: {MAX_NUM_SAMPLE_TEST}\n')

def main():
    """ Run training and validation process """
    
    createFoldersForExperiment(EXPERIMENT)
    # Load images (if there is any)
    #print("[INFO] Generating images as first step")    
    #StrokeImages().loadImagesForTrainValTest()
    start_time = datetime.now()
    
    logging.info(f"Starting process: {start_time}")
    # Define transformations for training and validation
    
    torch.random.manual_seed(200)
    logging.info("VNET instance: (input) 1 channels => (output) 1 channel [mask]")
    #model = VNet(in_channels=1, out_channels=1).to(DEVICE)
    
    # Model
    # 128 x 128  
    model = getModel(MODEL, DEVICE)
    
    # ===== Loss Function =========    
    loss_fn = getLossFunction(LOSS_FUNCTION)
    
    # ===== Optimizer =========    
    # ["Adam","AdamW", "SGD","RMSprop", "Adagrad"]
    optimizer = getOptimizer(OPTIMIZER, model, OPTIMIZER_CONFIG)    

    logging.info("Obtaining Data Loaders for Stroke Images in training and validation")
        
    stroke_dataset = StrokeModelDataset(
                                        batch_size=BATCH_SIZE,
                                        num_workers=NUM_WORKERS,
                                        max_num_sample_train= MAX_NUM_SAMPLE_TRAIN,
                                        max_num_sample_test= MAX_NUM_SAMPLE_TEST,
                                        train_transform=None,
                                        mask_transform=None,
                                        val_transform=None
        )

    # get train and validation data loader for training
    train_loader, val_loader = stroke_dataset.getTrainAndValData()
    # Test model's performance with unseen data
    #test_loader = stroke_dataset.getTestDataloader()
    

    # Inititalize the training process for the model 
    steps = ModelPerformanceSteps(
                        model=model,
                        train_loader=train_loader,
                        test_loader=None,
                        val_loader=val_loader,                        
                        epochs=NUM_EPOCHS,
                        loss=loss_fn,
                        optimizer=optimizer,
                        learning_rate=LEARNING_RATE,
                        batch_size=BATCH_SIZE,
                        device=DEVICE
    )

    # Run the process
    steps.training3DModel()
    
    logging.info(f"Finishing process: {(datetime.now()-start_time)/60} mins")

if __name__=="__main__":
    main()

