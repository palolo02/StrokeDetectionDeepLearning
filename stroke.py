from vnet.modelSteps import ModelPerformanceSteps
from vnet.strokeDataset import StrokeModelDataset
import torch.nn as nn
import torch.optim as optim
from torchvision.transforms import transforms
import torch.nn.functional as F
import torch
from datetime import datetime, date
import logging
from utils.helper import createFoldersForExperiment, getModel, getOptimizer
from utils.losses import getLossFunction
from readConfig import (
    DEPTH, EXPERIMENT,LEARNING_RATE,DEVICE, BATCH_SIZE, LOSS_FUNCTION, MAX_NUM_SAMPLE_TRAIN, IMAGE_WIDTH, IMAGE_HEIGHT, MODEL, MULTIPLE_TESTS, 
    NUM_EPOCHS, MAX_NUM_SAMPLE_TEST, NUM_WORKERS, LOGS_DIR, DATASET, MODEL, OPTIMIZER, OPTIMIZER_CONFIG,
    TRAINING_IMAGES, TRAINING_MASKS, VALIDATION_IMAGES, VALIDATION_MASKS
)
torch.cuda.empty_cache()
from dataPipeline.temporaryData import GenerateNiftiGroups


logFile = f'{LOGS_DIR}_{EXPERIMENT}_{date.today()}.log'
logging.basicConfig(filename=logFile, level=logging.INFO,format='%(asctime)s:%(levelname)s:%(message)s')

#print(DEVICE)
torch.cuda.empty_cache()

logging.info(f'Parameters:\n Learning_rate: {LEARNING_RATE}\n Device: {DEVICE}\n Batch size: {BATCH_SIZE}\n Epochs: {NUM_EPOCHS}\n Workers: {NUM_WORKERS}\n')
logging.info(f'Image dimensions: {IMAGE_HEIGHT}x{IMAGE_WIDTH}\n Training samples: {MAX_NUM_SAMPLE_TRAIN}\n Testing samples: {MAX_NUM_SAMPLE_TEST}\n')

def main():
    """ Run training and validation process """

    print("Creating temporary data")
    folders = [TRAINING_IMAGES, TRAINING_MASKS, VALIDATION_IMAGES, VALIDATION_MASKS]    
    g = GenerateNiftiGroups(slices=DEPTH)
    # Clean temporary data
    print("Cleaning data")
    logging.info(f"Cleaning data")
    for folder in folders:
        g.removeFiles(folder) 
    print("Generating files")
    logging.info(f"Generating files")
    for folder in folders:
       g.loadFilesFromFolder(folder)
    print("Files completed. Starting data pipeline")
    logging.info(f"Files completed. Starting datapipeline")
    #================================================
    
    
    # Load images (if there is any)
    #print("[INFO] Generating images as first step")    
    #StrokeImages().loadImagesForTrainValTest()
    start_time = datetime.now()
    
    logging.info(f"Starting process: {start_time}")
    # Define transformations for training and validation
    
    torch.random.manual_seed(200)
    logging.info("VNET instance: (input) 1 channels => (output) 1 channel [mask]")
    #model = VNet(in_channels=1, out_channels=1).to(DEVICE)
    
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
    train_loader, val_loader = stroke_dataset.getTrainAndValDataLoaders()
    # Test model's performance with unseen data
    #test_loader = stroke_dataset.getTestDataloader()
    # resizers: [256, 248, 240, 232, 224, 216, 208, 200, 192, 184, 176, 168, 160, 152, 144, 136, 128, 120, 112, 104, 96, 88, 80, 72, 64, 56, 48, 40, 32, 24, 16]
    for experiment in range(len(MULTIPLE_TESTS)):
        experiment_name = f"{EXPERIMENT}_{experiment}"
        print(f"================================== Starting Experiment {experiment_name} ==================================")
        createFoldersForExperiment(experiment_name)        

        # 128 x 128  
        model = getModel(MODEL, DEVICE)
        #print(f"Number of network parameters: {network_parameters(model)}")
        
        # ===== Loss Function =========    
        loss_fn = getLossFunction(LOSS_FUNCTION)
        
        # ===== Optimizer =========    
        # ["Adam","AdamW", "SGD","RMSprop", "Adagrad"]
        optimizer = getOptimizer(OPTIMIZER, model, MULTIPLE_TESTS[experiment][OPTIMIZER])

        # Inititalize the training process for the model 
        steps = ModelPerformanceSteps(
                        model=model,
                        train_loader=train_loader,
                        test_loader=None,
                        val_loader=val_loader,
                        experiment_name = experiment_name,                  
                        epochs=NUM_EPOCHS,
                        loss=loss_fn,
                        optimizer=optimizer,
                        learning_rate=MULTIPLE_TESTS[experiment][OPTIMIZER]["lr"],
                        batch_size=BATCH_SIZE,
                        device=DEVICE,
                        weight_decay=MULTIPLE_TESTS[experiment][OPTIMIZER]["weight_decay"],
                        momentum=MULTIPLE_TESTS[experiment][OPTIMIZER]["momentum"]
    )

    # Run the process
    steps.training3DModel()
    print(f"================================== Experiment {experiment_name} completed!==================================")
           
    
    logging.info(f"Finishing process: {(datetime.now()-start_time)/60} mins")

if __name__=="__main__":
    main()