import mlflow
from unet.dataset import BrainTumorModelDataset, StrokeModelDataset
from dataPipeline.modelSteps import ModelPerformanceSteps
from unet.brainTumorImages import BrainTumorImages
from utils.losses import getLossFunction
from utils.helper import getModel, getOptimizer
import torch
from datetime import datetime, date
import logging
from readConfig import (
    EXPERIMENT, DEVICE, MAX_NUM_SAMPLE_TRAIN, IMAGE_HEIGHT, IMAGE_WIDTH, MAX_NUM_SAMPLE_VAL, 
    MAX_NUM_SAMPLE_TEST, NUM_WORKERS, LOGS_DIR, DATASET,
    MULTIPLE_TESTS
)
from utils.helper import createFoldersForExperiment, network_parameters, isSigmoid
from utils.transformations import (CustomTestStrokTrans, stroke_train_transformations, 
                        stroke_val_transformations, CustomTrainingStrokeTrans)
torch.cuda.empty_cache()


logFile = f'{LOGS_DIR}_{EXPERIMENT}_{date.today()}.log'
logging.basicConfig(filename=logFile, level=logging.INFO,format='%(asctime)s:%(levelname)s:%(message)s')

# Printing the resources available in the system
logging.info(f"GPUs available: {str(torch.cuda.device_count())}")
for i in range(torch.cuda.device_count()):
    logging.info(torch.cuda.get_device_properties(f"cuda:{i}"))


#print(DEVICE)
torch.cuda.empty_cache()

def main():
    """ Run training and validation process """
    # Load images (if there is any)
    #print("[INFO] Generating images as first step")
    #BrainTumorImages("kaggle_3m").loadImagesForTrainValTest()
    start_time = datetime.now()
    
    logging.info(f"Starting process: {start_time}")
    torch.random.manual_seed(200)  
    
    #torch.random.manual_seed(100)
    # resizers: [256, 248, 240, 232, 224, 216, 208, 200, 192, 184, 176, 168, 160, 152, 144, 136, 128, 120, 112, 104, 96, 88, 80, 72, 64, 56, 48, 40, 32, 24, 16]
    for experiment in range(len(MULTIPLE_TESTS)):

        experiment_name = f"{EXPERIMENT}_{experiment}"
        print(f"================================== Starting Experiment {experiment_name} ==================================")
        createFoldersForExperiment(experiment_name)

        # Load node of current configuration
        current_config = MULTIPLE_TESTS[experiment]
        OPTIMIZER = current_config["Optimizer"]["name"]
        HYPERPARAMETERS = current_config["Hyperparameters"]
        MODEL = HYPERPARAMETERS["model"]
        LOSS = current_config["Loss"]
        LOSS_FUNCTION = LOSS["loss_function"]
        ALPHA = LOSS["alpha"]
        BETA = LOSS["beta"]
        SCHEDULER = HYPERPARAMETERS["scheduler"]
        NUM_EPOCHS = HYPERPARAMETERS["num_epochs"]
        NO_FILTERS =  HYPERPARAMETERS["no_filters"]

        logging.info("Obtaining Data Loaders for images in training and validation")
        if DATASET == "BrainTumor":
            dataset = BrainTumorModelDataset(
                                            batch_size=HYPERPARAMETERS["batch_size"],
                                            num_workers=NUM_WORKERS,
                                            max_num_sample_train= MAX_NUM_SAMPLE_TRAIN,
                                            max_num_sample_test= MAX_NUM_SAMPLE_TEST,
                                            max_num_sample_val= MAX_NUM_SAMPLE_VAL,
                                            train_transform=stroke_train_transformations,                                        
                                            val_transform=stroke_val_transformations                                        
            )
        elif DATASET == "BrainStroke":
            dataset = StrokeModelDataset(
                                            batch_size=HYPERPARAMETERS["batch_size"],
                                            num_workers=NUM_WORKERS,
                                            max_num_sample_train= MAX_NUM_SAMPLE_TRAIN,
                                            max_num_sample_test= MAX_NUM_SAMPLE_TEST,
                                            max_num_sample_val= MAX_NUM_SAMPLE_VAL,
                                            train_transform=CustomTrainingStrokeTrans,                                        
                                            val_transform=CustomTestStrokTrans
            )
        # Test model's performance with unseen data
        train_loader, val_loader = dataset.getTrainAndValDataLoaders() 
        
        model = getModel(MODEL, DEVICE)
        print(f"Number of network parameters: {network_parameters(model)}")
        logging.info(f"Number of network parameters: {network_parameters(model)}")
        
        # ===== Loss Function =========
        loss_fn = getLossFunction(LOSS_FUNCTION, alpha=ALPHA, beta=BETA)
        print(f'Loading parameters. Alpha: {ALPHA}, Beta: {BETA}')
        
        # ===== Optimizer =========            
        optimizer = getOptimizer(OPTIMIZER, model, current_config["Optimizer"][OPTIMIZER])
        
        # Inititalize the training process for the model 
        steps = ModelPerformanceSteps(
                            model=model,
                            train_loader=train_loader,
                            test_loader=None,
                            experiment_name = experiment_name,
                            val_loader=val_loader,                        
                            epochs=NUM_EPOCHS,
                            loss=loss_fn,
                            loss_name = LOSS_FUNCTION,
                            optimizer=optimizer,
                            scheduler = SCHEDULER,
                            learning_rate=current_config["Optimizer"][OPTIMIZER]["lr"],
                            batch_size=HYPERPARAMETERS["batch_size"],
                            device=DEVICE,
                            weight_decay=current_config["Optimizer"][OPTIMIZER]["weight_decay"],
                            momentum=current_config["Optimizer"][OPTIMIZER]["momentum"],
                            isSigmoid=isSigmoid(MODEL)
        )
        print(f'Parameters:\n Learning_rate: {current_config["Optimizer"][OPTIMIZER]["lr"]}\n Device: {DEVICE}\n Batch size: {HYPERPARAMETERS["batch_size"]}\n Epochs: {NUM_EPOCHS}\n Workers: {NUM_WORKERS}\n')
        print(f'Image dimensions: {IMAGE_HEIGHT}x{IMAGE_WIDTH}\n Training samples: {MAX_NUM_SAMPLE_TRAIN}\n Validation samples: {MAX_NUM_SAMPLE_VAL} Testing samples: {MAX_NUM_SAMPLE_TEST}\n')
        
        logging.info(f'Parameters:\n Learning_rate: {current_config["Optimizer"][OPTIMIZER]["lr"]}\n Device: {DEVICE}\n Batch size: {HYPERPARAMETERS["batch_size"]}\n Epochs: {NUM_EPOCHS}\n Workers: {NUM_WORKERS}\n')
        logging.info(f'Image dimensions: {IMAGE_HEIGHT}x{IMAGE_WIDTH}\n Training samples: {MAX_NUM_SAMPLE_TRAIN}\n Validation samples: {MAX_NUM_SAMPLE_VAL} Testing samples: {MAX_NUM_SAMPLE_TEST}\n')
        # Run the process
        steps.runModel()
        print(f"================================== Experiment {experiment_name} completed!==================================")
        
    
    logging.info(f"Finishing process: {(datetime.now()-start_time)/60} mins")
    

if __name__=="__main__":
    main()