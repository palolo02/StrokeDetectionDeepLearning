import json
import os
import torch

# ====================================================
# Read initial configuration variables
# ====================================================
with open("config.json") as config_file:
    print("Config loaded successfully!")
    configuration = json.load(config_file)


DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
## -------------------------------- Experiment --------------------------------
EXPERIMENT = configuration["experiment"]["name"]
DATASET = configuration["experiment"]["dataset"]
REPORT = configuration["experiment"]["results"]

# -------------------------------- Folders --------------------------------
OUTPUT_FOLDER = configuration["folders"]["output_folder"]
PLOTS_FOLDER = configuration["folders"]["plots_folder"]
MODELS_FOLDER = configuration["folders"]["models_folder"]
LOGS_DIR = configuration["folders"]["logs_dir"]

# -------------------------------- Folders Stroke --------------------------------
TRAINING_IMAGES = configuration["folders_brain_stroke"]["train_img_dir"]
TRAINING_MASKS = configuration["folders_brain_stroke"]["train_mask_dir"]
VALIDATION_IMAGES = configuration["folders_brain_stroke"]["val_img_dir"]
VALIDATION_MASKS = configuration["folders_brain_stroke"]["val_mask_dir"]

# -------------------------------- Hyperparameters --------------------------------
THRESHOLD = configuration["hyper_parameters"]["threshold"]
BATCH_SIZE = configuration["hyper_parameters"]["batch_size"]
NUM_EPOCHS = configuration["hyper_parameters"]["num_epochs"]
NUM_WORKERS = configuration["hyper_parameters"]["num_workers"]
PIN_MEMORY = configuration["hyper_parameters"]["pin_memory"] 
LOAD_MODEL = configuration["hyper_parameters"]["load_model"]
MODEL = configuration["hyper_parameters"]["model"]
LOSS_FUNCTION = configuration["hyper_parameters"]["loss_function"]
DEPTH = configuration["hyper_parameters"]["depth"]

# -------------------------------- Optimizer --------------------------------
OPTIMIZER = configuration["hyper_parameters"]["optimizer"]
OPTIMIZER_CONFIG = configuration["optimizer_config"][OPTIMIZER]
LEARNING_RATE = OPTIMIZER_CONFIG["lr"]

# -------------------------------- Early Stopping --------------------------------
EARLY_STOPPING_PATIENCE = configuration["scheduler_config"]["earlyStopping"]["patience"]
EARLY_STOPPING_ENABLE = configuration["scheduler_config"]["earlyStopping"]["enable"]

# -------------------------------- Samples --------------------------------
MAX_NUM_SAMPLE_TRAIN = int(configuration["samples"]["total_samples"] * configuration["samples"]["training_percentage"])
MAX_NUM_SAMPLE_TEST = int(configuration["samples"]["total_samples"] * configuration["samples"]["testing_percentage"])
MAX_NUM_SAMPLE_VAL = int(configuration["samples"]["total_samples"] * configuration["samples"]["validation_percentage"])

# -------------------------------- Input Image --------------------------------
IMAGE_HEIGHT = configuration["input_image"]["image_height"]
IMAGE_WIDTH = configuration["input_image"]["image_widths"]

# -------------------------------- MLFlow --------------------------------
HOST = configuration["mlflow"]["host"]
ENABLED = configuration["mlflow"]["enabled"]

# -------------------------------- Multiple Tests --------------------------------
MULTIPLE_TESTS = configuration["multiple_tests"]

## -------------------------------- loss Configuration --------------------------------
ALPHA = configuration["loss_config"]["alpha"]
BETA = configuration["loss_config"]["beta"]
GAMMA = configuration["loss_config"]["gamma"]
WEIGHTS = configuration["loss_config"]["weights"]

model_configuration = None
PARALLEL_GPUS = configuration["hyper_parameters"]["parallel_gpus"]

# ==========================
# Read model configuration variables
# ==========================
if LOAD_MODEL:
    try:
        print("Loading model configuration...")
        model_configuration = None
        model_config_path = f'{MODELS_FOLDER}{EXPERIMENT}'
        for file in os.listdir(model_config_path):
            if ".json" in file:
                with open(f'{model_config_path}/{file}') as model_config_file:
                    model_configuration = json.load(model_config_file)
                    print("Model Config loaded successfully!")
        if model_configuration is None:
            print("Model config not found. Using initial configuration.")
            model_configuration = configuration    
    except:
        print("Error in loading model config not found. Using initial configuration")
        model_configuration = configuration

