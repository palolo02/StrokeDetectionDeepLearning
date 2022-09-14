import torch
import matplotlib.pyplot as plt
from torch import nn
import numpy as np
import napari
from deepLearningModels.unet import test
from utils.helper import *
from utils.losses import *
from utils.transformations import stroke_val_transformations, CustomTestStrokTrans
from readConfig import model_configuration
from utils.helper import save_imagen3d, resize_nifti_file
from tqdm import tqdm
import torchvision
from readConfig import (EARLY_STOPPING_ENABLE, EARLY_STOPPING_PATIENCE, LEARNING_RATE, ENABLED, HOST, LOGS_DIR, MODELS_FOLDER, OUTPUT_FOLDER, LOSS_FUNCTION, REPORT,
                        PLOTS_FOLDER, BATCH_SIZE, THRESHOLD, configuration, MAX_NUM_SAMPLE_TRAIN, MAX_NUM_SAMPLE_TEST, MAX_NUM_SAMPLE_VAL)

# https://huggingface.co/spaces/pytorch/U-NET-for-brain-MRI

import torch
import torchvision.transforms.functional as TF
from readConfig import (
    EXPERIMENT,LEARNING_RATE,DEVICE, BATCH_SIZE, LOSS_FUNCTION, MAX_NUM_SAMPLE_TRAIN, IMAGE_WIDTH, IMAGE_HEIGHT, MAX_NUM_SAMPLE_VAL, MODEL, 
    NUM_EPOCHS, MAX_NUM_SAMPLE_TEST, NUM_WORKERS, LOGS_DIR, DATASET, MODEL, 
    OPTIMIZER, OPTIMIZER_CONFIG, MULTIPLE_TESTS
)
from utils.transformations import (CustomTestStrokTrans, stroke_train_transformations, 
                        stroke_val_transformations, CustomTrainingStrokeTrans)
from unet.dataset import StrokeModelDataset
# Hyperparameters

class TestMRI():

    def __init__(self, model):
        self.model = model
        self._loadDataLoader()
    
    def _loadDataLoader(self):
        dataset = StrokeModelDataset(
                                batch_size=BATCH_SIZE,
                                num_workers=NUM_WORKERS,
                                max_num_sample_train= MAX_NUM_SAMPLE_TRAIN,
                                max_num_sample_test= MAX_NUM_SAMPLE_TEST,
                                max_num_sample_val= MAX_NUM_SAMPLE_VAL,
                                train_transform=CustomTrainingStrokeTrans,                                        
                                val_transform=CustomTestStrokTrans
        )
        # get train and validation data loader for training
        self.test_loader = dataset.getTestDataLoader()
    
    def validation(self):
        """ 
        Performs one entire validation cycle for the specified epoch.
        Arguments:             
        Returns: None            
        """     
        threshold_value = THRESHOLD        
        cum_loss = 0        
        num_correct = 0
        num_pixels = 0
        dice_score = 0
        iou = 0                
        recall = 0
        precision = 0
        f1 = 0        
        # Data Loader iteration               
        self.model.eval()
        with torch.no_grad():
            loop = tqdm(self.test_loader)
            print(f"Starting validation")
            # Iteration over the batch of images
            for batch_idx, (x,y) in enumerate(loop):
                x = x.float().to(self.device)
                y = y.float().to(self.device)            
                predictions = self.model(x).to(self.device)               
                if batch_idx % 5 == 0:
                    result = torch.cat((x,y,predictions), dim=0)
                    grid = torchvision.utils.make_grid(result, nrow=BATCH_SIZE, padding=100)
                    torchvision.utils.save_image(grid, f'{self.folder_output}testing_{date.today()}.png')                                             
                # calculate accuracy and dice score            
                num_correct += (predictions == y).sum().item()
                num_pixels += torch.numel(predictions)
                dice_score += diceScore(predictions, y)
                iou += intersectionOverUnion(predictions,y)
                f1 += f1Score(predictions,y)
                val1, val2 = precisionAndRecall(predictions,y)
                precision += val1
                recall += val2                
                            
                loop.set_postfix(set="testing", batch=batch_idx+1)
                #logging.info(f"==Training== Epoch:{epoch+1}, Batch:{batch_idx+1} loss:{loss.item()}")

            print(f"calculating measures for patient")                 
            # Storing metrics for performance
            accuracy = (num_correct/num_pixels)
            iou = (iou/len(loop))
            dice_score = (dice_score/len(loop))

            self.val_precision.append(precision/len(loop))
            self.val_recall.append(recall/len(loop))
            self.val_f1.append(f1/len(loop))
            self.val_accuracy.append(accuracy)
            self.val_dice.append(dice_score)
            self.val_iou.append(iou)
            self.val_loss_values.append((cum_loss/len(loop)))

            #logging.info(f"----- Iou in Training Dataset for {epoch+1}: {num_correct}/{num_pixels} =  {accuracy:.2f} % & Dice: {dice_score:.2f} % & Iou {iou:.5f}%")
            print(f"[VALIDATION] Avg Loss: {(cum_loss/len(loop)):.2f}")
            print(f"[VALIDATION] Avg Dice Score: {dice_score:.5f}")
            print(f"[VALIDATION] Avg IoU: {iou:.5f}")
            print(f"[VALIDATION] Recall: {(recall/len(loop)):.2f}")
            print(f"[VALIDATION] Precision: {(precision/len(loop)):.2f}")
            print(f"[VALIDATION] F1: {(f1/len(loop)):.2f}")
            
           

            #Plot training metrics
            #self.plotPerformanceTrainingMetrics()


class Prediction():

    def __init__(self):
        # Load model configuration and its parameters
        folder_model_uri = f'{model_configuration["folders"]["models_folder"]}{model_configuration["experiment"]["name"]}/'        
        self.model_uri = getLoadedModel(folder_model_uri)
        print(self.model_uri)
        self.temp_folder = None
        self.test_loader = None
        self._loadModelParameters()
        self.testMRI = TestMRI()       


    def _loadModels(self):
        """Load list of models to run the validation"""


    def _loadModelParameters(self):        
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

        #loaded_model = mlflow.pytorch.load_model(model_uri)
        # ["UNET", "3DUNET", "5DUNET", "XNET"]
        self.model = getModel(model_configuration["hyper_parameters"]["model"], self.device	)
        
        # ===== Loss Function ========= 
        # ["BCE","Dice Score", "BCE + Dice Score"] 
        self.loss_fn = getLossFunction(model_configuration["hyper_parameters"]["loss_function"])    
        
        # ===== Optimizer =========    
        # ["Adam","AdamW", "SGD","RMSprop", "Adagrad"]
        type_optimizer = model_configuration["hyper_parameters"]["optimizer"]
        self.optimizer = getOptimizer(type_optimizer,self.model,model_configuration["optimizer_config"][type_optimizer] )
        
        # ================== Load Model ====================
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.epoch = checkpoint['epochs']
        self.loss_function = checkpoint['loss_function']

    def _createTemporarydata(self, patient):
        """ Create temporry data for storing PNG images of the patient to predict """
        # Create a folder for the patient
        folder_name = model_configuration["folders"]["temp"]
        patient_folder = patient.split("/")[-1].replace(".nii.gz","")
        self.temp_folder = f"{folder_name}{patient_folder}"
        if not os.path.exists(self.temp_folder):
            os.mkdir(self.temp_folder)
            print("Temporary folder created for this patient")
        
        # save all sequence images as png   
        viewer = napari.Viewer()
        test_load = nib.load(patient)
        input_image = test_load.get_fdata() # last dimension is depth
        print("Saving sequence as PNG images")       
        save_imagen3d(array=input_image, pathlibpath=f"{self.temp_folder}", axis=2, viewer=viewer)
        viewer.close()

    def _cleanTemporaryData(self):
        """" Cleaning temporary data for patients """
        shutil.rmtree(self.temp_folder)
        print("Cleaning temporary data for patients")

   

    def predictPNGSequenceOfImages(self, patient, ground_truth):
        
        self._createTemporarydata(patient)

        viewer = napari.Viewer()
    
        self.model.eval()
        with torch.no_grad(): 
            temp = resize_nifti_file(nib.load(patient).get_fdata(), size=model_configuration["input_image"]["image_height"])
            viewer.add_image(temp, name="Original")
            temp = resize_nifti_file(nib.load(ground_truth).get_fdata(), size=model_configuration["input_image"]["image_height"])
            viewer.add_image(temp, name="Mask", blending="additive", colormap="green")
            
            # Read imnages in the patient folder            
            output_model = []
            for file in os.listdir(self.temp_folder):
                #img_transformed = stroke_val_transformations(np.array(Image.open(f"{self.temp_folder}/{file}").convert("L")))
                #img_transformed, _ = CustomTestStrokTrans()(np.array(Image.open(f"{self.temp_folder}/{file}").convert("L")))
                img_transformed, _ = CustomTestStrokTrans()(Image.open(f"{self.temp_folder}/{file}"))
                #print(img_transformed.shape)
                # input model (channel and batch)
                img_transformed = torch.unsqueeze(img_transformed,dim=0)
                # send to device
                img_transformed = img_transformed.to(self.device)
                #img_transformed = torch.unsqueeze(img_transformed,dim=0)
                output = self.model(img_transformed).to(self.device)
                output = (output > self.threshold).float()
                output = torch.squeeze(output, dim=0)
                output_model.append(output)
                

            prediction = torch.concat(output_model,dim=0).detach().cpu()            
            # save results
            
            # Get all results
            viewer.add_image(prediction.numpy(), name="Pred", blending="additive", colormap="red" )     
            #viewer.add_image(np.transpose(input_image, axes=[2,0,1]), name="Mask")
            prediction.shape
        
        #viewer.close()
        self. _cleanTemporaryData()

    def predictNifTISequenceOfImages(self, patient, ground_truth):
        """ Run the prediction with the input provided """
        
        viewer = napari.Viewer()
    
        self.model.eval()
        with torch.no_grad():            
            test_load = nib.load(patient)
            viewer.add_image(np.transpose(test_load.get_fdata(), axes=[2,0,1]), name="Original")
            viewer.add_image(np.transpose(nib.load(ground_truth).get_fdata(), axes=[2,0,1]), name="Mask", blending="additive", colormap="green" )
            input_image = np.float32(test_load.get_fdata()) # last dimension is depth
            output_model = []
            for i in range(input_image.shape[2]):
                # Show original image
                # temp = np.transpose(np.expand_dims(input_image[:,:,i], axis=0), (1,2,0))
                # plt.imshow(temp, cmap="gray")
                # plt.show()
                # plt.close()

                img_transformed = stroke_val_transformations(input_image[:,:,i])
                #print(img_transformed.shape)
                # input model (channel and batch)
                img_transformed = torch.unsqueeze(img_transformed,dim=0)
                # send to device
                img_transformed = img_transformed.to(self.device)
                #img_transformed = torch.unsqueeze(img_transformed,dim=0)
                output = self.model(img_transformed).to(self.device)
                output = (output > self.threshold).float()
                output = torch.squeeze(output, dim=0)
                output_model.append(output)

                # output = output.detach().cpu().numpy()
                # output = np.transpose(output, (1,2,0))
                # output_model.append(output)
                # print(f"{output.shape} and {type(output)}")
                # print(f"Max:{np.max(output)} Min: {np.min(output)}: Sum:{np.sum(output)}")
                #print(final_input_img.shape)
                #assert output.shape == final_input_img.shape
                # plt.imshow(output, cmap="gray")pre
                # plt.show()
                # plt.close()
            prediction = torch.concat(output_model,dim=0).detach().cpu()
            
            #prediction = np.concatenate(output_model,axis=0)
            # Get all results
            viewer.add_image(np.transpose(prediction.numpy(), axes=[2,0,1]), name="Pred", blending="additive", colormap="red" )     
            #viewer.add_image(np.transpose(input_image, axes=[2,0,1]), name="Mask")
            prediction.shape
        viewer.close()


       
    