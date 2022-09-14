from unet.dataset import StrokeTestingModelDataset
from dataPipeline.modelSteps import ModelPerformanceSteps
import torch
from datetime import datetime, date
import logging
torch.cuda.empty_cache()
from utils.transformations import stroke_val_transformations, stroke_mask_transformations
from dataPipeline.loadModel import LoadPredictionModel
from tqdm import tqdm
import mlflow
import torchvision
from utils.helper import saveplot, saveComparisonPlot, diceScore, intersectionOverUnion, getLRScheduler
from monai.metrics import DiceMetric
# Load information from the 
from readConfig import model_configuration as configuration

class ModelTestingSteps():
    """Encapsulates the steps of the machine learning model to keep track of its performance in the testing dataset"""

    def __init__(self, model, test_loader, loss, optimizer, batch_size, device) -> None:
        self.model = model
        self.test_loader = test_loader        
        self.device = device
        self.loss = loss
        self.optimizer = optimizer
        self.batch_size =  batch_size
        self.folder_output = f'{configuration["folders"]["output_folder"]}{configuration["experiment"]["name"]}/'
        self.folder_plots = f'{configuration["folders"]["plots_folder"]}{configuration["experiment"]["name"]}/'
        self.folder_model = f'{configuration["folders"]["models_folder"]}{configuration["experiment"]["name"]}/'
        
        self.logFile = f'{configuration["folders"]["logs_dir"]}_{configuration["experiment"]["name"]}_{datetime.now().today()}.log'
        logging.basicConfig(filename=self.logFile, level=logging.INFO,format='%(asctime)s:%(levelname)s:%(message)s')

        if configuration["mlflow"]["enabled"]:
            mlflow.set_tracking_uri(configuration["mlflow"]["host"])

    def runTest(self):
        """ Run the model training and validation process to keep track of parameters in Mlflow """
        try:
            print("MLFlow server found! Recording metrics in experiment")
            with mlflow.start_run():
                self.testingModel()
        except Exception as e:
            print("Not MLFlow server found. Recording metrics locally.\n", e)
            self.testingModel()
        


    def testingModel(self):
        """ Main steps to keep track of the performance metrics """
        
        threshold_value = configuration["hyper_parameters"]["threshold"]
        dice_metric = DiceMetric(include_background=True, reduction="mean", get_not_nans=False)
       
        # Running epochs
        self.model.eval() 
        with torch.no_grad():               
            # ===============================
            # ======== Testing ===========
            # ===============================
            num_correct = 0
            num_pixels = 0
            dice_score = 0
            iou = 0
            val_loss = 0          
            denominator_iou = 0
            mean_dice_val = 0
            logging.info("Measuring model performance in Testing Dataset ====")
            loop = tqdm(self.test_loader)
            for batch_idx, (x, y) in enumerate(loop):                   
                x = x.float().to(self.device)
                y = y.float().to(self.device)                     
                predictions = self.model(x)
                loss = self.loss(predictions, y)
                preds = (predictions > threshold_value).float()
                # Monai metric
                dice_metric(y_pred=preds, y=y)           
                
                num_correct += (preds == y).sum()
                num_pixels += torch.numel(preds)                    
                
                # Ony consider those images with higlighted area 
                if torch.sum(y) > 0: 
                    denominator_iou +=1                                 
                dice_score += diceScore(preds, y)
                iou += intersectionOverUnion(preds,y)
                
                # # Saving images                       
                #x = x.resize_((x.shape[0], 1, x.shape[2], x.shape[3]))
                if batch_idx % 5 == 0:
                    result = torch.cat((x,y,predictions, preds), dim=0)
                    grid = torchvision.utils.make_grid(result, nrow=configuration["hyper_parameters"]["batch_size"], padding=100)
                    torchvision.utils.save_image(grid, f'{self.folder_output}original_{batch_idx}_{date.today()}.png')                        
                                    
                val_loss += loss.item()                    
                loop.set_postfix(set="validation", batch=batch_idx+1, loss=f"{loss.item():.2f}")
                logging.info(f"==Validation== Batch:{batch_idx+1},  loss:{loss.item()}")           

                try:
                    mlflow.log_metric(key="val_loss", value=loss.item())
                    mlflow.log_metric(key="val_total_time_epoch", value=loop.total)
                except:
                    pass                    
            
                # Monai metrics
            #mean_dice_val = dice_metric.aggregate().item()            
            #dice_metric.reset()

            accuracy = (num_correct/num_pixels)
            dice_score = (dice_score/denominator_iou)
            iou_score = (iou/denominator_iou)
            val_loss = val_loss/len(loop)
            
            logging.info(f"Got {num_correct}/{num_pixels} with acc {accuracy:.2f}")
            logging.info(f"Dice score: {dice_score:.2f}")
            logging.info(f"IoU score: {iou_score:.2f}")
            print(f"[TESTING] Avg Loss: {val_loss}")
            print(f"[TESTING] Avg Dice score: {dice_score:.5f}")
            print(f"[TESTING] Avg IoU: {iou_score:.5f}")                        
            
            try:
                mlflow.log_metric(key="test_dice_score", value=dice_score)
                mlflow.log_metric(key="test_avg_loss", value=val_loss)
                mlflow.log_metric(key="test_iou_score", value=iou_score)
            except Exception as e:
                pass            
            
            # =========================================
            # ======= Saving Testing Results  =========
            # =========================================            
            torch.save({
                    "test_loss" : val_loss,
                    "test_dice": dice_score,
                    "test_iou": iou_score
                    }, 
                    f'{self.folder_model}_testing_values__{date.today()}.pt')       
            try:            
                mlflow.pytorch.log_model(self.model, self.model._get_name(), registered_model_name=self.model._get_name())                               
            except:
                pass
        return val_loss, dice_score, iou_score


#print(DEVICE)
torch.cuda.empty_cache()

def test():
    """ Run testing process """

    start_time = datetime.now()    
    logging.info(f"Starting process: {start_time}")
    torch.random.manual_seed(200)

    loaded_model = LoadPredictionModel()
    test_img_dir, test_mask_dir = loaded_model.getTestingFolders()
    
    dataset = StrokeTestingModelDataset(
                                   test_img_dir = test_img_dir,
                                   test_mask_dir= test_mask_dir,
                                   batch_size= loaded_model.batch_size,
                                   test_transform= stroke_val_transformations,
                                   test_mask_transform= stroke_mask_transformations,
                                   max_num_samples= int(configuration["samples"]["total_samples"] * configuration["samples"]["testing_percentage"])
                                )

    # Test model's performance with unseen data
    test_loader = dataset.getTestDataloaderFromPNGImages()

    model_tesitng = ModelTestingSteps(
                            model=loaded_model.model, 
                            test_loader = test_loader, 
                            loss = loaded_model.loss_function, 
                            optimizer = loaded_model.optimizer, 
                            batch_size = loaded_model.batch_size, 
                            device = loaded_model.device)


    # Run the test
    model_tesitng.runTest()
    
    logging.info(f"Finishing process: {(datetime.now()-start_time)/60} mins")
    

# if __name__=="__main__":
#     test()