from numpy import append
from pkg_resources import DEVELOP_DIST
from tqdm import tqdm
import mlflow
from datetime import date
import torchvision
import matplotlib.pyplot as plt
import logging
from datetime import datetime, date
from torch.optim.lr_scheduler import ReduceLROnPlateau, ExponentialLR, StepLR, MultiStepLR
import torch
import torch.nn as nn
import nibabel as nib
from .transformations import ResizeImage
import napari
from torch.optim.lr_scheduler import StepLR
from sklearn.metrics import jaccard_score
from monai.metrics import DiceMetric
from utils.helper import (saveplot, saveComparisonPlot, diceScore, ConcatLossDiceIoULR,
                            intersectionOverUnion, getLRScheduler, precisionAndRecall, f1Score)
from readConfig import (DEVICE, EARLY_STOPPING_PATIENCE, ENABLED, HOST, LOGS_DIR, MAX_NUM_SAMPLE_TEST, MAX_NUM_SAMPLE_TRAIN, MAX_NUM_SAMPLE_VAL, MODELS_FOLDER, OUTPUT_FOLDER, 
                        PLOTS_FOLDER, EXPERIMENT, BATCH_SIZE, REPORT, THRESHOLD, configuration, LOSS_FUNCTION)
from monai.utils import first, set_determinism
from monai.transforms import Compose, EnsureType, AsDiscrete
import numpy as np
import pandas as pd
torch.cuda.empty_cache()
from dataPipeline.earlyStopping import EarlyStopping

class ModelPerformanceSteps():
    """Encapsulates the steps of the machine learning model to keep track of its performance"""

    def __init__(self, model, train_loader, test_loader, val_loader, experiment_name, epochs, loss, optimizer, learning_rate, batch_size, device, weight_decay, momentum) -> None:
        self.model = model
        self.experiment_name = experiment_name
        self.train_loader = train_loader
        self.test_loader = test_loader    
        self.val_loader = val_loader            
        self.epochs = epochs
        self.device = device
        self.loss = loss
        self.loss_function = LOSS_FUNCTION
        self.optimizer = optimizer
        self.optimizer_name = str(self.optimizer.__class__).split(".")[-1].replace("'>","")
        self.learning_rate = learning_rate
        self.weight_decay = weight_decay
        self.momentum = momentum
        self.initial_lr = learning_rate
        self.batch_size =  batch_size
        self.folder_output = f'{OUTPUT_FOLDER}{experiment_name}/'
        self.folder_plots = f'{PLOTS_FOLDER}{experiment_name}/'
        self.folder_model = f'{MODELS_FOLDER}{experiment_name}/'
        self.train_accuracy = []
        self.train_dice = []
        self.train_iou = []        
        self.train_loss_values = []
        self.train_iou = []
        self.test_accuracy = []
        self.test_dice = [] 
        self.test_loss_values = []
        self.test_iou = []
        self.val_accuracy = []
        self.val_dice = [] 
        self.val_loss_values = []
        self.val_iou = []
        
        self.val_dice_complete = []
        self.train_dice_complete = []
        self.val_iou_complete = []
        self.train_iou_complete = []


        # Monai metric
        self.monai_dice_training = []
        self.monai_dice_testing = []

        # Metrics
        self.train_precision = []
        self.train_recall = []
        self.train_f1 = []
        self.val_precision = []
        self.val_recall = []
        self.val_f1 = []
        self.lr_values = []

        self.best_model_acc = 0

        self.logFile = f'{LOGS_DIR}_{experiment_name}_{datetime.now().today()}.log'
        logging.basicConfig(filename=self.logFile, level=logging.INFO,format='%(asctime)s:%(levelname)s:%(message)s')
        if ENABLED:
            mlflow.set_tracking_uri(HOST)
    
    def runModel(self):
        """ Run the model training and validation process to keep track of parameters in Mlflow """
        try:
            print("MLFlow server found! Recording metrics in experiment")
            with mlflow.start_run():
                self.training3DModel()
        except Exception as e:
            print("Not MLFlow server found. Recording metrics locally.\n", e)
            self.training3DModel()
        

    # def return_subset_data(self, x, y, n, iterations):
    #     """ Returns a generator to process the minibatches of images for feeding the model within a patient's data sequence"""
    #     #print(f"Shape of x within generator: {x.shape}")
    #     start_offset = 0
    #     end_offset = n        
    #     # Start reading elements       
    #     for _ in range(iterations):
    #         yield _ , x[:, :, start_offset:end_offset, :, :], y[:, :, start_offset:end_offset, :, :]
    #         start_offset += n
    #         end_offset += n

    def training3DModel(self):
        """ main steps to keep track of the performance metrics """
        
        scheduler_config = configuration["hyper_parameters"]["scheduler"]
        scheduler = getLRScheduler(scheduler_config, self.optimizer,configuration["scheduler_config"][scheduler_config])
        early_stopping = EarlyStopping(patience=EARLY_STOPPING_PATIENCE)
        print(f"Initial Learning rate: {self.learning_rate}")
        # Running epochs                
        try:
            mlflow.log_param("learning_rate", self.learning_rate)
            mlflow.log_param("epochs", self.epochs)
            mlflow.log_param("optimizer",self.optimizer_name)
            mlflow.log_param("loss_function", self.loss._get_name().split(".")[-1])
            mlflow.log_param("batch_size", self.batch_size)
            mlflow.log_param("training_sample", len(self.train_loader))
            #mlflow.log_param("testing_sample", len(self.test_loader))
            mlflow.log_param("validation_sample", len(self.val_loader))
        except Exception as e:
            print("logging parameters")
            print(e)
            pass
        
        post_pred = Compose([EnsureType(), AsDiscrete(threshold=0.5)])
        post_label = Compose([EnsureType(), AsDiscrete(threshold=0.5)])
        threshold_value = THRESHOLD        
        
        dice_metric = DiceMetric(include_background=False, reduction="mean")
        for epoch in range(self.epochs):
            logging.info(f"Runing epoch {epoch+1} / {self.epochs}...")
            cum_train_loss = 0
            plot_train_loss = []
            # ===============================
            # ========= Training  ===========
            # ===============================
            loop = tqdm(self.train_loader)          
            num_correct = 0
            num_pixels = 0
            dice_score = 0
            denominator_iou = 0
            iou = 0
            recall = 0
            precision = 0
            f1 = 0
            self.model.train()
            for batch_idx, (data) in enumerate(loop):
            #for batch_idx, (x,y) in enumerate(loop):
                x = data["image"]
                y = data["mask"]
                # Reasign axis
                x = x.permute(0,1,4,2,3)
                y = y.permute(0,1,4,2,3)

                x = x.float().to(self.device)
                y = y.float().to(self.device)
                predictions = self.model(x).to(self.device)
                # Apply Sigmoid for computing the loss as it is a classification problem
                predictions = nn.Sigmoid()(predictions)
                loss = self.loss(predictions, y)
                predictions_ = (predictions > threshold_value).float()
                
                if batch_idx % 5 == 0:
                    result = torch.cat((x[:, :, 10, :, :],
                                        y[:, :, 10, :, :],
                                        predictions[:, :, 10, :, :], 
                                        predictions_[:, :,10, :, :]), dim=0)
                    grid = torchvision.utils.make_grid(result, nrow=BATCH_SIZE, padding=100)
                    torchvision.utils.save_image(grid, f'{self.folder_output}training_{date.today()}.png')                                 
                #torchvision.utils.save_image(grid, f'{self.folder_output}training_{batch_idx}_{date.today()}.png')                                             

                num_correct += (predictions_ == y).sum().item()
                num_pixels += torch.numel(predictions_)
                #dice_score += (2*(predictions*targets).sum()) / ((predictions+targets).sum()+1.e-8)
                # Ony consider those images with higlighted area 
                if torch.sum(y) > 0: 
                    denominator_iou +=1
                dice_score += diceScore(predictions, y)                                    
                iou += intersectionOverUnion(predictions,y) 
                f1 += f1Score(predictions,y)
                val1, val2 = precisionAndRecall(predictions,y)
                precision += val1
                recall += val2 

                # compute metric for current iteration
                predictions_ = post_pred(predictions)
                y = post_label(y)
                dice_metric(y_pred=predictions_, y=y)
                

                # backward prop
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()
                    
                # self.optimizer.zero_grad()
                # scaler.scale(loss).backward()
                # scaler.step(self.optimizer)
                # scaler.update()

                # Adding the loss for the current batch
                cum_train_loss += loss.item()
                plot_train_loss.append(loss.item())         

                # update tqdm loop
                loop.set_postfix(set="trainig", exp=self.experiment_name, loss=loss.item(), epoch=epoch+1, batch=batch_idx+1)
                logging.info(f"==Training== Epoch:{epoch+1}, Batch:{batch_idx+1} Iteration:{batch_idx+1} loss:{loss.item()}")
                
                #mlflow.log_metric(key="training_avg_loss", value=(cum_train_loss/len(loop)), step=epoch+1)
                
                try:
                    mlflow.log_metric(key="training_loss", value=loss.cpu().item(), step=epoch+1)
                    mlflow.log_metric(key="training_total_time_epoch", value=loop._time(), step=epoch+1)
                except Exception as e:
                    print(e)
                    pass 
            
            # Monai Dice Metric
            # aggregate the final mean dice result
            metric = dice_metric.aggregate().item()
            self.monai_dice_training.append(metric)            
            # reset the status for next validation round
            dice_metric.reset()
            # To avoid division by zero    
            denominator_iou = max(denominator_iou,1e-10)
            accuracy = (num_correct/num_pixels)          
            # Normal metric calculation
            self.train_dice_complete.append(dice_score/len(loop))
            self.train_iou_complete.append(iou/len(loop))
            self.train_precision.append(precision/len(loop))
            self.train_recall.append(recall/len(loop))
            self.train_f1.append(f1/len(loop))

            dice_score = (dice_score/denominator_iou)
            iou_sec = iou / denominator_iou
            iou = (iou/denominator_iou)
            logging.info(f"----- Iou in Training Dataset for {epoch+1}: {num_correct}/{num_pixels} =  {accuracy:.2f} % & Dice: {dice_score:.2f} % & Iou {iou:.5f}%")
            print(f"[TRAINING] Avg Loss: {(cum_train_loss/len(loop)):.2f}")
            print(f"[TRAINING] Avg Dice Score: {dice_score:.5f}")
            print(f"[TRAINING] Avg Monai Dice Score: {metric:.5f}")
            print(f"[TRAINING] Avg IoU: {iou:.5f} ~ Avg IoU (2nd): {iou_sec:.5f}")
            print(f"[TRAINING] Recall: {(recall/len(loop)):.2f}")
            print(f"[TRAINING] Precision: {(precision/len(loop)):.2f}")
            print(f"[TRAINING] F1: {(f1/len(loop)):.2f}")
            self.train_accuracy.append(accuracy)
            self.train_dice.append(dice_score)
            self.train_iou.append(iou)
            #self.monai_dice_training.append(metric)
            #self.train_iou_2.append(iou_sec)
            # Appending the loss average
            self.train_loss_values.append((cum_train_loss/len(loop)))            
            #self.train_loss.append(loss.item())
            #print(plot_train_loss)
            #saveplot(plot_train_loss, f"Training_Loss_Epoch_{epoch+1}")
            # loggin metrics in MLFLOW
            try:
                mlflow.log_metric(key="training_avg_loss", value=(cum_train_loss/len(loop)), step=epoch+1)
                mlflow.log_metric(key="training_dice_score", value=dice_score, step=epoch+1)
                mlflow.log_metric(key="training_iou", value=iou, step=epoch+1)
            except Exception as e:
                print(e)
                pass
            
            # Plot training metrics
            self.plotPerformanceTrainingMetrics()

           # ===============================
            # ======== Validation ===========
            # ===============================
            num_correct = 0
            num_pixels = 0
            dice_score = 0
            iou = 0
            accum_val_loss = 0
            plot_loss = []
            denominator_iou = 0
            mean_dice_val = 0
            recall = 0
            precision = 0
            f1 = 0
            logging.info("Measuring model performance in Validation Dataset ====")
            self.model.eval()  
            with torch.no_grad():
                loop = tqdm(self.val_loader)
                for batch_idx, (data) in enumerate(loop):                                    
                    x = data["image"]
                    y = data["mask"]
                    x = x.permute(0,1,4,2,3)
                    y = y.permute(0,1,4,2,3)

                    x = x.float().to(self.device)
                    y = y.float().to(self.device).float()
                    # Pixel by pixel comparison to determine the accuracy             
                    predictions = self.model(x)
                    # Apply Sigmoid for computing the loss as it is a classification problem
                    predictions = torch.sigmoid(predictions)
                    val_loss = self.loss(predictions, y)                    
                    preds = (predictions > threshold_value).float()
                    num_correct += (preds == y).sum()
                    num_pixels += torch.numel(preds)                                       
                    

                    # Ony consider those images with higlighted area 
                    if torch.sum(y) > 0:
                        denominator_iou +=1
                    dice_score += diceScore(preds, y)
                    iou += intersectionOverUnion(preds,y)
                    f1 += f1Score(predictions,y)
                    val1, val2 = precisionAndRecall(predictions,y)
                    precision += val1
                    recall += val2
                    
                    # # Saving images                       
                    #x = x.resize_((x.shape[0], 1, x.shape[2], x.shape[3]))
                    if batch_idx % 2 == 0:
                        result = torch.cat((x[:, :, 10, :, :],
                                        y[:, :, 10, :, :],
                                        predictions[:, :, 10, :, :], 
                                        predictions_[:, :,10, :, :]), dim=0)
                        grid = torchvision.utils.make_grid(result, nrow=BATCH_SIZE, padding=100)
                        torchvision.utils.save_image(grid, f'{self.folder_output}testing_{batch_idx}_{date.today()}.png')
                        # torchvision.utils.save_image(x, f"{self.folder_output}/original_{batch_idx}_{date.today()}.png")                    

                    # compute metric for current iteration
                    preds = post_pred(preds)
                    y = post_label(y)
                    dice_metric(y_pred=preds, y=y)

                    accum_val_loss += val_loss.item()
                    plot_loss.append(val_loss.item())
                    loop.set_postfix(set="validation", exp=self.experiment_name, epoch=epoch+1, batch=batch_idx+1, loss=f"{val_loss.item():.2f}")
                    logging.info(f"==Validation== Epoch:{epoch+1}, Batch:{batch_idx+1},  loss:{val_loss.item()}")           

                    try:
                        mlflow.log_metric(key="val_loss", value=val_loss.cpu().item(), step=epoch+1)
                        mlflow.log_metric(key="val_total_time_epoch", value=loop.total, step=epoch+1)
                    except:
                        pass   
                
            # Monai Dice Metric
            # aggregate the final mean dice result
            metric = dice_metric.aggregate().item()
            self.monai_dice_testing.append(metric)   
            # reset the status for next validation round
            dice_metric.reset()
            accuracy = (num_correct/num_pixels)
            self.val_dice_complete.append(dice_score/len(loop))
            self.val_iou_complete.append(iou/len(loop))            
            self.val_precision.append(precision/len(loop))
            self.val_recall.append(recall/len(loop))
            self.val_f1.append(f1/len(loop))
            dice_score = (dice_score/denominator_iou)
            iou_score = (iou/denominator_iou)

            #iou_second = (iou/denominator_iou)
            logging.info(f"Got {num_correct}/{num_pixels} with acc {accuracy:.2f}")
            logging.info(f"Dice score: {dice_score:.2f}")
            logging.info(f"IoU score: {iou_score:.2f}")
            print(f"[VALIDATION] Loss: {val_loss.item():.2f}")
            print(f"[VALIDATION] Dice score: {dice_score:.5f}")
            print(f"[VALIDATION] Monai Dice score: {metric:.5f}")
            print(f"[VALIDATION] IoU: {iou_score:.5f}")
            print(f"[VALIDATION] Avg Loss: {(accum_val_loss/len(loop)):.2f}")
            print(f"[VALIDATION] Total validation loss in the epochs: {val_loss:.2f} => Avg Loss: {(accum_val_loss/len(loop)):.2f} ({batch_idx+1})")             
            print(f"[VALIDATION] Recall: {(recall/len(loop)):.2f}")
            print(f"[VALIDATION] Precision: {(precision/len(loop)):.2f}")
            print(f"[VALIDATION] F1: {(f1/len(loop)):.2f}")
            print(f"[VALIDATION] Total validation loss in the epochs: {val_loss:.2f} => Avg Loss: {(accum_val_loss/len(loop)):.2f} ({batch_idx+1})")             
            
            self.val_accuracy.append(accuracy)
            self.val_dice.append(dice_score)
            # Appending the average loss
            self.val_loss_values.append(accum_val_loss/len(loop))
            self.val_iou.append(iou_score)
            #self.val_monai_dice.append(metric)
            #self.val_iou_2.append(iou_second)
            #self.test_loss.append(val_loss.item())
            try:
                mlflow.log_metric(key="test_dice_score", value=dice_score)
                mlflow.log_metric(key="val_avg_loss", value=accum_val_loss/len(loop), step=epoch+1)
                mlflow.log_metric(key="test_iou_score", value=iou_score)
            except Exception as e:
                pass       
            # Plot testing metrics
            #self.plotPerformanceTestingMetrics()             
            #self.plotPerformanceValidationMetrics()
            #self.plotResultMetrics()

            # ==== Update scheduler accordingly ========
            if scheduler_config == "MultiStepLR":
                scheduler.step()
            elif scheduler_config == "ReduceLROnPlateau":
                scheduler.step(accum_val_loss/len(loop))           
            
            
            print(f"------ Learning rate {self.optimizer.param_groups[0]['lr']}----- in epoch: {epoch+1}")
            logging.info(f"------ Learning rate {self.optimizer.param_groups[0]['lr']}----- in epoch: {epoch+1}")
            try:
                self.learning_rate = self.optimizer.param_groups[0]['lr']
                self.lr_values.append(self.learning_rate)
                print(f"{self.learning_rate} changed!")
                #mlflow.log_param("learning_rate", self.learning_rate)
            except:
                pass
            
            self.plotResultMetrics()
            ConcatLossDiceIoULR(folder=f"{PLOTS_FOLDER}{self.experiment_name}/")

            # ===============================
            # ======= Saving Model  =========
            # ===============================
            # save current model with this accuracy and dice metrics
            if dice_score > self.best_model_acc:                
                self.best_model_acc = dice_score
                torch.save({
                        'model': self.model._get_name(),
                        'epochs': self.epochs,
                        "learning_rate": self.learning_rate,
                        'model_state_dict': self.model.state_dict(),
                        'optimizer_state_dict': self.optimizer.state_dict(),
                        'loss_function': self.optimizer_name,
                        "batch": self.batch_size,
                        'tr_loss': self.train_loss_values,
                        "tr_dice": self.train_dice,
                        "tr_iou": self.train_iou,
                        'val_loss': self.val_loss_values,
                        "val_dice": self.val_dice,
                        "val_iou": self.val_iou,
                        #'test_sample': 0,
                        "training_sample": len(self.train_loader),
                        "val_sample": len(self.val_loader)
                        }, f'{self.folder_model}{self.model._get_name()}_{self.optimizer_name}_{date.today()}.pt')       
                try:            
                    mlflow.pytorch.log_model(self.model, self.model._get_name(), registered_model_name=self.model._get_name())                               
                except:
                    pass
            
            # ==== Early Stopping ========
            # early_stopping needs the validation loss to check if it has decresed            
            early_stopping(accum_val_loss/len(loop))            
            if early_stopping.early_stop:
                print("Early stopping")
                break
            

        # ===============================
        # ======= Record Results  =========
        # ===============================
        report = {
            "date": date.today(),
            "experiment" : self.experiment_name,
            'model': self.model._get_name(),
            'epochs': self.epochs,
            "learning_rate": self.initial_lr,
            "weight_decay": self.weight_decay,                
            'momentum': self.momentum,
            "batch": self.batch_size,
            "optimizer" : self.optimizer_name,
            "training_sample": MAX_NUM_SAMPLE_TRAIN,
            "validation_sample": MAX_NUM_SAMPLE_VAL,
            "testing_sample" : MAX_NUM_SAMPLE_TEST,
            'training_avg_loss': np.mean(self.train_loss_values),
            'training_last_loss': self.train_loss_values[-1],
            "training_avg_dice": np.mean(self.train_dice),
            "training_last_dice": self.train_dice[-1],
            "training_avg_iou": np.mean(self.train_iou),
            "training_last_iou": self.train_iou[-1],
            'training_avg_f1': np.mean(self.train_f1),
            'training_avg_precision': np.mean(self.train_precision),
            'training_avg_recall': np.mean(self.train_recall),                
            'validation_avg_loss': np.mean(self.val_loss_values),
            'validation_last_loss': self.val_loss_values[-1],
            "validation_avg_dice": np.mean(self.val_dice),
            "validation_last_dice": self.val_dice[-1],
            "validation_avg_iou": np.mean(self.val_iou),
            "validation_last_iou": self.val_iou[-1],
            'validation_avg_f1': np.mean(self.val_f1),
            'validation_avg_precision': np.mean(self.val_precision),
            'validation_avg_recall': np.mean(self.val_recall)

        }
        df = pd.DataFrame(report, index=[0])
        df.to_csv(REPORT, mode='a', index=False, header=False)




    def plotPerformanceTrainingMetrics(self):
        """Save plots for performance dice and accuraccy in training dataset """        
        logging.info(f"Saving training plots")
        metrics = [
                {"train_dice": self.train_dice},
                {"train_iou": self.train_iou},
                {"train_avg_loss": self.train_loss_values},
                {"train_monai_dice": self.monai_dice_training},
                {"train_dice_complete": self.train_dice_complete},
                {"train_iou_complete": self.train_iou_complete},
                 {"train_precision": self.train_precision},
                {"train_recall": self.train_recall},
                {"train_f1": self.train_f1}
                ]
        [saveplot(values=m[list(m.keys())[0]], name=list(m.keys())[0], sample=len(self.train_loader), folder=self.folder_plots, decimals=5) for m in metrics]        

    
    def plotPerformanceTestingMetrics(self):
        """Save plots for performance dice and accuraccy in testing dataset """        
        logging.info(f"Saving testing plots")
        metrics = [
                {"test_dice": self.test_dice},
                {"test_avg_loss": self.test_loss_values}                
                ]
        [saveplot(values=m[list(m.keys())[0]], name=list(m.keys())[0], sample=len(self.test_loader), folder=self.folder_plots, decimals=5) for m in metrics]       
       
    
    def plotPerformanceValidationMetrics(self):
        """Save plots for performance dice and accuraccy in validation dataset """        
        logging.info(f"Saving testing plots")
        metrics = [
                    {"val_dice": self.val_dice},
                    {"val_avg_loss": self.val_loss_values},
                    {"val_avg_iou": self.val_iou},
                    {"val_monai_dice": self.monai_dice_testing},
                    {"val_dice_complete": self.val_dice_complete},
                    {"val_iou_complete": self.val_iou_complete},
                    {"val_precision": self.val_precision},
                    {"val_recall": self.val_recall},
                    {"val_f1": self.val_f1}
                ]
        [saveplot(values=m[list(m.keys())[0]], name=list(m.keys())[0], sample=len(self.val_loader), folder=self.folder_plots, decimals=5) for m in metrics]

    def plotResultMetrics(self):
        logging.info(f"Showing results for comparison training vs validation")
        logging.info(f"Showing results for comparison training vs validation")        
        metrics = [
                    {f"Loss_WD{self.weight_decay}_Mom{self.momentum}_{self.optimizer_name}": [self.train_loss_values,self.val_loss_values]},
                    {f"Dice": [self.train_dice,self.val_dice]},
                    {"IoU": [self.train_iou,self.val_iou]},
                    {"Monai Dice": [self.monai_dice_training, self.monai_dice_testing]},
                    {"Complete Dice": [self.train_dice_complete, self.val_dice_complete]},
                    {"Complete Iou": [self.train_iou_complete, self.val_iou_complete]},
                    {"Precision": [self.train_precision,self.val_precision]},
                    {"Recall": [self.train_recall,self.val_recall]},
                    {"F1": [self.train_f1,self.val_f1]},
                    {"LearningRate": [self.lr_values]}
                ]

        [saveComparisonPlot(values=m[list(m.keys())[0]], name=list(m.keys())[0], sample=len(self.val_loader), folder=self.folder_plots, lr=self.learning_rate) for m in metrics]
