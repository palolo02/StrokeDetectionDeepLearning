import sched
from tabnanny import verbose
from sqlalchemy import values
from tqdm import tqdm
import mlflow
import torch
from datetime import date
import torchvision
import torch.nn as nn
import matplotlib.pyplot as plt
import logging
import pandas as pd
import numpy as np
from datetime import datetime, date
from torch.optim.lr_scheduler import ReduceLROnPlateau, ExponentialLR, StepLR, MultiStepLR, CyclicLR
import math
from torchvision.utils import save_image
torch.cuda.empty_cache()
from readConfig import (EARLY_STOPPING_ENABLE, EARLY_STOPPING_PATIENCE, ENABLED, HOST, LOGS_DIR, MODELS_FOLDER, OUTPUT_FOLDER, REPORT,
                        PLOTS_FOLDER, THRESHOLD, configuration, MAX_NUM_SAMPLE_TRAIN, MAX_NUM_SAMPLE_TEST, MAX_NUM_SAMPLE_VAL)
from utils.helper import (network_parameters, saveplot, saveComparisonPlot, diceScore, 
                        intersectionOverUnion, getLRScheduler, precisionAndRecall, f1Score, 
                        ConcatLossDiceIoULR)
from monai.metrics import DiceMetric
from dataPipeline.earlyStopping import EarlyStopping

class ModelPerformanceSteps():
    """Encapsulates the steps of the machine learning model to keep track of its performance"""
    

    def __init__(self, model, train_loader, test_loader, val_loader, experiment_name, epochs, loss, loss_name,
                optimizer, scheduler, learning_rate, batch_size, device, weight_decay, momentum, isSigmoid) -> None:
        self.isSigmoid = isSigmoid
        self.model = model
        self.experiment_name = experiment_name
        self.train_loader = train_loader
        self.test_loader = test_loader
        self.val_loader = val_loader            
        self.epochs = epochs
        self.device = device
        self.scheduler = scheduler
        self.scheduler_name = scheduler
        self.loss = loss
        self.loss_function = loss_name
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
        #self.train_iou_2 = []
        self.train_loss_values = []
        self.test_accuracy = []
        self.test_dice = [] 
        self.test_loss_values = []
        self.val_accuracy = []
        self.val_dice = []         
        self.val_loss_values = []
        self.val_iou = [] 

        self.val_dice_complete = []
        self.train_dice_complete = []
        self.val_iou_complete = []
        self.train_iou_complete = []

        self.monai_dice_training = []
        self.monai_dice_testing = []
        #self.val_iou_2 = [] 

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
        """ 
        Run training and validation processes and keep track of parameters in MLflow.
        Arguments: None            
        Returns: None            
        """        
        try:
            print("MLFlow server found! Recording metrics in experiment")
            with mlflow.start_run():
                #self.trainingModel()
                self._training_and_testing()
        except Exception as e:
            print("Not MLFlow server found. Recording metrics locally.\n", e)
            #self.trainingModel()
            self._training_and_testing()
    
    def _training_and_testing(self):
        """ 
        Run training and validation processes and keep track of parameters in MLflow.
        Arguments: None            
        Returns: None            
        """ 
        
        if self.scheduler_name != "NA":            
            self.scheduler = getLRScheduler(self.scheduler_name, self.optimizer, configuration["scheduler_config"][self.scheduler_name])
        early_stopping = EarlyStopping(patience=EARLY_STOPPING_PATIENCE, trace_func=logging.info)
        
        print(f"Initial Learning rate: {self.learning_rate}")
        try:
            mlflow.log_param("learning_rate", self.learning_rate)
            mlflow.log_param("epochs", self.epochs)
            mlflow.log_param("optimizer",self.optimizer_name)
            mlflow.log_param("loss_function", self.loss._get_name().split(".")[-1])
            mlflow.log_param("batch_size", self.batch_size)
            mlflow.log_param("training_sample", len(self.train_loader))            
            mlflow.log_param("validation_sample", len(self.val_loader))
        except Exception as e:
            print("logging parameters")
            print(e)
            pass        
        
        # Running epochs
        for epoch in range(self.epochs):
            logging.info(f"Runing epoch {epoch+1} / {self.epochs}...")

            self._training(epoch+1)
            self._validation(epoch+1)
            
            # Plotting results of experiment
            print(f"Plotting results")
            self.plotResultMetrics()
            ConcatLossDiceIoULR(folder=f"{PLOTS_FOLDER}{self.experiment_name}/")

            # Update scheduler accordingly
            if self.scheduler_name == "MultiStepLR":
                self.scheduler.step()                
            elif self.scheduler_name == "ReduceLROnPlateau":
                self.scheduler.step(self.val_loss_values[-1])
            
            self.learning_rate = self.optimizer.param_groups[0]['lr']
            self.lr_values.append(self.learning_rate)
                
            

            # Save model with best accuracy
            if self.val_dice[-1] > self.best_model_acc:                
                self.best_model_acc = self.val_dice[-1]
                self._saveModel()
            
            # ==== Early Stopping ========
            # early_stopping needs the validation loss to check if it has decresed       
            if EARLY_STOPPING_ENABLE:     
                early_stopping(self.val_loss_values[-1])
                if early_stopping.early_stop:
                    logging.info(f"Early stopping executed! Loss:\n")
                    print("Early stopping executed!")
                    logging.info(self.val_loss_values)                
                    break
        
        # Report results of experiment    
        self._reportResults()
    
    def _saveModel(self):
        """ 
        Saves the current model under the models' folder.
        Arguments: None            
        Returns: None            
        """          
        print("--------- Saving model version ----------")        
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
                #'test_sample': len(self.test_loader),
                "training_sample": len(self.train_loader),
                "val_sample": len(self.val_loader)
                }, f'{self.folder_model}{self.model._get_name()}_{self.optimizer_name}_{date.today()}.pt')       
        try:
            mlflow.pytorch.log_model(self.model, self.model._get_name(), registered_model_name=self.model._get_name())                               
        except:
            pass

    def _reportResults(self):
        """ 
        Stores the results and the hyper parameters of the model in the csv file specified under the config file.
        Arguments: None            
        Returns: None            
        """ 
        # =================================
        # ======= Record Results  =========
        # =================================
        report = {
            "date": date.today(),
            "experiment" : self.experiment_name,
            'model': self.model._get_name(),
            'parameters': network_parameters(self.model),
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

    def _training(self, epoch):
        """ 
        Performs one entire training cycle for the specified epoch.
        Arguments: 
            epoch (int) = Number of epoch for trainnig            
        Returns: None            
        """        
        threshold_value = THRESHOLD
        #dice_metric = DiceMetric(include_background=True, reduction="mean", get_not_nans=False)        
        # Define metrics for measuring performance
        cum_loss = 0        
        num_correct = 0
        num_pixels = 0
        dice_score = 0
        iou = 0        
        recall = 0
        precision = 0
        f1 = 0        
        # Data Loader iteration       
        loop = tqdm(self.train_loader)
        self.model.train()
        logging.info(f"Starting training for Epoch:{epoch}")
        # Iteration over the batch of images
        for batch_idx, (x,y) in enumerate(loop):                  
            x = x.float().to(self.device)
            y = y.float().to(self.device)            
            predictions = self.model(x).to(self.device)
            if self.isSigmoid:
                predictions = torch.nn.Sigmoid()(predictions)                            
            loss = self.loss(predictions, y)            
            predictions_ = (predictions >= threshold_value).float()                                        
            # Save sample image to visualize results every 5 epoch
            if batch_idx % 5 == 0:
                result = torch.cat((x,y,predictions, predictions_), dim=0)
                grid = torchvision.utils.make_grid(result, nrow=self.batch_size, padding=100)
                torchvision.utils.save_image(grid, f'{self.folder_output}training_{date.today()}.png')                                             
            # calculate accuracy and dice score            
            num_correct += (predictions_ == y).sum().item()
            num_pixels += torch.numel(predictions_)            
            dice_score += diceScore(predictions, y)
            iou += intersectionOverUnion(predictions,y)
            f1 += f1Score(predictions,y)
            val1, val2 = precisionAndRecall(predictions,y)
            precision += val1
            recall += val2

            # backward prop
            # According to pytorch doc
            # https://pytorch.org/tutorials/recipes/recipes/tuning_guide.html
            # for param in self.model.parameters():
            #     param.grad = None           
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()           

            # Adding the loss for the current batch
            cum_loss += loss.item()
            # update tqdm loop

            # Update scheduler accordingly
            if configuration["hyper_parameters"]["scheduler"] == "CyclicLR":
                self.scheduler.step()
                self.learning_rate = self.optimizer.param_groups[0]['lr']
                self.lr_values.append(self.learning_rate)
                
                         
            loop.set_postfix(set="trainig", exp=self.experiment_name, loss=loss.item(), 
                            epoch=epoch, batch=batch_idx+1,
                            lr=self.learning_rate)
            #logging.info(f"==Training== Epoch:{epoch+1}, Batch:{batch_idx+1} loss:{loss.item()}")

            
        


        logging.info(f"calculating measures for Epoch:{epoch}")                 
        # Storing metrics for performance
        accuracy = (num_correct/num_pixels)
        iou = (iou/len(loop))
        dice_score = (dice_score/len(loop))

        self.train_precision.append(precision/len(loop))
        self.train_recall.append(recall/len(loop))
        self.train_f1.append(f1/len(loop))
        self.train_accuracy.append(accuracy)
        self.train_dice.append(dice_score)
        self.train_iou.append(iou)        
        self.train_loss_values.append((cum_loss/len(loop)))                    

        #logging.info(f"----- Iou in Training Dataset for {epoch+1}: {num_correct}/{num_pixels} =  {accuracy:.2f} % & Dice: {dice_score:.2f} % & Iou {iou:.5f}%")
        print(f"[TRAINING] Avg Loss: {(cum_loss/len(loop)):.2f}")
        print(f"[TRAINING] Avg Dice Score: {dice_score:.5f}")
        print(f"[TRAINING] Avg IoU: {iou:.5f}")
        print(f"[TRAINING] Recall: {(recall/len(loop)):.2f}")
        print(f"[TRAINING] Precision: {(precision/len(loop)):.2f}")
        print(f"[TRAINING] F1: {(f1/len(loop)):.2f}")
        
        try:
            mlflow.log_metric(key="training_avg_loss", value=(cum_loss/len(loop)), step=epoch)
            mlflow.log_metric(key="training_dice_score", value=dice_score, step=epoch)
            mlflow.log_metric(key="training_iou", value=iou, step=epoch)
        except Exception as e:
            print(e)
            pass

        #Plot training metrics
        #self.plotPerformanceTrainingMetrics()

    def _validation(self, epoch):
        """ 
        Performs one entire validation cycle for the specified epoch.
        Arguments: 
            epoch (int) = Number of epoch for validation            
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
            loop = tqdm(self.val_loader)
            logging.info(f"Starting training for Epoch:{epoch}")
            # Iteration over the batch of images
            for batch_idx, (x,y) in enumerate(loop):                  
                x = x.float().to(self.device)
                y = y.float().to(self.device)            
                predictions = self.model(x).to(self.device)
                if self.isSigmoid:
                    predictions = torch.nn.Sigmoid()(predictions)
                loss = self.loss(predictions, y)            
                predictions_ = (predictions >= threshold_value).float()                                        
                # Save sample image to visualize results every 5 epoch
                if batch_idx % 5 == 0:
                    result = torch.cat((x,y,predictions, predictions_), dim=0)
                    grid = torchvision.utils.make_grid(result, nrow=self.batch_size, padding=100)
                    torchvision.utils.save_image(grid, f'{self.folder_output}validation_{date.today()}.png')                                             
                # calculate accuracy and dice score            
                num_correct += (predictions_ == y).sum().item()
                num_pixels += torch.numel(predictions_)
                dice_score += diceScore(predictions, y)
                iou += intersectionOverUnion(predictions,y)
                f1 += f1Score(predictions,y)
                val1, val2 = precisionAndRecall(predictions,y)
                precision += val1
                recall += val2

                # Adding the loss for the current batch
                cum_loss += loss.item()
                # update tqdm loop
                            
                loop.set_postfix(set="validation", exp=self.experiment_name, loss=loss.item(), epoch=epoch, batch=batch_idx+1)
                #logging.info(f"==Training== Epoch:{epoch+1}, Batch:{batch_idx+1} loss:{loss.item()}")
            logging.info(f"calculating measures for Epoch:{epoch}")                 
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
            
            try:
                mlflow.log_metric(key="val_avg_loss", value=(cum_loss/len(loop)), step=epoch)
                mlflow.log_metric(key="val_dice_score", value=dice_score, step=epoch)
                mlflow.log_metric(key="val_iou", value=iou, step=epoch)
            except Exception as e:
                print(e)
                pass

        #Plot training metrics
        #self.plotPerformanceTrainingMetrics()

    

    def trainingModel(self):
        """ 
        (Obsolete) Performs the entire training and validation cycle for different epochs.
        Arguments: None            
        Returns: None            
        """     

        scheduler_config = configuration["hyper_parameters"]["scheduler"]
        scheduler = getLRScheduler(scheduler_config, self.optimizer,configuration["scheduler_config"][scheduler_config])
        early_stopping = EarlyStopping(patience=EARLY_STOPPING_PATIENCE, trace_func=logging.info)
        #scaler = torch.cuda.amp.grad_scaler.GradScaler()        
        #scheduler = ReduceLROnPlateau(self.optimizer, mode='min', patience=10, factor=0.1, verbose=True, cooldown=0, min_lr=1e-9,) # Update when the monitored amount has stopped decreasing
        #scheduler = ExponentialLR(self.optimizer, gamma=0.1)
        #scheduler = StepLR(self.optimizer, step_size=3, gamma=0.1, verbose=True)
        #scheduler = MultiStepLR(self.optimizer, milestones=[10,50,80], gamma=0.1, verbose=True)
        # Logging MLFlow parameters
        print(f"Initial Learning rate: {self.learning_rate}")
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
        threshold_value = THRESHOLD
        dice_metric = DiceMetric(include_background=True, reduction="mean", get_not_nans=False)
       
        # Running epochs
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
            iou = 0
            denominator_iou = 0
            mean_dice_val = 0
            recall = 0
            precision = 0
            f1 = 0
            self.model.train()
            for batch_idx, (x,y) in enumerate(loop):                  
                x = x.float().to(self.device)
                y = y.float().to(self.device)
                #y = y.float().unsqueeze(1).to(self.device) # with Albumentations                   
                #with torch.cuda.amp.autocast():
                predictions = self.model(x).to(self.device)                
                #loss = self.loss(predictions, y)clea
                # Monai Loss
                loss = self.loss(predictions, y)
                #predictions = torch.sigmoid(predictions) # Apply classification
                predictions_ = (predictions >= threshold_value).float()                
                
                # Monai Dice Metric
                dice_metric(y_pred=predictions_, y=y)           
                # save_image(x, 'saved_images/temp/img.png')                                      
                # save_image(y, 'saved_images/temp/mask.png')
                # save_image(predictions_, 'saved_images/temp/pred.png')                                                                 
                #x = x.resize_((x.shape[0], 1, x.shape[2], x.shape[3]))
                if batch_idx % 5 == 0:
                    result = torch.cat((x,y,predictions, predictions_), dim=0)
                    grid = torchvision.utils.make_grid(result, nrow=self.batch_size, padding=100)
                    torchvision.utils.save_image(grid, f'{self.folder_output}training_{date.today()}.png')                                 
                #torchvision.utils.save_image(grid, f'{self.folder_output}training_{batch_idx}_{date.today()}.png')                                 
                # calculate accuracy and dice score
                #predictions = torch.sigmoid(predictions)   
                num_correct += (predictions_ == y).sum().item()
                num_pixels += torch.numel(predictions_)
                #dice_score += (2*(predictions*targets).sum()) / ((predictions+targets).sum()+1.e-8)
                # Ony consider those images with higlighted area 
                if torch.sum(y) > 0: 
                    denominator_iou += 1
                dice_score += diceScore(predictions, y)                                    
                iou += intersectionOverUnion(predictions,y)                                    

                f1 += f1Score(predictions,y)
                val1, val2 = precisionAndRecall(predictions,y)
                precision += val1
                recall += val2

                # backward prop
                #self.optimizer.zero_grad()
                # According to pytorch doc
                # https://pytorch.org/tutorials/recipes/recipes/tuning_guide.html
                for param in self.model.parameters():
                    param.grad = None
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
                logging.info(f"==Training== Epoch:{epoch+1}, Batch:{batch_idx+1} loss:{loss.item()}")
                try:
                    mlflow.log_metric(key="training_loss", value=loss.cpu().item(), step=epoch+1)
                    mlflow.log_metric(key="training_total_time_epoch", value=loop._time(), step=epoch+1)
                except Exception as e:
                    print(e)
                    pass                
            # Monai metrics
            mean_dice_val = dice_metric.aggregate().item()
            self.monai_dice_training.append(mean_dice_val)            
            dice_metric.reset()
            # To avoid division by zero    
            denominator_iou = max(denominator_iou,1e-10)
            accuracy = (num_correct/num_pixels)        
            # add complete values
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
            print(f"[TRAINING] Avg IoU: {iou:.5f} ~ Avg Monai IoU (2nd): {mean_dice_val:.5f}")
            print(f"[TRAINING] Recall: {(recall/len(loop)):.2f}")
            print(f"[TRAINING] Precision: {(precision/len(loop)):.2f}")
            print(f"[TRAINING] F1: {(f1/len(loop)):.2f}")
            self.train_accuracy.append(accuracy)
            self.train_dice.append(dice_score)
            self.train_iou.append(iou)
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
            #self.plotPerformanceTrainingMetrics()

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
                for batch_idx, (x, y) in enumerate(loop):
                    # if batch_idx == 100:
                    #     print("Reading values")
                    x = x.float().to(self.device)
                    y = y.float().to(self.device)
                    #y = y.to(self.device).float().unsqueeze(1) # with Albumentations
                    # Pixel by pixel comparison to determine the accuracy             
                    predictions = self.model(x)
                    #val_loss = self.loss(predictions, y)
                    
                    #preds = torch.sigmoid(predictions)         
                    preds = (predictions > threshold_value).float()
                    # Monai Loss
                    val_loss = self.loss(predictions, y)
                    # Monai metric
                    dice_metric(y_pred=preds, y=y)
                    
                    num_correct += (preds == y).sum()
                    num_pixels += torch.numel(preds)
                    
                    # Ony consider those images with higlighted area 
                    if torch.sum(y) > 0: 
                        denominator_iou +=1                                 
                    dice_score += diceScore(predictions, y)
                    iou += intersectionOverUnion(predictions,y)
                    f1 += f1Score(predictions,y)
                    val1, val2 = precisionAndRecall(predictions,y)
                    precision += val1
                    recall += val2
                    # # Saving images
                    #x = x.resize_((x.shape[0], 1, x.shape[2], x.shape[3]))
                    if batch_idx % 10 == 0:
                        result = torch.cat((x,y,predictions, preds), dim=0)
                        grid = torchvision.utils.make_grid(result, nrow=self.batch_size, padding=100)
                        torchvision.utils.save_image(grid, f'{self.folder_output}original_{batch_idx}_{date.today()}.png')
                        # torchvision.utils.save_image(x, f"{self.folder_output}/original_{batch_idx}_{date.today()}.png")                    
                                        
                    accum_val_loss += val_loss.item()
                    plot_loss.append(val_loss.item())
                    loop.set_postfix(set="validation",exp=self.experiment_name, epoch=epoch+1, batch=batch_idx+1, loss=f"{val_loss.item():.2f}")
                    logging.info(f"==Validation== Epoch:{epoch+1}, Batch:{batch_idx+1},  loss:{val_loss.item()}")           

                    try:
                        mlflow.log_metric(key="val_loss", value=val_loss.cpu().item(), step=epoch+1)
                        mlflow.log_metric(key="val_total_time_epoch", value=loop.total, step=epoch+1)
                    except:
                        pass                    
                
                 # Monai metrics
                mean_dice_val = dice_metric.aggregate().item()
                self.monai_dice_testing.append(mean_dice_val)            
                dice_metric.reset()  
                denominator_iou = max(denominator_iou,1e-10)
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
                print(f"[VALIDATION] IoU: {iou_score:.5f}")
                print(f"[VALIDATION] Avg Loss: {(accum_val_loss/len(loop)):.2f} & Avg Monai Dice {mean_dice_val:.2f}")
                print(f"[VALIDATION] Recall: {(recall/len(loop)):.2f}")
                print(f"[VALIDATION] Precision: {(precision/len(loop)):.2f}")
                print(f"[VALIDATION] F1: {(f1/len(loop)):.2f}")
                print(f"[VALIDATION] Total validation loss in the epochs: {val_loss:.2f} => Avg Loss: {(accum_val_loss/len(loop)):.2f} ({batch_idx+1})")             
                
                self.val_accuracy.append(accuracy)
                self.val_dice.append(dice_score)
                # Appending the average loss
                self.val_loss_values.append(accum_val_loss/len(loop))
                self.val_iou.append(iou_score)
                #self.val_iou_2.append(iou_second)
                #self.test_loss.append(val_loss.item())
                try:
                    mlflow.log_metric(key="test_dice_score", value=dice_score)
                    mlflow.log_metric(key="val_avg_loss", value=accum_val_loss/len(loop), step=epoch+1)
                    mlflow.log_metric(key="test_iou_score", value=iou_score)
                except Exception as e:
                    pass            
                       
            # ==== Update scheduler accordingly ========
            if scheduler_config == "MultiStepLR":
                scheduler.step()
            elif scheduler_config == "ReduceLROnPlateau":
                scheduler.step(accum_val_loss/len(loop))           

            # Keeping track of the learning rate            
            print(f"------ Learning rate {self.optimizer.param_groups[0]['lr']}----- in epoch: {epoch+1}")
            logging.info(f"------ Learning rate {self.optimizer.param_groups[0]['lr']}----- in epoch: {epoch+1}")
            try:
                self.learning_rate = self.optimizer.param_groups[0]['lr']
                self.lr_values.append(self.learning_rate)
                print(f"{self.learning_rate} changed!")
                logging.info(f"{self.learning_rate} changed!")
                #mlflow.log_param("learning_rate", self.learning_rate)
            except:
                pass
            
            # Plot testing metrics
            #self.plotPerformanceTestingMetrics()             
            #self.plotPerformanceValidationMetrics()
            print(f"Plotting results")
            self.plotResultMetrics()
            ConcatLossDiceIoULR(folder=f"{PLOTS_FOLDER}{self.experiment_name}/")

            # ===============================
            # ======= Saving Model  =========
            # ===============================
            # save current model only of it has a better Dice Score
            print(f"--------- Comparing dice score for saving model {dice_score} vs {self.best_model_acc} ----------")
            if dice_score > self.best_model_acc:                
                self.best_model_acc = dice_score
                print("--------- Saving model version ----------")
                logging.info(f"--------- Saving current version of the model (best until this point: Dice Score: {dice_score}) ----------")
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
                        #'test_sample': len(self.test_loader),
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
                logging.info(f"Early stopping")
                print("Early stopping")
                break
           
        
        # =================================
        # ======= Record Results  =========
        # =================================
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
        """ 
        Run training and validation processes and keep track of parameters in MLflow.
        Arguments: None            
        Returns: None            
        """        
        logging.info(f"Saving training plots")
        metrics = [
                {"train_dice": self.train_dice},
                {"train_iou": self.train_iou},
                {"train_avg_loss": self.train_loss_values},
                #{"train_dice_complete": self.train_dice_complete},
                #{"train_iou_complete": self.train_iou_complete},
                # {"train_precision": self.train_precision},
                # {"train_recall": self.train_recall},
                # {"train_f1": self.train_f1},
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
                    #{"val_dice_complete": self.val_dice_complete},
                    #{"val_iou_complete": self.val_iou_complete},
                    # {"val_precision": self.val_precision},
                    # {"val_recall": self.val_recall},
                    # {"val_f1": self.val_f1},
                ]
        [saveplot(values=m[list(m.keys())[0]], name=list(m.keys())[0], sample=len(self.val_loader), folder=self.folder_plots, decimals=5) for m in metrics]
    
    def plotResultMetrics(self):
        """Save plots for comparison between training and validation"""        
        logging.info(f"Showing results for comparison training vs validation")        
        metrics = [
                    {f"Loss_WD{self.weight_decay}_Mom{self.momentum}_{self.optimizer_name}": [self.train_loss_values,self.val_loss_values]},                    
                    {f"Dice": [self.train_dice,self.val_dice]},
                    {"IoU": [self.train_iou,self.val_iou]},
                    #{"Monai Dice": [self.monai_dice_training,self.monai_dice_testing]},
                    #{"Complete Dice": [self.train_dice_complete,self.val_dice_complete]},
                    # {"Precision": [self.train_precision,self.val_precision]},
                    # {"Recall": [self.train_recall,self.val_recall]},
                    # {"F1": [self.train_f1,self.val_f1]},
                    {"LearningRate": [self.lr_values]}
                ]

        [saveComparisonPlot(values=m[list(m.keys())[0]], name=list(m.keys())[0], sample=len(self.val_loader), folder=self.folder_plots, lr=self.initial_lr) for m in metrics]
        
