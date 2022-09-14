import shutil
from tabnanny import check
import unittest
import random
from tqdm import tqdm
from unet.dataset import StrokeTestingModelDataset
from utils.transformations import CustomTrainingStrokeTrans
from PIL import Image, ImageDraw, ImageFont
from readConfig import IMAGE_WIDTH, IMAGE_HEIGHT, MAX_NUM_SAMPLE_TEST, BATCH_SIZE
from monai.networks.nets.unet import UNet
from monai.networks.nets.vnet import VNet
from monai.networks.nets.densenet import DenseNet
import torch
import shutil
import napari
import os
import torchvision
from utils.helper import diceScore, f1Score, getLoadedModel, getModel, getOptimizer, intersectionOverUnion, isSigmoid, network_parameters, precisionAndRecall
from utils.losses import BCEDiceLoss, getLossFunction
# Define the class you want to test
from utils.transformations import CustomTestStrokTrans, CustomComparisonStrokTrans
import matplotlib.pyplot as plt
from datetime import date
import pandas as pd
import numpy as np
from torchvision.utils import make_grid

class TestTransforms(unittest.TestCase):

    def test_CustomTrainingStrokeTrans(self):
        # load images
        pair = {
            "image" : Image.open("dataset/train/images/sub-r001s001_ses/0100.png"),
            "mask"  : Image.open("dataset/train/masks/sub-r001s001_ses_mask/0100.png")
        }
        # Apply transformation
        img_, mask_ = CustomTrainingStrokeTrans()(img=pair["image"], mask=pair["mask"])        
        
        # Evaluate same size after resizing to the value specified in the config file        
        self.assertEqual(img_.shape[1], IMAGE_WIDTH)
        self.assertEqual(img_.shape[2], IMAGE_HEIGHT)
        self.assertEqual(mask_.shape[1], IMAGE_WIDTH)
        self.assertEqual(mask_.shape[2], IMAGE_HEIGHT)
        self.assertEqual(mask_.shape[1], img_.shape[1])
        self.assertEqual(mask_.shape[2], img_.shape[2])



class TestModels(unittest.TestCase):
    
    def test_3d_unet_model(self):
        pass

    def test_2d_unet_model(self):
        model = UNet(
            spatial_dims=2,
            in_channels=1,
            out_channels=1,
            channels=(64, 128, 256),
            strides=(2, 2),
            num_res_units=2
        )
   
    def test_vnet_model(self):
        vnet_model = VNet(
            spatial_dims=3, 
            in_channels=1, 
            out_channels=1, 
            #act=('elu', {'inplace': True}), 
            dropout_prob=0.5, 
            dropout_dim=3, 
            #bias=False
        )
    
class TestResults(unittest.TestCase):

    def test_export_results_txt(self, results):
        
        report = {
            "date": date.today(),            
            'model' : results["model"],
            'image' : results["image"],
            'dsc'   : results["dsc"]
        }
        df = pd.DataFrame(report, index=[0])
        df.to_csv("validation/DiceScore.csv", mode='a', index=False, header=False)



    def test_report_results(self, results=None):
        # =================================
        # ======= Record Results  =========
        # =================================
        if results is None:
            results = {}
            results["model"]="TestModel"
            results["parameters"]=1254664
            results["avg_loss"]=1254664
            results["avg_dice_score"]=1254664
            results["avg_iou"]=1254664
            results["avg_recall"]=1254664
            results["avg_precision"]=1254664
            results["avg_f1_score"]=1254664
            results["std_dice_score"]=1254664
            results["std_iou"]=1254664
        
        report = {
            "date": date.today(),            
            'model': results["model"],
            'parameters': results["parameters"],
            'avg_loss': results["avg_loss"],
            "avg_dice_score": results["avg_dice_score"],
            "avg_iou": results["avg_iou"],
            "avg_recall": results["avg_recall"],
            "avg_precision": results["avg_precision"],
            'avg_f1_score': results["avg_f1_score"],
            'std_dice_score': results["std_dice_score"],
            'std_iou_score': results["std_iou"]
        }
        df = pd.DataFrame(report, index=[0])
        df.to_csv("results/Results.csv", mode='a', index=False, header=False)

    def test_predict_same_image(self):
        device = "cuda" if torch.cuda.is_available() else "cpu"        
        torch.manual_seed(100)
        batch_size = 1
        dataset = StrokeTestingModelDataset(batch_size=batch_size,max_num_samples=MAX_NUM_SAMPLE_TEST)
        print(f"Considering {MAX_NUM_SAMPLE_TEST} Images for testing")       
        test_loader = dataset.getTestDataloaderFromPNGImages()
        loop = tqdm(test_loader)
        loss_function = BCEDiceLoss()        
        folder = f"validation/comparison/"

        # Store every metric for individual models
        info_models = {}
        models = self.test_load_model()
        for _model in models:
            # Create a folder for each of the models
            if not os.path.exists(f'{folder}{_model["model"]["loaded_model"]}'):
                os.mkdir(folder)
            info_models[_model["model"]] = {}
            info_models[_model["model"]]["avg_loss"] = 0
            info_models[_model["model"]]["avg_dice_score"] = 0
            info_models[_model["model"]]["avg_iou"] = 0
            info_models[_model["model"]]["avg_recall"] = 0
            info_models[_model["model"]]["avg_precision"] = 0
            info_models[_model["model"]]["avg_f1_score"] = 0
            info_models[_model["model"]]["predictions"] = []
            info_models[_model["model"]]["values_dice_score"] = []
            info_models[_model["model"]]["values_iou"] = []

        #print(info_models)

        # variables
        threshold_value = 0.4
        with torch.no_grad():          
            # Iteration over the batch of images
            for batch_idx, (x,y) in enumerate(loop):
                x = x.float().to(device)
                y = y.float().to(device)                
                image_results = []
                image_results.append(x)
                image_results.append(y)
                # Need to predict same image for several models
                for m_dict in models:
                    model = m_dict["loaded_model"]
                    model.eval()
                    predictions = model(x).to(device)
                    if isSigmoid(m_dict["model"]):
                        predictions = torch.nn.Sigmoid()(predictions)                                    
                    loss = loss_function(predictions, y)
                    predictions_ = (predictions >= threshold_value).float() 
                    # Precision and recall
                    val1, val2 = precisionAndRecall(predictions,y)
                    # Add metrics                                    
                    info_models[m_dict["model"]]["avg_loss"] += loss.item()
                    info_models[m_dict["model"]]["avg_dice_score"] += diceScore(predictions, y)
                    info_models[m_dict["model"]]["avg_iou"] += intersectionOverUnion(predictions,y)
                    info_models[m_dict["model"]]["avg_recall"] += val2
                    info_models[m_dict["model"]]["avg_precision"] += val1
                    info_models[m_dict["model"]]["avg_f1_score"] += f1Score(predictions,y)
                    image_results.append(predictions)
                    # Values to calculate std
                    info_models[m_dict["model"]]["values_dice_score"].append(diceScore(predictions, y))
                    info_models[m_dict["model"]]["values_iou"].append(intersectionOverUnion(predictions,y))

                    loop.set_postfix(set="testing", model=m_dict['model'], loss=loss.item(), batch=batch_idx+1)

                # Save sample image to visualize results every 5 epoch
                if batch_idx % 3 == 0:
                    result = torch.cat((image_results), dim=0)
                    grid = torchvision.utils.make_grid(result, nrow=batch_size, padding=100)                        
                    torchvision.utils.save_image(grid, f'{folder}/{batch_idx}_{date.today()}.png')                                             
                    # calculate accuracy and dice score
                    
                #logging.info(f"==Training== Epoch:{epoch+1}, Batch:{batch_idx+1} loss:{loss.item()}")
            
            print(f"calculating measures")
            for m_dict in models:
                info_models[m_dict["model"]]["model"] = m_dict['model']
                info_models[m_dict["model"]]["parameters"] = network_parameters(m_dict["loaded_model"])                                
                info_models[m_dict["model"]]["avg_loss"] /= len(loop)                
                info_models[m_dict["model"]]["avg_dice_score"] /= len(loop)                
                info_models[m_dict["model"]]["avg_iou"] /= len(loop)                
                info_models[m_dict["model"]]["avg_recall"] /= len(loop)
                info_models[m_dict["model"]]["avg_precision"] /= len(loop)
                info_models[m_dict["model"]]["avg_f1_score"] /= len(loop)                
                info_models[m_dict["model"]]["std_dice_score"] = np.std(info_models[m_dict["model"]]["values_dice_score"], axis=0)
                info_models[m_dict["model"]]["std_iou"] = np.std(info_models[m_dict["model"]]["values_iou"], axis=0)
                self.test_report_results(results=info_models[m_dict["model"]])
                
    def test_predict_same_image_per_model(self):
        device = "cuda" if torch.cuda.is_available() else "cpu"        
        torch.manual_seed(200)
        batch_size = 1
        dataset = StrokeTestingModelDataset(batch_size=batch_size,max_num_samples=MAX_NUM_SAMPLE_TEST)
        print(f"Considering {MAX_NUM_SAMPLE_TEST} Images for testing")       
        test_loader = dataset.getTestDataloaderFromPNGImages()
        loop = tqdm(test_loader)
        loss_function = BCEDiceLoss()
        viewer = napari.Viewer()
        # Change folder to have access from Flask
        #folder = f"validation/comparison/model/"
        folder = "static/data/validation/"
        if os.path.exists(folder):
            shutil.rmtree(folder)  
        #if not os.path.exists(folder):
        os.mkdir(folder)
        
        subfolders = ["input","mask","prediction","comparison"]

        # Store every metric for individual models
        info_models = {}
        models = self.test_load_model()
        for _model in models:
             # Create a main folder for each model
            if os.path.exists(f'{folder}{_model["model"]}/'):
                shutil.rmtree(f'{folder}{_model["model"]}/') 
            #if not os.path.exists(f'{folder}{_model["model"]}/'):
            os.mkdir(f'{folder}{_model["model"]}/')
            
            for subfolder in subfolders:
                if os.path.exists(f'{folder}{_model["model"]}/{subfolder}/'):
                    shutil.rmtree(f'{folder}{_model["model"]}/{subfolder}/') 
                #if not os.path.exists(f'{folder}{_model["model"]}/{subfolder}/'):
                os.mkdir(f'{folder}{_model["model"]}/{subfolder}/')            
            
            info_models[_model["model"]] = {}
            info_models[_model["model"]]["avg_loss"] = 0
            info_models[_model["model"]]["avg_dice_score"] = 0
            info_models[_model["model"]]["avg_iou"] = 0
            info_models[_model["model"]]["avg_recall"] = 0
            info_models[_model["model"]]["avg_precision"] = 0
            info_models[_model["model"]]["avg_f1_score"] = 0
            info_models[_model["model"]]["predictions"] = []
            info_models[_model["model"]]["values_dice_score"] = []
            info_models[_model["model"]]["values_iou"] = []

        #print(info_models)

        # variables
        threshold_value = 0.2
        with torch.no_grad():          
            # Iteration over the batch of images
            for batch_idx, (x,y) in enumerate(loop):
                x = x.float().to(device)
                y = y.float().to(device)                
                image_results = []
                image_results.append(x)
                image_results.append(y)
                # Need to predict same image for several models
                for m_dict in models:
                    model = m_dict["loaded_model"]
                    model.eval()
                    predictions = model(x).to(device)
                    if isSigmoid(m_dict["model"]):
                        predictions = torch.nn.Sigmoid()(predictions)                                    
                    loss = loss_function(predictions, y)
                    predictions_ = (predictions >= threshold_value).float() 
                    # Precision and recall
                    val1, val2 = precisionAndRecall(predictions,y)
                    # Add metrics
                    _temp_dsc = diceScore(predictions, y)
                    info_models[m_dict["model"]]["avg_loss"] += loss.item()
                    info_models[m_dict["model"]]["avg_dice_score"] += _temp_dsc #diceScore(predictions, y)
                    info_models[m_dict["model"]]["avg_iou"] += intersectionOverUnion(predictions,y)
                    info_models[m_dict["model"]]["avg_recall"] += val2
                    info_models[m_dict["model"]]["avg_precision"] += val1
                    info_models[m_dict["model"]]["avg_f1_score"] += f1Score(predictions,y)
                    image_results.append(predictions)
                    # Values to calculate std
                    info_models[m_dict["model"]]["values_dice_score"].append(_temp_dsc)
                    info_models[m_dict["model"]]["values_iou"].append(intersectionOverUnion(predictions,y))

                    loop.set_postfix(set="testing", model=m_dict['model'], loss=loss.item(), batch=batch_idx+1)

                    # Save sample image to visualize results every 10 epoch
                    if batch_idx % 10 == 0:
                        #result = torch.cat((image_results), dim=0)
                        result = torch.cat((x,y,predictions), dim=0)
                        grid = torchvision.utils.make_grid(result, nrow=batch_size, padding=100)
                        #image_name = f'{folder}/{batch_idx}_{date.today()}.png'
                        image_name = f'{folder}{m_dict["model"]}/{m_dict["model"]}_{batch_idx}_{date.today()}.png'
                        torchvision.utils.save_image(grid, image_name)

                        self.save_individual_images(model=m_dict['model'], x=x, y=y, prediction=predictions, batch_idx=batch_idx, viewer=viewer)                       
                        
                        # calculate accuracy and dice score
                        results = {}
                        results["dsc"] = f"{_temp_dsc:.2f}"
                        results["image"] = image_name
                        results["model"] = m_dict["model"]
                        self.test_export_results_txt(results)

            print(f"calculating measures")
            for m_dict in models:
                info_models[m_dict["model"]]["model"] = m_dict['model']
                info_models[m_dict["model"]]["parameters"] = network_parameters(m_dict["loaded_model"])                                
                info_models[m_dict["model"]]["avg_loss"] /= len(loop)                
                info_models[m_dict["model"]]["avg_dice_score"] /= len(loop)                
                info_models[m_dict["model"]]["avg_iou"] /= len(loop)                
                info_models[m_dict["model"]]["avg_recall"] /= len(loop)
                info_models[m_dict["model"]]["avg_precision"] /= len(loop)
                info_models[m_dict["model"]]["avg_f1_score"] /= len(loop)                
                info_models[m_dict["model"]]["std_dice_score"] = np.std(info_models[m_dict["model"]]["values_dice_score"], axis=0)
                info_models[m_dict["model"]]["std_iou"] = np.std(info_models[m_dict["model"]]["values_iou"], axis=0)
                self.test_report_results(results=info_models[m_dict["model"]])
        viewer.close()
                
    def save_individual_images(self, model, x, y, prediction, batch_idx, viewer):
         
        folder = f"static/data/validation/{model}/"
        subfolders = ["input","mask","prediction","comparison"]        
        
        # -- Input --
        image_name = f'{folder}{subfolders[0]}/{batch_idx}_{date.today()}.png'
        torchvision.utils.save_image(x, image_name)

        # -- Mask --
        image_name = f'{folder}{subfolders[1]}/{batch_idx}_{date.today()}.png'
        torchvision.utils.save_image(y, image_name)

        # -- Prediction --
        image_name = f'{folder}{subfolders[2]}/{batch_idx}_{date.today()}.png'
        torchvision.utils.save_image(prediction, image_name)

        # -- Comparison --        
        viewer.add_image(x[0,:,:,:].detach().cpu().numpy(), name="Original")
        viewer.add_image(y[0,:,:,:].detach().cpu().numpy(), name="Mask", blending="additive", colormap="green")
        viewer.add_image(prediction[0,:,:,:].detach().cpu().numpy(), name="Prediction", blending="additive", colormap="red")        
        image_name = f'{folder}{subfolders[3]}/{batch_idx}_{date.today()}.png'
        viewer.screenshot(path=f"{image_name}", canvas_only=True, flash=False) 
        viewer.layers.pop()
        #torchvision.utils.save_image(x, image_name)
        #viewer.close()


    def test_final_results_per_model(self, n=5) :        
        folder = "static/data/validation/"
        images= []
        random.seed(100)

        # Headers
        W, H = (224,224)
        headers = ["Input", "Ground-truth", "Prediction", "Comparison", "Dice Score"]
        img_headers = []
        for header in headers:
            img = Image.new('RGB', (W, H), color = (0, 0, 0))
            d = ImageDraw.Draw(img)
            arial = ImageFont.truetype("arial.ttf", 40)
            w, h = arial.getsize(header)
            d.text(((W-w)/2,(H-h)/2), header, font=arial, fill="white")
            custom_header =  CustomComparisonStrokTrans()(img)
            custom_header = custom_header.unsqueeze(dim=0)
            img_headers.append(custom_header)

        for model in os.listdir(folder):
            
            # Read files within a model, shuffle results and then pick n to save it as image
            for file in os.listdir(f"{folder}{model}/input/"):
                images.append(f"{file}")
            random.shuffle(images)

            row_images = []
            # Append headers
            for i in range(len(img_headers)):
                row_images.append(img_headers[i])
            
            for i in range(n):
                input =  CustomComparisonStrokTrans()(img=Image.open(f"{folder}{model}/input/{images[i]}").convert("RGB"))
                mask =  CustomComparisonStrokTrans()(img=Image.open(f"{folder}{model}/mask/{images[i]}").convert("RGB"))
                prediction =  CustomComparisonStrokTrans()(img=Image.open(f"{folder}{model}/prediction/{images[i]}").convert("RGB"))
                comparison =  CustomComparisonStrokTrans()(img=Image.open(f"{folder}{model}/comparison/{images[i]}").convert("RGB"))                        

                input = input.unsqueeze(dim=0)            
                mask = mask.unsqueeze(dim=0)            
                prediction = prediction.unsqueeze(dim=0)
                comparison = comparison.unsqueeze(dim=0)            
                # Calculate DSC
                result = diceScore(mask, prediction)*100
                text = f"{result:.2f}%"                
                W, H = (224,224)
                img = Image.new('RGB', (W, H), color = (0, 0, 0))
                d = ImageDraw.Draw(img)
                arial = ImageFont.truetype("arial.ttf", 32)
                w, h = arial.getsize(text)
                d.text(((W-w)/2,(H-h)/2), text, font=arial, fill="white")

                dice_result =  CustomComparisonStrokTrans()(img)
                dice_result = dice_result.unsqueeze(dim=0)     
                row_images.append(input)
                row_images.append(mask)
                row_images.append(prediction)
                row_images.append(comparison)
                row_images.append(dice_result)
                #
                #row_images.append(result)
            grid_result = torch.cat(row_images, dim=0)
            grid = torchvision.utils.make_grid(grid_result, nrow=5, padding=100)
            torchvision.utils.save_image(grid, f'dataProcessing/Stroke/{model}Results.png')

            

    def test_predict_loader_metrics(self):
        device = "cuda" if torch.cuda.is_available() else "cpu"        
        torch.manual_seed(200)
        batch_size = 1
        dataset = StrokeTestingModelDataset(batch_size=batch_size,max_num_samples=150)        
        test_loader = dataset.getTestDataloaderFromPNGImages()
        loop = tqdm(test_loader)
        loss_function = BCEDiceLoss()

        for m_dict in self.test_load_model():
            print("="*20)            
            model = m_dict["loaded_model"]
            folder = f"validation/{m_dict['model']}/"
            if not os.path.exists(folder):
                os.mkdir(folder)
            # variables
            threshold_value = 0.8        
            cum_loss = 0        
            num_correct = 0
            num_pixels = 0
            dice_score = 0
            iou = 0                
            recall = 0
            precision = 0
            f1 = 0
            val_precision = []
            val_recall=[]
            val_f1=[]
            val_accuracy=[]
            val_dice=[]
            val_iou=[]
            val_loss_values = []
            model.eval()
            
            with torch.no_grad():                             
                # Iteration over the batch of images
                for batch_idx, (x,y) in enumerate(loop):                  
                    x = x.float().to(device)
                    y = y.float().to(device)
                    predictions = model(x).to(device)
                    if isSigmoid(m_dict["model"]):
                        predictions = torch.nn.Sigmoid()(predictions)
                    loss = loss_function(predictions, y)
                    predictions_ = (predictions >= threshold_value).float()
                                          
                    # calculate accuracy and dice score            
                    num_correct += (predictions_ == y).sum().item()
                    num_pixels += torch.numel(predictions_)
                    _temp_dsc = diceScore(predictions, y)
                    dice_score += _temp_dsc
                    iou += intersectionOverUnion(predictions,y)
                    f1 += f1Score(predictions,y)
                    val1, val2 = precisionAndRecall(predictions,y)
                    precision += val1
                    recall += val2

                    # Adding the loss for the current batch
                    cum_loss += loss.item()
                    # update tqdm loop

                    # Save sample image to visualize results every 5 epoch
                    if batch_idx % 5 == 0:
                        result = torch.cat((x,y,predictions), dim=0)
                        #result = torch.cat((x,y,predictions, predictions_), dim=0)
                        grid = torchvision.utils.make_grid(result, nrow=batch_size, padding=100)
                        name_image = f'{folder}{m_dict["model"]}_{batch_idx}_{date.today()}.png'
                        torchvision.utils.save_image(grid, name_image)
                        
                        results = {}
                        results["dsc"] = f"{_temp_dsc:.2f}"
                        results["image"] = name_image
                        results["model"] = m_dict['model']
                        self.test_export_results_txt(results)

                    loop.set_postfix(set="testing", model=m_dict['model'], loss=loss.item(), batch=batch_idx+1)
                    #logging.info(f"==Training== Epoch:{epoch+1}, Batch:{batch_idx+1} loss:{loss.item()}")
                print(f"calculating measures")
                # Storing metrics for performance
                accuracy = (num_correct/num_pixels)
                iou = (iou/len(loop))
                dice_score = (dice_score/len(loop))

                val_precision.append(precision/len(loop))
                val_recall.append(recall/len(loop))
                val_f1.append(f1/len(loop))
                val_accuracy.append(accuracy)
                val_dice.append(dice_score)
                val_iou.append(iou)
                val_loss_values.append((cum_loss/len(loop)))

                results = {}
                results["model"]=m_dict['model']
                results["parameters"]=network_parameters(model)
                results["avg_loss"]=cum_loss/len(loop)
                results["avg_dice_score"]=dice_score
                results["avg_iou"]=iou
                results["avg_recall"]=recall/len(loop)
                results["avg_precision"]=precision/len(loop)
                results["avg_f1_score"]=f1/len(loop)
                results["std_dice_score"]= np.std(val_dice)
                results["std_iou"] = np.std(val_iou)                
                #logging.info(f"----- Iou in Training Dataset for {epoch+1}: {num_correct}/{num_pixels} =  {accuracy:.2f} % & Dice: {dice_score:.2f} % & Iou {iou:.5f}%")
                print(f"[VALIDATION {m_dict['model']}] Avg Loss: {(cum_loss/len(loop)):.2f}")
                print(f"[VALIDATION {m_dict['model']}] Avg Dice Score: {dice_score:.5f}")
                print(f"[VALIDATION {m_dict['model']}] Avg IoU: {iou:.5f}")
                print(f"[VALIDATION {m_dict['model']}] Recall: {(recall/len(loop)):.2f}")
                print(f"[VALIDATION {m_dict['model']}] Precision: {(precision/len(loop)):.2f}")
                print(f"[VALIDATION {m_dict['model']}] F1: {(f1/len(loop)):.2f}")

                self.test_report_results(results=results)

    def test_predict_loader(self):
        device = "cuda" if torch.cuda.is_available() else "cpu"
        m_dict = self.test_load_model()[0]
        model = m_dict["loaded_model"]

        dataset = StrokeTestingModelDataset(batch_size=4,max_num_samples=150)        
        test_loader = dataset.getTestDataloaderFromPNGImages()

        # Data Loader iteration       
        loop = tqdm(test_loader)
        model.eval()
        with torch.no_grad():            
            for batch_idx, (x,y) in enumerate(loop):                  
                x = x.float().to(device)
                y = y.float().to(device)            
                predictions = model(x).to(device)
                print(predictions.shape)
                plt.imshow(predictions[0,0,:,:].cpu().detach(), cmap="gray")
                plt.show()
                plt.close()
            pass

    def test_single_predict(self):
        patient = "dataset/test/images/sub-r040s086_ses/0121.png"
        mask = "dataset/test/masks/sub-r040s086_ses_mask/0121.png"
        device = "cuda" if torch.cuda.is_available() else "cpu"


        
        # load first model
        m_dict = self.test_load_model()[0]

        # apply transform to image
        trans = CustomTestStrokTrans()
        img, msk  = trans(Image.open(patient), Image.open(mask))
        img = torch.unsqueeze(img,dim=0)
        msk = torch.unsqueeze(msk,dim=0)
        #img = torch.ones(4,1,224,224).float().to(device)
        #msk = torch.ones(4,1,224,224).float().to(device)
        img = img.float().to(device)
        msk = msk.float().to(device)
        print(img.shape)
        print(msk.shape)
        # print(type(img))
        # plt.imshow(img[0,:,:], cmap="gray")
        # plt.show()
        # plt.imshow(msk[0,:,:], cmap="gray")
        # plt.show()
        # plt.close()
        model = m_dict["loaded_model"]
        model.eval()
        with torch.no_grad():
            output = model(img)
            output = (output > 0.8).float()
            print(output.shape)
            plt.imshow(output[0,0,:,:].cpu().detach(), cmap="gray")
            plt.show()
            plt.imshow(msk[0,0,:,:].cpu().detach(), cmap="gray")
            plt.show()
            plt.close()



    def test_load_model(self):
        folder = "validation"
        models = []
        for file in os.listdir(folder):
            if ".pt" in file:
                print(f"\n{folder}/{file}")
                _model = self.test_load_single_Model(f"{folder}/{file}")
                models.append(_model)
        return models


    def test_load_single_Model(self, file="validation/AttentionUnet_Adam_2022-08-03.pt"):
        print(f"Load model from {file}")
        model_dict = {}
        device = "cuda" if torch.cuda.is_available() else "cpu"
        checkpoint = torch.load(file)
        print(checkpoint["model"])
        print(checkpoint["loss_function"])
        print(checkpoint["batch"])
        #print(checkpoint["val_dice"])        
        #model_dict["loss_function"] = getLossFunction("BCEDiceLoss")
        loadedModel = getModel(checkpoint["model"], device)       
        type_optimizer = checkpoint["loss_function"]
        optimizer = getOptimizer(type_optimizer,loadedModel, None)
        
        # ================== Load Model ====================
        loadedModel.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])        
        model_dict["loaded_model"] = loadedModel
        model_dict["optimizer"] = optimizer
        model_dict["batch"] = checkpoint["batch"]
        model_dict["model"] = checkpoint["model"]
        return model_dict
        
class TestDisplayKernels(unittest.TestCase):

    def test_show_kernels(self):
        # Visualize feature maps
        model_dict = self.test_load_single_Model()
        model = model_dict["loaded_model"]

        kernels = model.conv1.weight.detach().clone()
        kernels = kernels - kernels.min()
        kernels = kernels / kernels.max()
        img = make_grid(kernels)
        plt.imshow(img.permute(1, 2, 0))


    def test_show_Activation_Maps(self):        
        # Visualize feature maps
        model_dict = self.test_load_single_Model()
        model = model_dict["loaded_model"]

        activation = {}
        def get_activation(name):
            def hook(model, input, output):
                activation[name] = output.detach()
            return hook

        # #name_conv = "conv1" # conv_trans2
        # name_conv = "conv_trans1" # 
        # model.conv1.register_forward_hook(get_activation(name_conv))
        # data, _ = dataset[random.randint(0,60000)]
        # data.unsqueeze_(0)
        # output = model(data)

        # act = activation[name_conv].squeeze()
        # fig, axarr = plt.subplots(act.size(0))
        # for idx in range(act.size(0)):
        #     axarr[idx].imshow(act[idx])


    def test_load_single_Model(self, file="validation/UnetDeep3_Adam_2022-08-03.pt"):
        print(f"Load model from {file}")
        model_dict = {}
        device = "cuda" if torch.cuda.is_available() else "cpu"
        checkpoint = torch.load(file)
        print(checkpoint["model"])
        print(checkpoint["loss_function"])
        print(checkpoint["batch"])
        #print(checkpoint["val_dice"])        
        #model_dict["loss_function"] = getLossFunction("BCEDiceLoss")
        loadedModel = getModel(checkpoint["model"], device)       
        type_optimizer = checkpoint["loss_function"]
        optimizer = getOptimizer(type_optimizer,loadedModel, None)
        
        # ================== Load Model ====================
        loadedModel.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])        
        model_dict["loaded_model"] = loadedModel
        model_dict["optimizer"] = optimizer
        model_dict["batch"] = checkpoint["batch"]
        model_dict["model"] = checkpoint["model"]
        return model_dict
        


    
