
from sqlalchemy import false
import torch.nn as nn
import numpy as np
import torch
from .helper import diceScore
from monai.losses import DiceCELoss as DCL
from monai.losses import DiceLoss as MonainDiceLoss
from monai.losses.focal_loss import FocalLoss
from monai.losses.dice import DiceFocalLoss
import torch.nn.functional as F
from readConfig import DEVICE, GAMMA, WEIGHTS, ALPHA, BETA
import matplotlib.pyplot as plt # Remove import!

def getLossFunction(typeLossFunction, alpha=1, beta=1):
    """ Available loss functions such as: 
        DiceLoss,  BCEDiceLoss, BCE, CEDiceLoss, CrossEntrophy, MonaiDiceLoss, FocalLoss, DiceFolcalLoss
    """
    if typeLossFunction == "DiceLoss":
        return DiceLoss()
    elif typeLossFunction == "BCE":
        if len(WEIGHTS) > 0:
            return  nn.BCELoss(weight=torch.FloatTensor(WEIGHTS).to(DEVICE))        
        return  nn.BCELoss()
    elif typeLossFunction == "BCEDiceLoss":
        return  BCEDiceLoss(alpha=alpha, beta=beta)
    elif typeLossFunction == "CEDiceLoss":        
        return  CEDiceLoss()
    elif typeLossFunction == "FocalLoss":        
        return FocalLoss(gamma=GAMMA)
    elif typeLossFunction == "DiceFolcalLoss":        
        return DiceFocalLoss(gamma=GAMMA)
    elif typeLossFunction == "FocalTverskyLoss":
        return FocalTverskyLoss()

   

class DiceLoss(nn.Module):
    """ Dice Loss: 1 - dice Score
        (Dice Score => 2 * AnB / AuB)
    """
    def __init__(self, weight=None, size_average=True):
        super(DiceLoss, self).__init__()

    def forward(self, inputs, targets, smooth=1):          
        dice = diceScore(inputs, targets, noGrad=False)        
        result = 1 - dice        
        return result

class BCEDiceLoss(nn.Module):
    """ BCE + Dice"""
    def __init__(self, alpha=1, beta=1):
        # alpha and beta represent the contribution of each individual function for the total calculation
        self.alpha = alpha
        self.beta = beta
        super(BCEDiceLoss, self).__init__()

    def forward(self, inputs, targets, smooth=1e-9):        
        dice = diceScore(inputs, targets, noGrad=False)
        dice_loss = 1 - dice
        BCE = nn.BCELoss()
        value_bce = BCE(inputs, targets)                
        Dice_BCE =  (self.alpha*value_bce) + (self.beta*dice_loss)

        return Dice_BCE



class EnhancedMixingLoss(nn.Module):
    def __init__(self, weight=None, size_average=True):
        super(EnhancedMixingLoss, self).__init__()

    def forward(self, inputs, targets, smooth=1e-9):
        dice = diceScore(inputs, targets, noGrad=False)
        dice_loss = 1 - dice               
        BCE = nn.BCELoss()
        value_bce = BCE(inputs, targets)                
        Dice_BCE = value_bce + dice_loss        

        return Dice_BCE

class CEDiceLoss(nn.Module):
    """ Dice + Cross Entropy"""
    def __init__(self, weight=None, size_average=True):
        super(CEDiceLoss, self).__init__()

    def forward(self, inputs, targets, smooth=1e-9):
        #inputs = torch.sigmoid(inputs)       
        dice = diceScore(inputs, targets, noGrad=False)        
        dice_loss = 1 - dice        
        CEL = nn.CrossEntropyLoss()     
        value_bce = CEL(inputs, targets)                
        Dice_BCE = value_bce + dice_loss        
        return Dice_BCE


class FocalTverskyLoss(nn.Module):
    """ Focal Tversky Loss"""
    def __init__(self, weight=None, size_average=True):
        super(FocalTverskyLoss, self).__init__()
    
    def Tversky(self, inputs, targets, smooth=1, alpha=0.7):
        y_true_pos = torch.flatten(targets)
        y_pred_pos = torch.flatten(inputs)
        true_pos = torch.sum(y_true_pos * y_pred_pos)
        false_neg = torch.sum(y_true_pos * (1 - y_pred_pos))
        false_pos = torch.sum((1 - y_true_pos) * y_pred_pos)
        return (true_pos + smooth) / (true_pos + alpha * false_neg + (1 - alpha) * false_pos + smooth)


    def forward(self, inputs, targets, gamma=0.75):
        tv = self.Tversky(inputs, targets)
        return torch.pow((1-tv), gamma)
