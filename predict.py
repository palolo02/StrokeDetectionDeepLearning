from doctest import TestResults
from dataPipeline.predict import Prediction
from dataPipeline.testing import test
from utils.helper import ConcatLossDiceIoULR
from tests_.unitTests import TestDisplayKernels, TestResults as TR
#from dataProcessing.temporaryData import GenerateNiftiGroups


def sampledata():
    # Load images of one patient and show 10
    # Below show the images of the mask segmentation
    from torchvision.transforms import Resize, ToTensor
    import torchvision
    from PIL import Image
    import os
    import torch

    patient = "dataset/train/images/sub-r001s001_ses/"
    mask = "dataset/train/masks/sub-r001s001_ses_mask/"


    def getSequenceOfData(folder, a=80, b=100, size=224):
        """ Returns certain sequence of data from a set of png images in a folder"""
        sequence = 0
        images = []
        for img in os.listdir(folder):
            if ".png" in img:
                sequence += 1
                if sequence >= a and sequence <= b:
                    with Image.open(f"{folder}/{img}").convert("L") as temp:
                        temp = Resize(size=(size,size))(temp)
                        temp = ToTensor()(temp)
                        temp = torch.unsqueeze(temp, dim=0)
                        #temp = torch.unsqueeze(temp, dim=0)
                        images.append(temp)
        return  torch.concat(images,dim=0)

    rows_1 = getSequenceOfData(patient)
    rows_2 = getSequenceOfData(mask)

    result = torch.cat((rows_1, rows_2), dim=0)
    grid = torchvision.utils.make_grid(result, nrow=21, padding=100)
    torchvision.utils.save_image(grid, f'dataProcessing/Stroke/sample.png')                                


def runTesting():
    pass
    #test()

def predict():
    # ================================================
    # # Generate groups of nifti files
    # folders = ["dataset/validation/images/","dataset/validation/masks/", "dataset/train/images/","dataset/train/masks/"]    
    # g = GenerateNiftiGroups(32)
    # for folder in folders:
    #     g.loadFilesFromFolder(folder)
    # ================================================
    
    # 1. Load the model
    p = Prediction()

    # 2. Load the data (data loader)
    
    # ============= Prediction ==============
    # Path where the image is located
    #img_file = "kaggle_3m/train_images/TCGA_CS_4941_19960909_11.tif"
    
    patients = [
        # {
        #     "image" : "dataset/test/images/sub-r009s020_ses-1_T1w.nii.gz",
        #     "mask"  : "dataset/test/masks/sub-r009s020_ses-1_space-orig_label-L_desc-T1lesion_mask.nii.gz"
        # },
        {
            "image" : "dataset/test/images/sub-r009s021_ses-1_T1w.nii.gz",
            "mask"  : "dataset/test/masks/sub-r009s021_ses-1_space-orig_label-L_desc-T1lesion_mask.nii.gz"
        },
        {
            "image" : "dataset/test/images/sub-r009s022_ses-1_T1w.nii.gz",
            "mask"  : "dataset/test/masks/sub-r009s022_ses-1_space-orig_label-L_desc-T1lesion_mask.nii.gz"
        },
        {
            "image" : "dataset/test/images/sub-r009s024_ses-1_T1w.nii.gz",
            "mask"  : "dataset/test/masks/sub-r009s024_ses-1_space-orig_label-L_desc-T1lesion_mask.nii.gz"
        },
        {
            "image" : "dataset/test/images/sub-r009s025_ses-1_T1w.nii.gz",
            "mask"  : "dataset/test/masks/sub-r009s025_ses-1_space-orig_label-L_desc-T1lesion_mask.nii.gz"
        }
    ]

    for patient in patients:
        p.predictPNGSequenceOfImages(patient["image"],patient["mask"])
        print("Predictions generated")

    # patient = "dataset/test/images/sub-r009s001_ses-1_T1w.nii.gz"
    # patient = "dataset/test/images/sub-r009s002_ses-1_T1w.nii.gz"
    # patient = "dataset/test/images/sub-r009s003_ses-1_T1w.nii.gz"
    
    # ground_truth = "dataset/test/masks/sub-r009s001_ses-1_space-orig_label-L_desc-T1lesion_mask.nii.gz"
    # ground_truth = "dataset/test/masks/sub-r009s002_ses-1_space-orig_label-L_desc-T1lesion_mask.nii.gz"
    # ground_truth = "dataset/test/masks/sub-r009s003_ses-1_space-orig_label-L_desc-T1lesion_mask.nii.gz"
    
    # p.predictSequenceOfImages(patient,ground_truth)


if __name__=="__main__":
    #sampledata()
    #runTesting()
    # Experiment => 3LayersUnet_BCEDiceLoss_LR-2_SGD_WD-3_50Epochs_20K
    # 3LayersUnet_BCEDiceLoss_LR-2_SGD_WD-3_50Epochs_20K
    #predict()
    #ConcatLossDiceIoULR()

    # =============== Compare models ==============
    t = TR()    
    # #t.test_single_predict()
    # #t.test_predict_loader()
    # #t.test_predict_loader_metrics()
    t.test_predict_same_image()
    #t.test_report_results()


    # =============== Display Kernels ==============
    #t = TestDisplayKernels()
    #t.test_show_Activation_Maps()
    #t.test_show_kernels()
    

    


    
