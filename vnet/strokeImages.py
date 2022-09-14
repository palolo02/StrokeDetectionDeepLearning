import os
from turtle import clear
from PIL import Image
import shutil
import os
import random
import logging
from datetime import date
import shutil

class StrokeImages():
    """ Move the images into the different sets for the model """

    def __init__(self, path="dataset/data/"):
        self.path = path
        self.test_set = None
        self.train_set = None
        self.validation_set = None        
        self.logFile = f'logs/images-{date.today()}.log'
        logging.basicConfig(filename=self.logFile, level=logging.INFO,format='%(asctime)s:%(levelname)s:%(message)s')
       

    def _moveImagesToFolder(self, datafolder, name):
        """ Move the generated images to the corresponding folders for training and validation """
        logging.info("Moving .nii.gz images into train and test folders")
        print(f"Moving .nii.gz images into train and test folders {datafolder} \n {name}")
        for path in datafolder:
            #print(path)
            for dirpath, dirnames, filenames in os.walk(path):
                for file in filenames:
                    #print(f"Reading: {dirpath}/{file}")
                    if ".nii.gz" in file:                                  
                        if "mask" in file:                        
                            shutil.copy(f"{dirpath}/{file}", f"{name}/masks/")
                        else:                        
                            shutil.copy(f"{dirpath}/{file}", f"{name}/images/")
                        #print(f" File {file} moved successfully!")
       

    def _shufflePatients(self):
        """ Randonmly shuffle the patient's folders to split sets into training and validation by keeping the sequences' order"""
        patients_list = []
        # Read the patients' folders
        logging.info("Reading patient's folders...") 
        
        # Shuffling cohorts in study
        for cohort in os.listdir(self.path):
            # Directories starting with R are part of the cohort            
            patients_list.append(f"{self.path}{cohort}")
            #print(f"{self.path}{cohort}")
        
        # Randomly split the sample into train and test
        logging.info("Shuffling cohort of patients to have the three data set we need")        
        #random.seed(100)
        random.shuffle(patients_list)
        print(patients_list)
        # For testing
        # self.train_set = [patients_list[0]]
        # self.validation_set = [patients_list[1]]
        # self.test_set = [patients_list[2]]
        # print(self.train_set)
        # print(self.validation_set)
        # print(self.test_set)
        n = int(len(patients_list) * .6)
        logging.info(f"Total length: {len(patients_list)} - N: {n} Proportion: (60% - 20% - 20%)")        
        print(f"===== Total length: {len(patients_list)} - N: {n} Proportion: (60% - 20% - 20%) =========")
        self.train_set = patients_list[:n]
        sub_n = int(len(patients_list[n:]) * .5)
        self.validation_set = patients_list[n:n+sub_n]
        self.test_set = patients_list[n+sub_n:]
        logging.info(f"Results for training: {len(self.train_set)} \n Results for validation: {len(self.validation_set)}  \n Results for testing: {len(self.test_set)} ")        
        #print(f"Results for training: {len(os.listdir(self.train_set))} \n Results for validation: {len(os.listdir(self.validation_set))}  \n Results for testing: {len(os.listdir(self.test_set))} ")        
    
    def loadImagesForTrainValTest(self):
        """ Executes the whole sequence of steps to get the images for trainin and validation. """        
        self._shufflePatients()
        self._moveImagesToFolder(self.train_set, "dataset/train/")
        self._moveImagesToFolder(self.validation_set, "dataset/validation/")
        self._moveImagesToFolder(self.test_set, "dataset/test/")
    
    def cleanFolders(self):        
        folders = ['dataset/train/images/',
                    'dataset/train/masks/',
                    'dataset/validation/images/',
                    'dataset/validation/masks/',
                    'dataset/test/images/',
                    'dataset/test/masks/'
                ]
        for folder in folders:
            try:
                print("Erasing folder...")
                shutil.rmtree(folder)     
                print("Creating folder...")           
                os.mkdir(folder)
            except OSError as e:
                print("Error: %s : %s" % (folder, e.strerror))

# ========================
# i = StrokeImages()
# #i.cleanFolders()
# i.loadImagesForTrainValTest()
