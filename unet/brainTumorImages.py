import os
from PIL import Image
import shutil
import os
import random
import logging
from datetime import date

class BrainTumorImages():
    """ Generates JPG images and split them into training and test set """

    def __init__(self, path="kaggle_3m") -> None:
        self.path = path
        self.test_set = None
        self.train_set = None
        self.validation_set = None
        self.logFile = f'logs/images-{date.today()}.log'
        logging.basicConfig(filename=self.logFile, level=logging.INFO,format='%(asctime)s:%(levelname)s:%(message)s')

    def _generateJPG(self):
        """ Generates JPG from TIF images in the current folder """
        logging.info("Generating JPG images from TIF")        
        for dirpath, dirnames, filenames in os.walk(self.path, topdown=False):
            for name in filenames:                
                if os.path.splitext(os.path.join(dirpath, name))[1].lower() == ".tif":
                    outfile = os.path.splitext(os.path.join(dirpath, name))[0] + ".jpg"
                    if os.path.isfile(outfile):
                        logging.info("A jpeg file already exists for %s" % name)                        
                    # If a jpeg is *NOT* present, create one from the tiff.
                    else:
                        outfile = os.path.splitext(os.path.join(dirpath, name))[0] + ".jpg"
                        try:
                            with Image.open(os.path.join(dirpath, name)) as im:
                                logging.info("Generating jpeg for %s" % name)                              
                                im.thumbnail(im.size)
                                im.save(outfile, "JPEG", quality=100)                          
                        except Exception as e:
                            print(e)
                            logging.error(e)
               

    def _moveImagesToFolder(self, dirs, name):
        """ Move the generated images to the corresponding folders for training and validation """
        logging.info("Moving .tif images into train and test folders")        
        for dir in dirs:
            for file in os.listdir(dir):      
                if ".tif" in file:                
                    file = f"{dir}/{file}"
                    if "mask" in file:                        
                        shutil.move(file, f"{self.path}/{name}_masks/")
                    else:                        
                        shutil.move(file, f"{self.path}/{name}_images/")


    def _shuffleImages(self):
        """ Randomnly shuffle the patient's folders to split sets into training and validation by keeping the sequences' order"""
        patients_list = []
        # Read the patients' folders
        logging.info("[INFO] Reading images") 
        for dirpath, _, _ in os.walk(self.path, topdown=False):
            if "TCGA" in dirpath:                
                patients_list.append(f"{dirpath}")
        
        # Randomly split the sample into train and test
        logging.info("Shuffling images and having train and test sets")        
        #random.seed(200)
        random.shuffle(patients_list)                
        n = int(len(patients_list) * .6)
        logging.info(f"Total length: {len(patients_list)} - N: {n} Proportion: (60% - 20% - 20%)")
        self.train_set = patients_list[:n]
        sub_n = int(len(patients_list[n:]) * .5)
        self.validation_set = patients_list[n:n+sub_n]
        self.test_set = patients_list[n+sub_n:]
        logging.info(f"Results for training: {len(self.train_set)} \n Results for validation: {len(self.validation_set)}  \n Results for testing: {len(self.test_set)} ")        
        print(f"Results for training: {len(self.train_set)} \n Results for validation: {len(self.validation_set)}  \n Results for testing: {len(self.test_set)} ")        
    
    def loadImagesForTrainValTest(self):
        """ Executes the whole sequence of steps to get the images for trainin and validation. """
        #self._generateJPG()
        self._shuffleImages()
        self._moveImagesToFolder(self.train_set, "train")
        self._moveImagesToFolder(self.validation_set, "val")
        self._moveImagesToFolder(self.test_set, "test")
        
# ========================
# i = BrainTumorImages("kaggle_3m")
# i.loadImagesForTrainAndVal()
