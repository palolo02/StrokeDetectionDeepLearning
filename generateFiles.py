
from utils.file import DataProcessor




if __name__=="__main__":
    d = DataProcessor()

    # # # generate images
    #folders = ["dataset/train/images/", "dataset/train/masks/", "dataset/validation/images/" , "dataset/validation/masks/" ]
    #folders = ["dataset/test/masks/"]#, "dataset/test/images/", ]
    #[d.readPatientsFileAndSaveTIFF(folder) for folder in folders]

    # # # Adding images and masks
    # #[d.addingPaddingOfImages(main_folder=folder, number=32) for folder in folders]

    # # Renaming images and masks
    #sub_folders = ["dataset/train/","dataset/validation/"]
    sub_folders = ["dataset/test/"]    
    [d.renameImagesAndMasks(folder) for folder in sub_folders]


    #d.readPatientsTIFFFile("dataset/train/images/sub-r001s001_ses-1_T1w/")