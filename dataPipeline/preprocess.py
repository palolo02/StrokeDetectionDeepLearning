from temporaryData import GenerateNiftiGroups

if __name__=="__main__":
    # ================================================
    # Generate groups of nifti files
    folders = ["dataset/validation/images/","dataset/validation/masks/", "dataset/train/images/","dataset/train/masks/"]    
    g = GenerateNiftiGroups(slices=64)
    print("generating files")
    for folder in folders:
        g.loadFilesFromFolder(folder)
    
    # Clean data
    # print("cleaning data")
    # for folder in folders:
    #     g.removeFiles(folder)
    #================================================