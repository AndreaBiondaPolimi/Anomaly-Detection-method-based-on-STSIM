from ModelCreator import Model
from DataLoader import load_patches, load_patches_from_file, show_patches, load_patches_from_file_fixed
import cv2
import numpy as np
import matplotlib.pyplot as plt

train_patch_size = 48
valid_patch_size = 48
stride = 8

height=3
orientations=4


if __name__ == "__main__":

    train_patches = load_patches('Dataset\\SEM_Data\\Normal', patch_size=train_patch_size, random=True, n_patches=100)

    #show_patches(train_patches) 

    #valid_patches = load_patches_from_file_fixed('Dataset\\SEM_Data\\Anomalous\\images\\ITIA1108.tif', patch_size=valid_patch_size, 
        #positions = ((249,296),(341,427),(256,171),(122,865),(11,468),(131,365),(328,848))) 

    valid_patches = load_patches_from_file_fixed('Dataset\\SEM_Data\\Anomalous\\images\\ITIA1107.tif', patch_size=valid_patch_size, 
        positions = ((219,261),(249,296),(341,427),(256,171),(122,865),(600,576),(420,900),(215,913))) 


    #model_train = Model('stsim', 'mahalanobis', height=height, orientations=orientations)
    #model_train = Model('stsim', 'stsim', height=height, orientations=orientations)
    model_train = Model('stsim', 'loglikelihood', height=height, orientations=orientations)
    #model_train = Model('stsim', 'kde', height=height, orientations=orientations)
    
    model_train.model_create(train_patches)
    train = model_train.database


    model = Model('stsim', None, height=height, orientations=orientations)
    model.model_create(valid_patches)
    valid = model.database

    for i in range (len(valid_patches)):
        dist = model_train.detector.calculate_distance(valid[i])

        print (dist)

        plt.imshow(valid_patches[i])
        plt.show()
    


