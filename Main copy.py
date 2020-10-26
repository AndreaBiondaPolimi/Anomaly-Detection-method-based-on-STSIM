from ModelCreator import Model
from DataLoader import load_patches, load_patches_from_file, show_patches, load_patches_from_file_fixed
import cv2
import numpy as np
import matplotlib.pyplot as plt

train_patch_size = 32

valid_patch_size = 32
stride = 8


if __name__ == "__main__":

    train_patches = load_patches('Dataset\\SEM_Data\\Normal', patch_size=train_patch_size, random=True, n_patches=10)

    #show_patches(train_patches) 

    valid_patches = load_patches_from_file_fixed('Dataset\\SEM_Data\\Anomalous\\images\\ITIA1108.tif', patch_size=valid_patch_size, 
        positions = ((249,296),(341,427),(256,171),(122,865),(11,468),(131,365),(328,848))) 

    model_train = Model('stsim', 'stsim-m')
    #model_train = Model('stsim', 'stsim-i')
    model_train.model_create(train_patches)
    train = model_train.database

    model = Model('stsim', None)
    model.model_create(valid_patches)
    valid = model.database

    for i in range (len(valid_patches)):
        dist = model_train.calculate_distance(valid[i], 50)

        print (dist)

        plt.imshow(valid_patches[i])
        plt.show()
    


