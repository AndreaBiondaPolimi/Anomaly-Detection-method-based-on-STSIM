from ModelCreator import Model
from DataLoader import load_patches, load_patches_from_file, show_patches, load_gt_from_file
import cv2
import numpy as np
import matplotlib.pyplot as plt

train_patch_size = 32
valid_patch_size = 32
stride = 2


if __name__ == "__main__":

    train_patches = load_patches('Dataset\\SEM_Data\\Normal', patch_size=train_patch_size, random=True, n_patches=500)

    valid_patches, valid_img = load_patches_from_file('Dataset\\SEM_Data\\Anomalous\\images\\ITIA1106.tif', patch_size=valid_patch_size, 
        random=False, stride=stride, cut_size=(0,672,0,1024)) 
    valid_gt = load_gt_from_file ('Dataset\\SEM_Data\\Anomalous\\gt\\ITIA1106_gt.png', (0,672,0,1024))


    #model = Model('stsim', 'mahalanobis', height=3, orientations=4)
    #model = Model('stsim', 'stsim')
    #model = Model('stsim', 'loglikelihood')
    model = Model('stsim', 'kde', height=3, orientations=4)
    #model = Model('stsim', 'ifor', height=3, orientations=4)
    
    model.model_create(train_patches)

    model.model_evaluate(valid_patches, (672,1024), stride, valid_patch_size, valid_gt)

    model.model_visualize (valid_patches, (672,1024), stride, valid_patch_size, valid_gt, valid_img, alpha=0.965)

