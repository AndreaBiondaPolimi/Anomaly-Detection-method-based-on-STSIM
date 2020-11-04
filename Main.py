from ModelCreator import Model
from DataLoader import load_patches, load_patches_from_file, show_patches, load_gt_from_file
import cv2
import numpy as np
import matplotlib.pyplot as plt

train_patch_size = 48

valid_patch_size = 48
stride = 8


if __name__ == "__main__":

    train_patches = load_patches('Dataset\\SEM_Data\\Normal', patch_size=train_patch_size, random=True, n_patches=200)

    valid_patches = load_patches_from_file('Dataset\\SEM_Data\\Anomalous\\images\\ITIA1107.tif', patch_size=valid_patch_size, 
        random=False, stride=stride) 
    valid_gt = load_gt_from_file ('Dataset\\SEM_Data\\Anomalous\\gt\\ITIA1107_gt.png', (672,1024))


    #model = Model('stsim', 'mahalanobis')
    #model = Model('stsim', 'stsim')
    #model = Model('stsim', 'loglikelihood')
    model = Model('stsim', 'kde')
    model.model_create(train_patches)


    density = model.get_distance_density_from_model (valid_patches, density_shape=(672,1024), stride=stride, patch_size=valid_patch_size)
    print (model.iou_coef(valid_gt, density))
    plt.imshow(density)
    plt.show()


