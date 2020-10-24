from ModelCreator import Model
from DataLoader import load_patches
import cv2
import numpy as np
import matplotlib.pyplot as plt

def calculate_stsim_m (database, index):
    f1 = database[index]

    distances = []
    for i in range(database.shape[0]):
        f2 = database[i]

        dists = (f1 - f2) ** 2
        vars = []
        for i in range(database.shape[1]):
            vars.append(np.var(database[:,i]))
        
        distances.append (np.sum(dists/vars))
    
    return distances

def get_p_at_1 (distances, index, ref):
    distances = np.array(distances)
    distances[index] = np.max(distances)

    idx_min = np.argmin(distances)
    print ('index:',index,'idx_min:',idx_min, 'ref:',ref)

    if (idx_min >= ref and idx_min <= ref+2):
        print('1')
        return 1
    else:
        print('0')
        return 0



if __name__ == "__main__":

    patches = load_patches('Dataset\\CUReT_Data', patch_size=128, n_patches=3, random=True, cut_size=(140,340,200,500))

    model = Model('stsim')
    model.model_create(patches)
    
    database = model.database

    corrects = 0
    for index in range (database.shape[0]):
        distances = calculate_stsim_m(database , index)
        
        corrects += get_p_at_1 (distances, index, int(index/3) * 3)

    print(corrects/database.shape[0])


