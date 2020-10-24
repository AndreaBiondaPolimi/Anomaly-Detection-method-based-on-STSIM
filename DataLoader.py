
import cv2
import matplotlib.pyplot as plt
import os
import numpy as np

def load_patches_from_file (file, patch_size, random, n_patches=3, stride=32, cut_size=None):
    im1 = cv2.imread(file, cv2.IMREAD_GRAYSCALE)

    if (cut_size is not None):
        im1 = im1[cut_size[0]:cut_size[1], cut_size[2]:cut_size[3]]

    cropped = []
    if (random == True):
        for _ in range (n_patches):
            j = np.random.randint(0, im1.shape[0] - patch_size)
            i = np.random.randint(0, im1.shape[1] - patch_size)
            cropped.append(im1[j:j+patch_size, i:i+patch_size])
    else:
        for j in range (int((im1.shape[0] - patch_size) / stride) + 1):
            for i in range (int((im1.shape[1] - patch_size) / stride) + 1):
                #print (i,j)
                cropped.append(im1[(j*stride):(j*stride)+patch_size, (i*stride):(i*stride)+patch_size])


    return cropped


def load_patches (folder, patch_size, random=True, n_patches=3, stride=32, cut_size=None):
    patches = []

    for file in os.listdir(folder):
        if file.endswith(".bmp") or file.endswith(".tif"):
            ret = load_patches_from_file(os.path.join(folder, file), patch_size, random, n_patches, stride, cut_size)
            for r in ret:
                patches.append(r)

    return patches


if __name__ == "__main__":
    load_patches('CUReT_Data')