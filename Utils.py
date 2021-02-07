import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
import cv2
import albumentations as A

def visualize_results (img, res, txt):
    f = plt.figure(figsize=(17, 7))

    f.add_subplot(1,2, 1)
    plt.xticks([])
    plt.yticks([])
    plt.title("Image")
    plt.imshow(img)

    f.add_subplot(1,2, 2)
    plt.xticks([])
    plt.yticks([])
    plt.title("Reconstructed")
    plt.imshow(res)

    f.text(.5, .05, txt, ha='center')
    plt.show()


### Performance Evaluations ###
def get_performance (y_true, y_pred):
    y_true = np.array (y_true, dtype=int)
    y_pred = np.array (y_pred, dtype=int)

    iou = iou_coef(y_true, y_pred)
    tpr, fpr = roc_coef (y_true, y_pred)
    ovr = ovr_coef (y_true, y_pred) if fpr <= 0.05 else None
    return iou, tpr, fpr, ovr

def get_roc (y_true, y_pred):
    y_true = np.array (y_true, dtype=int)
    y_pred = np.array (y_pred, dtype=int)

    tpr, fpr = roc_coef (y_true, y_pred)
    return tpr, fpr

def get_ovr_iou (y_true, y_pred):
    y_true = np.array (y_true, dtype=int)
    y_pred = np.array (y_pred, dtype=int)

    iou = iou_coef(y_true, y_pred)
    ovr = ovr_coef (y_true, y_pred)
    return ovr, iou

def iou_coef(y_true, y_pred):
    intersection = np.logical_and(y_true,y_pred) # Logical AND
    union = np.logical_or(y_true,y_pred)    # Logical OR
    IOU = float(np.sum(intersection)/np.sum(union))
    return IOU

def roc_coef (y_true, y_pred):
    tp = np.sum (y_true*y_pred)
    fn = np.sum ((y_true - y_pred).clip(min=0))
    tpr = tp / (tp + fn)

    fp = np.sum ((y_pred - y_true).clip(min=0))
    tn = np.sum ((1-y_true)*(1-y_pred))
    fpr = fp / (fp + tn)

    if (tn < 0):
        visualize_results (y_true, y_pred, "aa")

    return tpr, fpr

def ovr_coef (y_true, y_pred):
    r,l = cv2.connectedComponents(np.array(y_true*255, np.uint8))
    blobs = [np.argwhere(l==i) for i in range(1,r)]
    ovrs = []

    for blob in blobs:
        ground = np.zeros_like(y_true, dtype=int)
        for b in blob:
            ground[b[0],b[1]] = 1

        #if (np.sum(ground) > 130):
        tp = np.sum (ground*y_pred)
        ovr = tp / np.sum(ground)
        ovrs.append (ovr)

    return ovrs    


