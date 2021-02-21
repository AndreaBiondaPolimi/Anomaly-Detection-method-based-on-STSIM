from ModelCreator import Model
from DataLoader import load_patches, load_patches_from_file, show_patches, load_gt_from_file
import cv2
import numpy as np
import matplotlib.pyplot as plt
from scipy import integrate 

import argparse
import configparser
import os

patch_size = 18
stride = 1
n_patches = 100
scales = 2
orients = 4
val_fpr = 0.05

cut_size = (0,672,0,1024)
step=0.005


def validation (model):
    print ("START VALIDATION PHASE")
    
    valid_fprs = []
    for i in [8,15,27,31,35]:
        iou, tpr, fpr, ovr = evaluation(str(i).zfill(2), model, None, False)
        fpr[fpr > val_fpr] = 0
        valid_fprs.append (np.argmax(fpr) - 1)

    ovr_threshold = (np.mean(valid_fprs) * step) + 0.1
    print ("OVR Threshold:", ovr_threshold)

    return ovr_threshold

def evaluation (n_img, model, ovr_threshold, to_show):
    print("TEST IMAGE ", n_img)

    valid_patches, valid_img = load_patches_from_file('Dataset\\SEM_Data\\Anomalous\\images\\ITIA11' + n_img + '.tif', patch_size=patch_size, 
        random=False, stride=stride, cut_size=cut_size) 
    valid_gt = load_gt_from_file ('Dataset\\SEM_Data\\Anomalous\\gt\\ITIA11' + n_img + '_gt.png', cut_size=cut_size)
    valid_gt[valid_gt > 0] = 1

    iou, tprs, fprs, ovr = model.model_evaluate(valid_patches, (672,1024), stride, patch_size, valid_gt, valid_img, ovr_threshold, step, to_show)

    return iou, tprs, fprs, ovr


def evaluation_complete(model, ovr_threshold):
    print ("START TEST PHASE")    
    tprs = []; fprs = []; ious = []; ovrs = []
    for i in range (1,41):
        if (i not in [8,15,27,31,35]):
            iou, tpr, fpr, ovr = evaluation(str(i).zfill(2), model, ovr_threshold, False)
            tprs.append(tpr); fprs.append(fpr); ious.append(iou); ovrs.append(ovr)
            print ()
    
    flat_ovr = [item for sublist in ovrs for item in sublist]
    flat_ovr = np.sort(flat_ovr)
    flat_half = flat_ovr[int(-len(flat_ovr)/2):]
    print (flat_half)
    print ("Mean IoU:", np.mean(ious))
    print ("Min OVR:", np.min(flat_half))    
    tprs = np.mean(tprs, axis=0); fprs = np.mean(fprs, axis=0)
    print ("AUC: " + str(-1 * integrate.trapz(tprs, fprs)))
    plt.xlabel('FPR')
    plt.ylabel('TPR')
    plt.plot (fprs, tprs)
    plt.show()


def check_metric(value):
    if value != "mahalanobis" and value != "kde":
        raise argparse.ArgumentTypeError("Invalid anomaly metrics argument")
    return value

def check_config_file(value):
    if os.path.exists(value):
        if value.endswith('.ini'):
            return value
    raise argparse.ArgumentTypeError("Invalid file path")


def parse_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument('-t', action="store", help='Type of anomaly metrics to use, one of: mahalanobis or kde', dest="type", type=check_metric, required=True)
    parser.add_argument('-f', action="store", help='Configuration file path', dest="file", type=check_config_file, required=True)

    args = parser.parse_args()
    return args


if __name__ == "__main__":
    args = parse_arguments()

    config = configparser.ConfigParser()
    config.read(args.file)

    n_patches = int(int(config['TRAINING']['NPatches']) / 5); patch_size = int(config['TRAINING']['PatchSize'])
    stride = int(config['TRAINING']['Stride']); scales = int(config['TRAINING']['Scales'])
    orients = int(config['TRAINING']['Orientations']); eval_type = int(config['EVALUATION']['EvaluationType'])
    val_fpr = float(config['EVALUATION']['ThresholdFPR'])

    #n_patches = 100 maha / 500 kde
    train_patches = load_patches('Dataset\\SEM_Data\\Normal', patch_size=patch_size, random=True, n_patches=n_patches, preprocess_limit=100)
    print ("START TRAINING PHASE")
    print ("Training patches:", len(train_patches))

    model = Model('stsim', args.type, height=scales, orientations=orients)
    model.model_create(train_patches)


    tresh = validation(model)
    if (eval_type == 0):
            evaluation_complete(model, tresh)
    elif (eval_type >  0 and eval_type < 41):
        evaluation(str(eval_type).zfill(2), model, tresh, True)
    else:
        raise argparse.ArgumentTypeError("Evaluation type")



