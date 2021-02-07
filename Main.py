from ModelCreator import Model
from DataLoader import load_patches, load_patches_from_file, show_patches, load_gt_from_file
import cv2
import numpy as np
import matplotlib.pyplot as plt
from scipy import integrate 

patch_size = 22
stride = 1

scales = 3
orients = 4

cut_size = (0,672,0,1024)
step=0.005

def validation (n_img, model, ovr_threshold):
    print(n_img)
    valid_patches, valid_img = load_patches_from_file('Dataset\\SEM_Data\\Anomalous\\images\\ITIA11' + n_img + '.tif', patch_size=patch_size, 
        random=False, stride=stride, cut_size=cut_size) 
    valid_gt = load_gt_from_file ('Dataset\\SEM_Data\\Anomalous\\gt\\ITIA11' + n_img + '_gt.png', cut_size=cut_size)
    valid_gt[valid_gt > 0] = 1

    iou, tprs, fprs, ovr = model.model_evaluate(valid_patches, (672,1024), stride, patch_size, valid_gt, valid_img, ovr_threshold, step)

    return iou, tprs, fprs, ovr


def validation_complete(model):
    print ("START VALIDATION PHASE")
    
    valid_fprs = []
    for i in [8,15,27,31,35]:
        iou, tpr, fpr, ovr = validation(str(i).zfill(2), model, None)
        fpr[fpr > 0.05] = 0
        valid_fprs.append (np.argmax(fpr) - 1)

    ovr_threshold = (np.mean(valid_fprs) * step) + 0.1
    print ("OVR Threshold:", ovr_threshold)

    print()

    print ("START TEST PHASE")    
    tprs = []; fprs = []; ious = []; ovrs = []
    for i in range (1,41):
        if (i not in [8,15,27,31,35]):
            iou, tpr, fpr, ovr = validation(str(i).zfill(2), model, ovr_threshold)
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



if __name__ == "__main__":
    #n_patches = 100 maha / 500 kde
    train_patches = load_patches('Dataset\\SEM_Data\\Normal', patch_size=patch_size, random=True, n_patches=500, preprocess_limit=100)
    print ("START TRAINING PHASE")
    print ("Training patches:", len(train_patches))

    model = Model('stsim', 'kde', height=scales, orientations=orients)
    model.model_create(train_patches)

    #validation("11", model, 0.95)
    validation_complete(model)



