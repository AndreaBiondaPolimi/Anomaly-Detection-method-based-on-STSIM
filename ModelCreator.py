from STSIM.metrics import Metric
from DataLoader import load_patches
import cv2
import numpy as np
import matplotlib.pyplot as plt
import scipy as sp
from OutlierDetection.MahalanobisDetector import MahalanobisDetector
from OutlierDetection.StsimDetector import StsimDetector
from OutlierDetection.LogLikelihoodDetector import LogLikelihoodDetector
from OutlierDetection.KdeDetector import KdeDetector
from OutlierDetection.IForDetector import IForDetector
from DataLoader import check_preprocessing
from scipy import integrate 

#Best H=3,O=4

class Model():
    def __init__(self, indictator, distance, height=3, orientations=4):
        self.indicator = indictator
        self.distance = distance
        self.height = height
        self.orientations = orientations

        self.database = None
        self.database_flags = None

        self.detector = None


    def model_create (self, patches, alpha = 0.95):
        if (self.indicator == 'stsim'):
            self.database, self.database_flags = self.create_stsim_db(patches)
        else:
            raise "Invalid indicator"
            
        self.detector = self.assign_detector(self.distance, self.database)
        if (self.detector is not None):
            self.detector.calculate_statistics()
            self.detector.calculate_acceptances(alpha)

        print (self.database.shape)



    def get_distance_density_from_model (self, valid_patches, density_shape, stride, patch_size, tresholded=True):
        h,w = density_shape
        density = np.zeros(shape=(density_shape))
        normalizator = np.zeros(shape=(density_shape))

        model_valid = Model('stsim', None, self.height, self.orientations)
        model_valid.model_create(valid_patches)
        valid_db = model_valid.database
        valid_flags = model_valid.database_flags


        for j in range (int((h - patch_size) / stride) + 1):
            for i in range (int((w - patch_size) / stride) + 1):
                if (valid_flags[j*int((w - patch_size) / stride + 1) + i]):
                    f_valid = valid_db [j*int((w - patch_size) / stride + 1) + i] 
                    dens_val = self.detector.calculate_distance(f_valid)
                    density[(j*stride):(j*stride)+patch_size, (i*stride):(i*stride)+patch_size] += dens_val
                else:
                    density[(j*stride):(j*stride)+patch_size, (i*stride):(i*stride)+patch_size] += self.detector.normal_score
                normalizator[(j*stride):(j*stride)+patch_size, (i*stride):(i*stride)+patch_size] += 1

        density = density / normalizator
        if (tresholded):
            density = self.detector.get_density_tresholded(density, False)

        return density


    ### Performance Evaluation ###
    def model_evaluate (self, valid_patches, density_shape, stride, patch_size, valid_gt):
        tprs = []
        fprs = []
        model_density = self.get_distance_density_from_model(valid_patches, density_shape, stride, patch_size, False)

        for alpha in np.arange (0.1, 1.0, 0.005):
            self.detector.calculate_acceptances(alpha)
            density = self.detector.get_density_tresholded(np.copy(model_density), False)
            _ , tpr, fpr = self.get_performance (valid_gt, density)
            tprs.append (tpr)
            fprs.append (fpr)
        tprs.append(0); fprs.append(0)
        
        print ("auc: " + str(-1 * integrate.trapz(np.array(tprs), np.array(fprs))))
        plt.xlabel('FPR')
        plt.ylabel('TPR')
        plt.plot (np.array(fprs), np.array(tprs))
        plt.show()

    def get_performance (self, y_true, y_pred):
        y_true = np.array (y_true, dtype=int)
        y_pred = np.array (y_pred, dtype=int)

        iou = self.iou_coef(y_true, y_pred)
        tpr, fpr = self.roc_coef (y_true, y_pred)
        return iou, tpr, fpr

    def iou_coef(self, y_true, y_pred):
        intersection = np.logical_and(y_true,y_pred) # Logical AND
        union = np.logical_or(y_true,y_pred)    # Logical OR
        IOU = float(np.sum(intersection)/np.sum(union))
        return IOU

    def roc_coef (self, y_true, y_pred):
        tp = np.sum (y_true*y_pred)
        fn = np.sum ((y_true - y_pred).clip(min=0))
        tpr = tp / (tp + fn)

        fp = np.sum ((y_pred - y_true).clip(min=0))
        tn = np.sum ((1-y_true)*(1-y_pred))
        fpr = fp / (fp + tn)

        return tpr, fpr





    ### Database Creation ###
    def create_stsim_db(self, patches):
        m = Metric()
        flags = []
        database = []

        #creation of the feature vectores
        for i in range(0,len(patches)):
            database.append(m.STSIM_M(patches[i], self.height, self.orientations))
            flags.append(check_preprocessing(patches[i]))
        
        return np.array(database), np.array(flags)

    ### Detector Assignment ###
    def assign_detector (self, detector_type, database):
        if (detector_type == 'stsim'):
            return StsimDetector(database)
        elif (detector_type == 'loglikelihood'):
            return LogLikelihoodDetector(database)
        elif (detector_type == 'mahalanobis'):
            return MahalanobisDetector (database)
        elif (detector_type == 'kde'):
            return KdeDetector (database)
        elif (detector_type == 'ifor'):
            return IForDetector (database)

        return None



    ### Visualization ###
    def visualize_subbands (self, patches):
        m = Metric()
        for i in range(0,len(patches)):
            m.visualize(patches[i], self.height, self.orientations)

    def model_visualize (self, valid_patches, density_shape, stride, patch_size, valid_gt, valid_img, alpha):
        model_density = self.get_distance_density_from_model(valid_patches, density_shape, stride, patch_size, False)
        self.detector.calculate_acceptances(alpha)
        density = self.detector.get_density_tresholded(np.copy(model_density), False)
        
        iou , tpr, fpr = self.get_performance (valid_gt, density)
        txt = "iou: " + str(iou) + "       tpr: " + str(tpr) + "       fpr: " + str(fpr)
        
        self.visualize_results (valid_img, density, txt)

    def visualize_results (self,img, res, txt):
        f = plt.figure(figsize=(12, 4))

        f.add_subplot(1,2, 1)
        plt.xticks([])
        plt.yticks([])
        plt.title("Image")
        plt.imshow(img)

        """
        f.add_subplot(1,3, 2)
        plt.xticks([])
        plt.yticks([])
        plt.title("Ground truth")
        plt.imshow(gt)
        """

        f.add_subplot(1,2, 2)
        plt.xticks([])
        plt.yticks([])
        plt.title("Density")
        plt.imshow(res)

        f.text(.5, .05, txt, ha='center')
        plt.show()
