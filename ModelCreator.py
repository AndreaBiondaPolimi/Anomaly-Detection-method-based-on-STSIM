import cv2
import copy
import numpy as np
import matplotlib.pyplot as plt
import scipy as sp
from multiprocessing import Pool
from scipy import integrate 

from STSIM.Numpy.metrics import Metric as Metric_NP
from STSIM.Tensorflow.metrics_TF import Metric as Metric_TF
from DataLoader import load_patches
from OutlierDetection.MahalanobisDetector import MahalanobisDetector
from OutlierDetection.StsimDetector import StsimDetector
from OutlierDetection.LogLikelihoodDetector import LogLikelihoodDetector
from OutlierDetection.KdeDetector import KdeDetector
from OutlierDetection.IForDetector import IForDetector
from DataLoader import check_preprocessing
from Utils import visualize_results, get_performance, get_roc, get_ovr_iou

import warnings
warnings.filterwarnings('error')
#Best H=3,O=4

class Model():
    def __init__(self, indictator, distance, height=3, orientations=4, mode='tf'):
        self.Metric = Metric_TF if mode == 'tf' else Metric_NP 
        self.mode = mode
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



    def get_distance_density_from_model (self, valid_patches, density_shape, stride, patch_size):
        model_valid = Model('stsim', None, self.height, self.orientations)
        model_valid.model_create(valid_patches)
        valid_db = model_valid.database
        valid_flags = model_valid.database_flags

        valid_db = self.detector.calculate_distance(np.array(valid_db))

        density = self.image_reconstruction(valid_db, valid_flags, density_shape, patch_size, stride)

        return density


    ### Performance Evaluation ###
    def model_evaluate (self, valid_patches, density_shape, stride, patch_size, valid_gt, valid_img, ovr_threshold, step):
        tprs = []; fprs = []#; ious = []; ovrs = []    
        model_density = self.get_distance_density_from_model(valid_patches, density_shape, stride, patch_size)

        for alpha in np.arange (0.1, 1.0, step):
            self.detector.calculate_acceptances(alpha)
            _ , density_tresholded = self.detector.get_density_tresholded(np.copy(model_density), False)
            tpr, fpr = get_roc(valid_gt, density_tresholded)
            tprs.append (tpr); fprs.append (fpr)
        
        #Calculate evaluations
        tprs = np.array(tprs); fprs = np.array(fprs) 

        ovr = None; iou=None
        if (ovr_threshold is not None):
            self.detector.calculate_acceptances(ovr_threshold)
            density, density_tresholded = self.detector.get_density_tresholded(np.copy(model_density), False)
            _, fpr = get_roc (valid_gt, density_tresholded)
            ovr, iou = get_ovr_iou(valid_gt, density_tresholded)

            print ("FPR:", fpr)
            print ("OVR", ovr)
            print ("IoU:", iou)
            print ("Auc: ", (-1 * integrate.trapz(np.array(tprs), np.array(fprs))))

            #Print residual & score map
            #visualize_results(valid_img/255, density, "Residual Map")
            #visualize_results(valid_gt, density_tresholded, "Score Map")

        return iou, tprs, fprs, ovr


    def image_reconstruction (self, y_valid, valid_flags, density_shape, patch_size, stride):
        _,w = density_shape
        reconstrunction = np.zeros((density_shape))
        normalizator = np.zeros((density_shape))
        

        i=0; j=0
        for idx in range (len(y_valid)):
            if (valid_flags[idx]):
                reconstrunction [j:j+patch_size, i:i+patch_size] += np.full ((patch_size,patch_size),y_valid[idx] )
            else:
                reconstrunction [j:j+patch_size, i:i+patch_size] += np.full ((1,patch_size),self.detector.normal_score ) 
            normalizator [j:j+patch_size, i:i+patch_size] += np.ones((patch_size,patch_size))
            if (i+patch_size < w):
                i=i+stride
            else:
                i=0; j=j+stride
        reconstrunction =  reconstrunction/normalizator

        return reconstrunction


    ### Database Creation ###
    def create_stsim_db(self, patches):
        m = self.Metric()
        flags = [None] * len(patches)
        
        if (self.mode == 'tf'):
            database = None

            max_batch_size = 10000 #Does not fit whole in memory
            if (len(patches) > max_batch_size): 
                for i in range(int(len(patches) / max_batch_size) + 1):
                    batch_patch = patches[i*max_batch_size : min(((i+1) * max_batch_size), len(patches)) ]
                    if database is None:
                        database = m.STSIM_M(batch_patch, self.height, self.orientations)
                    else:
                        database = np.concatenate((database, m.STSIM_M(batch_patch, self.height, self.orientations)))
            else:
                database = m.STSIM_M(patches, self.height, self.orientations)
                    
            #Damn flags
            for i in range(len(patches)):
                flags[i] = check_preprocessing(patches[i])

        else:    
            database = [None]  * len(patches)

            args = [{'model': copy.deepcopy(m), 'patch_idx': idx, 'patch_value': copy.deepcopy(patches[idx]),
                        'height': self.height, 'orientations': self.orientations} for idx in range(len(patches))] 

            with Pool(processes=5) as pool:
                results = pool.map(extract_features, args, chunksize=1)

            for result in results:
                idx = result['patch_idx']
                database [idx] = result['features']
                flags [idx] = result['flag']
        
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



def extract_features(args):
    model = args['model']
    idx = args['patch_idx']
    patch = args['patch_value']
    height = args['height']
    orientations = args['orientations']

    features = model.STSIM_M(patch, height, orientations)
    flag = check_preprocessing(patch)

    return {'patch_idx':idx, 'features': features, 'flag': flag}