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
from DataLoader import check_preprocessing

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


    def model_create (self, patches):
        if (self.indicator == 'stsim'):
            self.database, self.database_flags = self.create_stsim_db(patches)
        else:
            raise "Invalid indicator"
            
        self.detector = self.assign_detector(self.distance, self.database)
        if (self.detector is not None):
            self.detector.calculate_statistics()
            self.detector.calculate_acceptances()

        print (self.database.shape)



    def get_distance_density_from_model (self, valid_patches, density_shape, stride, patch_size):
        h,w = density_shape
        density = np.zeros(shape=(density_shape))
        normalizator = np.zeros(shape=(density_shape))

        model_valid = Model('stsim', None)
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
        density = self.detector.get_density_tresholded(density, False)

        return density


    def iou_coef(self, y_true, y_pred):
        y_true = np.array (y_true, dtype=int)
        y_pred = np.array (y_pred, dtype=int)

        intersection = np.logical_and(y_true,y_pred) # Logical AND
        union = np.logical_or(y_true,y_pred)    # Logical OR
        IOU = float(np.sum(intersection)/np.sum(union))
        return IOU




    ### Database Creation ###
    def create_stsim_db(self, patches):
        m = Metric()

        #creation of the feature vectores
        flags = np.ones((len(patches)), dtype=bool)
        database = np.array(m.STSIM_M(patches[0], self.height, self.orientations))
        for i in range(1,len(patches)):
            database = np.vstack ((database, m.STSIM_M(patches[i], self.height, self.orientations)))
            flags[i] = check_preprocessing(patches[i])
        
        return database, flags

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

        return None

