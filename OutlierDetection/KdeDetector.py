from OutlierDetection.OutlierDetector import OutlierDetector
from sklearn.neighbors import KernelDensity
from DataLoader import load_patches
import numpy as np
import scipy as sp

class KdeDetector (OutlierDetector):
    def __init__(self, database):
        super().__init__(database)
        self.kde = None

    def calculate_statistics (self):
        self.split = int(self.data_train.shape[0] * 0.8)
        self.kde = KernelDensity(kernel='gaussian', bandwidth=0.2).fit(self.data_train[:self.split,:])

    def calculate_acceptances (self, alpha = 0.995):
        normal_scores = self.kde.score_samples(self.data_train[self.split:,:])
        self.normal_score = np.mean(np.array(normal_scores))
        self.treshold = self.calculate_quantile(normal_scores, 1. - alpha)
            
    def calculate_distance (self, f_valid):
        f_valid = np.array(f_valid).reshape((1,-1))
        ret = self.kde.score_samples(f_valid)
        return ret

    def get_density_tresholded (self, density, binarize=True):
        if (not binarize):
            density[density <= self.treshold] = self.treshold
            density = density * -1 
            density = (density - np.min(density)) / (np.max(density) - np.min(density))
        else:
            density[density <= self.treshold] = 1
            density[density > self.treshold] = 0
        return density    