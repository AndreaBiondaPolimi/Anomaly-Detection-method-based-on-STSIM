from OutlierDetection.OutlierDetector import OutlierDetector
from sklearn.neighbors import KernelDensity
from DataLoader import load_patches
import numpy as np
import scipy as sp
from skimage import morphology 

class KdeDetector (OutlierDetector):
    def __init__(self, database):
        super().__init__(database)
        self.kde = None

    def calculate_statistics (self):
        self.split = int(self.data_train.shape[0] * 0.8)
        self.kde = KernelDensity(kernel='gaussian', bandwidth=0.2).fit(self.data_train[:self.split,:])

    def calculate_acceptances (self, alpha):
        normal_scores = self.kde.score_samples(self.data_train[self.split:,:])
        self.normal_score = np.mean(np.array(normal_scores))
        self.treshold = self.calculate_quantile(normal_scores, 1. - alpha)
            
    def calculate_distance (self, f_valid):
        #f_valid = np.array(f_valid).reshape((1,-1))
        ret = self.kde.score_samples(f_valid)
        return ret

    def get_density_tresholded (self, density, post_process=True):
        density[density <= self.treshold] = self.treshold
        density = density * -1 
        if (np.max(density) - np.min(density) > 0):
            density = (density - np.min(density)) / (np.max(density) - np.min(density))
        else:
            if (np.max(density) != 0):
                density = density / density

        density_tresholded = np.copy(density)
        density_tresholded[density_tresholded < 1] = 0
        if post_process:
            kernel = morphology.disk(5)
            density_tresholded = morphology.opening(density_tresholded, kernel)
        
        return density, density_tresholded