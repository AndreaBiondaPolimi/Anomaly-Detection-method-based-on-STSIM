from OutlierDetection.StsimDetector import OutlierDetector
import numpy as np
import scipy as sp
from skimage import morphology 

class MahalanobisDetector (OutlierDetector):
    def __init__(self, database):
        super().__init__(database)
        self.mean = None
        self.cov = None
        self.cov_i = None

    def calculate_statistics (self):
        self.mean = np.mean(self.data_train, 0)
        self.cov = np.cov(self.data_train.T)
        self.cov_i = np.linalg.inv(self.cov)

    def calculate_acceptances (self, alpha = 0.98):
        normal_scores = self.calculate_distance(self.data_train)
        self.normal_score = np.mean(np.array(normal_scores))
        self.treshold = self.calculate_quantile(np.array(normal_scores), alpha)

    def calculate_distance (self, f_valids):
        y_valid = []
        for f_valid in f_valids:
            ret = sp.spatial.distance.mahalanobis(f_valid, self.mean, self.cov_i)
            y_valid.append(ret) 
        return np.array(y_valid)
    
    def get_density_tresholded (self, density, post_process=True):
        density[density >= self.treshold] = self.treshold
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