from OutlierDetection.StsimDetector import StsimDetector
import numpy as np
import scipy as sp

class MahalanobisDetector (StsimDetector):
    def __init__(self, database):
        super().__init__(database)
        self.mean = None
        self.cov = None
        self.cov_i = None

    def calculate_statistics (self):
        super().calculate_statistics()
        self.cov = np.cov(self.data_train.T)
        self.cov_i = np.linalg.inv(self.cov)

    def calculate_acceptances (self, alpha = 0.98):
        super().calculate_acceptances(alpha)

    def calculate_distance (self, f_valid):
        ret = sp.spatial.distance.mahalanobis(f_valid, self.mean, self.cov_i)
        return ret
    
    def get_density_tresholded (self, density, binarize=True):
        return super().get_density_tresholded(density, binarize)