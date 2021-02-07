from OutlierDetection.OutlierDetector import OutlierDetector
from OutlierDetection.MahalanobisDetector import MahalanobisDetector
import numpy as np
import scipy as sp

class LogLikelihoodDetector (MahalanobisDetector):
    def __init__(self, database):
        super().__init__(database)

    def calculate_statistics (self):
        super().calculate_statistics()

    def calculate_acceptances (self, alpha = 0.995):
        super().calculate_acceptances(alpha)

    #this calculation is not a distance but a likelihood, for this reason I have to invert the result
    def calculate_distance (self, f_valids):
        y_valid = []
        for f_valid in f_valids:
            k,_ = self.cov.shape
            maha = super().calculate_distance(f_valid)
            likelihood = -0.5 * (np.log(np.linalg.det(self.cov)) + maha + k * np.log(2 * np.pi))
            y_valid.append(1. / likelihood ) 
        return np.array(y_valid)
    
    def get_density_tresholded (self, density, binarize=True):
        return super().get_density_tresholded(density, binarize)    