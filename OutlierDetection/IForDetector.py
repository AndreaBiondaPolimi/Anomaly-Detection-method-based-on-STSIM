from OutlierDetection.OutlierDetector import OutlierDetector
from sklearn.ensemble import IsolationForest
import numpy as np
import scipy as sp

class IForDetector (OutlierDetector):
    def __init__(self, database):
        super().__init__(database)
        self.ifor = None

    def calculate_statistics (self):
        #self.split = int(self.data_train.shape[0] * 0.8)
        self.ifor = IsolationForest(random_state=0).fit(self.data_train)
    
    
    def calculate_acceptances (self, alpha = 0.995):
        self.normal_score = 1
    
    def calculate_distance (self, f_valid):
        f_valid = np.array(f_valid).reshape((1,-1))
        score = self.ifor.predict(f_valid)
        #score = self.ifor.score_samples(f_valid)
        #score = self.ifor.decision_function(f_valid)
        return score

    def get_density_tresholded (self, density, binarize=True):
        density = (density - np.min(density)) / (np.max(density) - np.min(density))
        return density    