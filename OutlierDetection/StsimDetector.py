from OutlierDetection.OutlierDetector import OutlierDetector
import numpy as np
import scipy as sp

class StsimDetector (OutlierDetector):
    def __init__(self, database):
        super().__init__(database)
        self.mean = None
        self.var = None

    def calculate_statistics (self):
        self.mean = np.mean(self.data_train, 0)
        self.var = np.var (self.data_train, 0)

    def calculate_acceptances (self, alpha = 0.995):
        normal_scores = []
        for dt in self.data_train:
            normal_scores.append(self.calculate_distance(dt))
        self.normal_score = np.mean(np.array(normal_scores))
        self.treshold = self.calculate_quantile(np.array(normal_scores), alpha)
        

    def calculate_distance (self, f_valid):
        dists = (f_valid - self.mean) ** 2
        s = np.sum(dists/self.var)
        return np.sqrt(s)

    def get_density_tresholded (self, density, binarize=True):
        if (not binarize):
            density[density >= self.treshold] = self.treshold
            density = (density - np.min(density)) / (np.max(density) - np.min(density))
        else:
            density[density >= self.treshold] = 1.
            density[density < self.treshold] = 0
        return density




    """
    #Maggioni distance (?)
    def get_stsim_i_statistic (self):
        mean = np.mean (self.database, 0)
        vars = []
        for i in range(self.database.shape[1]):
            vars.append(np.var(self.database[:,i] - mean[i]))
        return mean, np.array(vars)
    """
    