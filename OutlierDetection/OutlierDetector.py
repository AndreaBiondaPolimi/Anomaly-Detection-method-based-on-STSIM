from scipy.stats.mstats import mquantiles
import numpy as np

class OutlierDetector ():
    def __init__(self, database):
        self.data_train = database
        self.treshold = None
        
    def calculate_statistics (self):
        pass

    def calculate_distance (self, f_valid):
        pass

    def get_treshold (self):
        pass

    def calculate_acceptances (self, alpha_set = 0.995):
        pass


    

    def calculate_quantile (self, normal_scores, alpha_val):
        return mquantiles(normal_scores, alpha_val)