from STSIM.metrics import Metric
from DataLoader import load_patches
import cv2
import numpy as np
import matplotlib.pyplot as plt


class Model():
    def __init__(self, indictator):
        self.indicator = indictator
        self.database = None
        self.database_variances = None

    def model_create (self, patches, calculate_variances=True):
        if (self.indicator == 'stsim'):
            self.database = self.create_stsim_db(patches)
            if (calculate_variances):
                self.database_variances = self.calculate_db_vars_intraclass()
            print (self.database.shape)


    def get_distance_density_from_model (self, valid_patches, density_shape, stride, patch_size):
        density = np.zeros(shape=(density_shape))
        normalizator = np.zeros(shape=(density_shape))

        model_valid = Model('stsim')
        model_valid.model_create(valid_patches, False)
        valid_db = model_valid.database

        for j in range (int((density.shape[0] - patch_size) / stride) + 1):
            for i in range (int((density.shape[1] - patch_size) / stride) + 1):
                f_valid = valid_db [j*int((density.shape[1] - patch_size) / stride + 1) + i] 
                dens_val = self.calculate_stsim_distance(f_valid)
                
                density[(j*stride):(j*stride)+patch_size, (i*stride):(i*stride)+patch_size] += dens_val
                
                normalizator[(j*stride):(j*stride)+patch_size, (i*stride):(i*stride)+patch_size] += 1

                
        density = density / normalizator
        density[density>50] = 50
        density = (density - np.min(density)) / (np.max(density) - np.min(density))

        return density




    def create_stsim_db(self, patches):
        m = Metric()
        
        #creation of the feature vectores
        database = np.array(m.STSIM_M(patches[0]))
        for i in range(1,len(patches)):
            database = np.vstack ((database, m.STSIM_M(patches[i])))
        
        return database

    def calculate_db_vars_overall (self):
        vars = []
        for i in range(self.database.shape[1]):
            vars.append(np.var(self.database[:,i]))

        return np.array(vars)

    def calculate_db_vars_intraclass (self):
        means = []
        for i in range(self.database.shape[1]):
            means.append(np.mean(self.database[:,i]))

        vars = []
        for i in range(self.database.shape[1]):
            vars.append(np.var(self.database[:,i] - means[i]))

        return np.array(vars)


    def calculate_stsim_distance (self, f_valid):
        f1 = f_valid

        distances = []
        for i in range (self.database.shape[0]):
            f2 = self.database[i]
            dists = (f1 - f2) ** 2
            s = np.sum(dists/self.database_variances)
            distances.append (np.sqrt(s))
            #distances.append (np.sum(dists))

        return sum(distances) / len(distances)

