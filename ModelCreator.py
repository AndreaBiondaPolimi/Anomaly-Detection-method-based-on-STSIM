from STSIM.metrics import Metric
from DataLoader import load_patches
import cv2
import numpy as np
import matplotlib.pyplot as plt


class Model():
    def __init__(self, indictator, distance):
        self.indicator = indictator
        self.distance = distance
        self.database = None
        self.database_variances = None
        self.database_covariance = None
        self.database_mean = None

    def model_create (self, patches):
        if (self.indicator == 'stsim'):
            self.database = self.create_stsim_db(patches)
        else:
            raise "Invalid indicator"
            
        if (self.distance == 'stsim-m'):
            self.database_variances = self.get_stsim_m_statistic()
        elif (self.distance == 'stsim-i'):
            self.database_variances = self.get_stsim_i_statistic()
        elif (self.distance == 'mahalanobis'):
            self.database_mean, self.database_covariance = self.get_mahalanobis_statistic()

        print (self.database.shape)



    def get_distance_density_from_model (self, valid_patches, density_shape, stride, patch_size):
        density = np.zeros(shape=(density_shape))
        normalizator = np.zeros(shape=(density_shape))

        model_valid = Model('stsim', None)
        model_valid.model_create(valid_patches)
        valid_db = model_valid.database

        for j in range (int((density.shape[0] - patch_size) / stride) + 1):
            for i in range (int((density.shape[1] - patch_size) / stride) + 1):
                f_valid = valid_db [j*int((density.shape[1] - patch_size) / stride + 1) + i] 
                dens_val = self.calculate_distance(f_valid)
                
                #print (dens_val)
                #plt.imshow(valid_patches[j*int((density.shape[1] - patch_size) / stride + 1) + i])
                #plt.show()

                density[(j*stride):(j*stride)+patch_size, (i*stride):(i*stride)+patch_size] += dens_val
                
                normalizator[(j*stride):(j*stride)+patch_size, (i*stride):(i*stride)+patch_size] += 1

                
        density = density / normalizator
        density[density>=50] = 50
        #density[density< 20] = 0

        density = (density - np.min(density)) / (np.max(density) - np.min(density))

        return density




    def create_stsim_db(self, patches):
        m = Metric()
        
        #creation of the feature vectores
        database = np.array(m.STSIM_M(patches[0]))
        for i in range(1,len(patches)):
            database = np.vstack ((database, m.STSIM_M(patches[i])))
        
        return database



    ### Variance calculations ###

    #Return the variance of the database 
    def get_stsim_m_statistic (self):
        vars = []
        for i in range(self.database.shape[1]):
            vars.append(np.var(self.database[:,i]))
        return np.array(vars)

    #Return the variance of the database taking into account class belonging to
    def get_stsim_i_statistic (self):
        means = []
        for i in range(self.database.shape[1]):
            means.append(np.mean(self.database[:,i]))
        vars = []
        for i in range(self.database.shape[1]):
            vars.append(np.var(self.database[:,i] - means[i]))
        return np.array(vars)

    #Return the mean and the covariance matrix of the database
    def get_mahalanobis_statistic (self):
        mean = np.mean(self.database, 0)
        covariance = np.cov(self.database.T)
        return mean, covariance



    ### Distance calculations ###
    def calculate_distance (self, f_valid, k=5):
        if (self.distance == 'stsim-m' or self.distance == 'stsim-i'):
            return self.calculate_stsim_distance(f_valid,k)
        elif (self.distance == 'mahalanobis'):
            return self.calculate_mahalanobis_distance(f_valid)
        else:
            raise "Invalid distance metric"

    #Calculate distance as specified in the stsim papers
    def calculate_stsim_distance (self, f_valid, k=5):
        mean = np.mean(self.database, 0)
        dists = (f_valid - mean) ** 2
        s = np.sum(dists/self.database_variances)
        return np.sqrt(s)

    #Calculate the Mahalanobis distance (x-u)S(x-u)
    #TODO capire perchè da risultati grandi e diversi
    def calculate_mahalanobis_distance (self, f_valid):
        diff = f_valid - self.database_mean
        
        #div = np.diag(self.database_covariance)
        #div = np.diag(div)
        #ret = np.matmul(diff, div) #diff * self.database_variance
        ret = np.matmul(diff, self.database_covariance) #diff * self.database_covariance
        
        ret = np.matmul(ret, diff.T) #ret = ret * diff.T
        return np.sqrt(ret)

