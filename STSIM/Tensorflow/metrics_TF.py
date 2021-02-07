import numpy as np
import tensorflow as tf
import itertools
import matplotlib.pyplot as plt
from STSIM.Tensorflow.SCFpyr_TF import SCFpyr_TF

class Metric:
    def STSIM_M (self, imgs_batch, height, orientations):
        im_batch_numpy = np.expand_dims(imgs_batch, axis=-1)
        #im_batch_numpy = np.expand_dims(im_batch_numpy, axis=0)
        im_batch_tf = tf.convert_to_tensor(im_batch_numpy, tf.complex128)

        pyr = SCFpyr_TF(
            height= height, 
            nbands= orientations,
            scale_factor= 2,
            precision = 64
        )
        coeff = pyr.build(im_batch_tf)

        """
        higpass = coeff[0]
        bandpass = coeff[1]
        lowpass = coeff[2]
        for i in range (5,6):
            print(i)
            plt.imshow(np.squeeze(im_batch_numpy[i]))
            plt.show()

            plt.imshow(np.real(higpass[i].numpy()))
            plt.show()

            for orient in bandpass:
                plt.imshow(np.real(orient[i].numpy()) )
                plt.show()
        
            plt.imshow(np.real(lowpass[i].numpy()))
            plt.show()
        """

        features = []

        # Extract mean / var / H_autocorr/ V_autocorr from every subband
        features = self.extract_basic(coeff[0], False, features)
        for orients in coeff[1:-1]:
            for band in orients:
                features = self.extract_basic(band, True, features)
        features = self.extract_basic(coeff[-1], False, features)            

        # Extract correlation across orientations
        for orients in coeff[1:-1]:
            for (s1, s2) in list(itertools.combinations(orients, 2)):
                s1 = tf.math.abs(s1)
                s2 = tf.math.abs(s2)
                #features.append(tf.reduce_mean(s1 * s2, axis = (1,2)))
                features.append(tf.reduce_mean(s1 * s2, axis = (1,2)) / ((tf.math.reduce_std(s1, axis = (1,2)) * tf.math.reduce_std(s2, axis = (1,2))))) #(why not this?)

        # Extract correlation across heigth
        for orient in range(len(coeff[1])):
            for height in range(len(coeff) - 3):
                s1 = tf.math.abs(coeff[height + 1][orient])
                s2 = tf.math.abs(coeff[height + 2][orient])

                new_shape = tf.dtypes.cast(tf.shape(s1)[1:3] / 2, tf.int32)
                s1 = tf.image.resize(tf.expand_dims(s1, axis=-1), new_shape)
                s1 = tf.cast(tf.squeeze(s1), tf.float64)
                s2 = tf.cast(tf.squeeze(s2), tf.float64)

                features.append(tf.reduce_mean(s1 * s2, axis = (1,2)) / ((tf.math.reduce_std(s1, axis = (1,2)) * tf.math.reduce_std(s2, axis = (1,2)))))
        

        return tf.stack([f for f in features], axis=1)



    def extract_basic(self,band, is_complex , features):
        if (is_complex):
            band = tf.math.abs(band)

        shiftx = tf.roll(band, 1, axis = 1)
        shifty = tf.roll(band, 1, axis = 2)
        
        features.append(tf.reduce_mean(band, axis = (1,2)))
        features.append(tf.math.reduce_variance(band, axis = (1,2)))
        features.append(tf.math.reduce_mean(shiftx * band, axis = (1,2)) / tf.math.reduce_variance(band, axis = (1,2)))
        features.append(tf.math.reduce_mean(shifty * band, axis = (1,2)) / tf.math.reduce_variance(band, axis = (1,2)))
    
        return features