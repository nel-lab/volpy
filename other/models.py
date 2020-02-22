#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jul 30 17:20:27 2019

@author: hoss
"""
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Concatenate, Dense, Dropout, Flatten, Activation, Reshape, Lambda
from tensorflow.keras.layers import Conv2D, BatchNormalization, AveragePooling2D, GlobalAveragePooling2D, SpatialDropout2D, Reshape
from tensorflow.keras.layers import MaxPooling2D, UpSampling2D
from tensorflow.keras.layers import Conv3D, AveragePooling3D, GlobalAveragePooling3D, SpatialDropout3D
from tensorflow.keras.layers import MaxPooling3D, UpSampling3D
from tensorflow.keras.backend import squeeze, expand_dims
from tensorflow.keras.utils import plot_model
from tensorflow.keras import regularizers
from tensorflow.keras import backend as K
from tensorflow.keras.losses import mean_squared_error
import numpy as np
import tensorflow as tf
import keras

def FCN_2D( input_shape, n_kernels ):
    inp = Input(shape=input_shape)
    # Block 1
    x = Conv2D(n_kernels[0], (3, 3), activation='relu', padding='same')(inp)
    x = BatchNormalization()(x)
    x = Conv2D(n_kernels[0], (3, 3), activation='relu', padding='same')(x)
    x = BatchNormalization()(x)
    x = MaxPooling2D((2, 2), padding='same')(x)
    
    
    # Block 2
    x = Conv2D(n_kernels[1], (3, 3), activation='relu', padding='same')(x)
    x = BatchNormalization()(x)
    x = Conv2D(n_kernels[1], (3, 3), activation='relu', padding='same')(x)
    x = BatchNormalization()(x)
    x = MaxPooling2D((2, 2), padding='same')(x)
    
    
    # Block 3
    x = Conv2D(n_kernels[2], (3, 3), activation='relu', padding='same')(x)
    x = BatchNormalization()(x)
    x = Conv2D(n_kernels[2], (3, 3), activation='relu', padding='same')(x)
    x = BatchNormalization()(x)
    x = MaxPooling2D((2, 2), padding='same')(x)
    
    
    # Block 4
    x = Conv2D(n_kernels[3], (3, 3), activation='relu', padding='same')(x)
    x = BatchNormalization()(x)
    x = Conv2D(n_kernels[3], (3, 3), activation='relu', padding='same')(x)
    x = BatchNormalization()(x)
    x = MaxPooling2D((2, 2), padding='same')(x)
    
    
    # Block 5
    x = Conv2D(n_kernels[4], (3, 3), activation='relu', padding='same')(x)
    x = BatchNormalization()(x)
    x = Conv2D(n_kernels[4], (3, 3), activation='relu', padding='same')(x)
    x = BatchNormalization()(x)
    x = MaxPooling2D((2, 2), padding='same')(x)


    #
    x = Conv2D(n_kernels[5], (3, 3), activation='relu', padding='same')(x)
    x = Conv2D(n_kernels[5], (3, 3), activation='relu', padding='same')(x)
    code = UpSampling2D(2)(x)
    
    
    #
    x = Conv2D(n_kernels[4], (3, 3), activation='relu', padding='same')(code)
    x = Conv2D(n_kernels[4], (3, 3), activation='relu', padding='same')(x)
    x = UpSampling2D(2)(x)
    
    
    #
    x = Conv2D(n_kernels[3], (3, 3), activation='relu', padding='same')(x)
    x = Conv2D(n_kernels[3], (3, 3), activation='relu', padding='same')(x)
    x = UpSampling2D(2)(x)
    
    
    #
    x = Conv2D(n_kernels[2], (3, 3), activation='relu', padding='same')(x)
    x = Conv2D(n_kernels[2], (3, 3), activation='relu', padding='same')(x)
    x = UpSampling2D(2)(x)
    
    
    #
    x = Conv2D(n_kernels[1], (3, 3), activation='relu', padding='same')(x)
    x = Conv2D(n_kernels[1], (3, 3), activation='relu', padding='same')(x)
    x = UpSampling2D(2)(x)
    
    
    x = Conv2D(n_kernels[0], (3, 3), activation='relu', padding='same')(x)
    x = Conv2D(n_kernels[0], (3, 3), activation='relu', padding='same')(x)
    out = Conv2D(1, (3, 3), activation='signoid',  padding='same', name='phase')(x)
    
    model = Model(inp, out)
    return model


def unet_2D( input_shape, n_kernels ):
    inp = Input(shape = input_shape)
    # Block 1
    x1 = Conv2D(n_kernels[0], (3, 3), activation='relu', padding='same')(inp)
    x1 = BatchNormalization()(x1)
    x1 = Conv2D(n_kernels[0], (3, 3), activation='relu', padding='same')(x1)
    x1 = BatchNormalization()(x1)
    pool1 = MaxPooling2D((2, 2), padding='same')(x1)
    
    
    # Block 2
    x2 = Conv2D(n_kernels[1], (3, 3), activation='relu', padding='same')(pool1)
    x2 = BatchNormalization()(x2)
    x2 = Conv2D(n_kernels[1], (3, 3), activation='relu', padding='same')(x2)
    x2 = BatchNormalization()(x2)
    pool2 = MaxPooling2D((2, 2), padding='same')(x2)
    
    
    # Block 3
    x3 = Conv2D(n_kernels[2], (3, 3), activation='relu', padding='same')(pool2)
    x3 = BatchNormalization()(x3)
    x3 = Conv2D(n_kernels[2], (3, 3), activation='relu', padding='same')(x3)
    x3 = BatchNormalization()(x3)
    pool3 = MaxPooling2D((2, 2), padding='same')(x3)
    
    
    # Block 4
    x4 = Conv2D(n_kernels[3], (3, 3), activation='relu', padding='same')(pool3)
    x4 = BatchNormalization()(x4)
    x4 = Conv2D(n_kernels[3], (3, 3), activation='relu', padding='same')(x4)
    x4 = BatchNormalization()(x4)
    pool4 = MaxPooling2D((2, 2), padding='same')(x4)
    
    
    # Block 5
    x5 = Conv2D(n_kernels[4], (3, 3), activation='relu', padding='same')(pool4)
    x5 = BatchNormalization()(x5)
    x5 = Conv2D(n_kernels[4], (3, 3), activation='relu', padding='same')(x5)
    x5 = BatchNormalization()(x5)
    pool5 = MaxPooling2D((2, 2), padding='same')(x5)


    #
    code = Conv2D(n_kernels[5], (3, 3), activation='relu', padding='same')(pool5)
    code = Conv2D(n_kernels[5], (3, 3), activation='relu', padding='same')(code)
    code = UpSampling2D(2)(code)
    
    
    #
    up5 = Concatenate()([code,x5])
    up5 = Conv2D(n_kernels[4], (3, 3), activation='relu', padding='same')(up5)
    up5 = Conv2D(n_kernels[4], (3, 3), activation='relu', padding='same')(up5)
    up5 = UpSampling2D(2)(up5)
    
    
    #
    up4 = Concatenate()([up5,x4])
    up4 = Conv2D(n_kernels[3], (3, 3), activation='relu', padding='same')(up4)
    up4 = Conv2D(n_kernels[3], (3, 3), activation='relu', padding='same')(up4)
    up4 = UpSampling2D(2)(up4)
    
    
    #
    up3 = Concatenate()([up4,x3])
    up3 = Conv2D(n_kernels[2], (3, 3), activation='relu', padding='same')(up3)
    up3 = Conv2D(n_kernels[2], (3, 3), activation='relu', padding='same')(up3)
    up3 = UpSampling2D(2)(up3)
    
    
    #
    up2 = Concatenate()([up3,x2])
    up2 = Conv2D(n_kernels[1], (3, 3), activation='relu', padding='same')(up2)
    up2 = Conv2D(n_kernels[1], (3, 3), activation='relu', padding='same')(up2)
    up2 = UpSampling2D(2)(up2)
    
    
    up1 = Concatenate()([up2,x1])
    up1 = Conv2D(n_kernels[0], (3, 3), activation='relu', padding='same')(up1)
    up1 = Conv2D(n_kernels[0], (3, 3), activation='relu', padding='same')(up1)
    out = Conv2D(1, (3, 3), activation='sigmoid',  padding='same', name='phase')(up1)
    
    model = Model(inp, out)
    return model


def unet_3D( input_shape, n_kernels, k_size ):
    inp = Input(shape = input_shape)
    # Block 1
    x1 = Conv3D(n_kernels[0], (k_size, k_size, k_size), activation='relu', padding='same')(inp)
    x1 = BatchNormalization()(x1)
    x1 = Conv3D(n_kernels[0], (k_size, k_size, k_size), activation='relu', padding='same')(x1)
    x1 = BatchNormalization()(x1)
    pool1 = MaxPooling3D((2, 2, 2), padding='same')(x1)
    
    
    # Block 2
    x2 = Conv3D(n_kernels[1], (k_size, k_size, k_size), activation='relu', padding='same')(pool1)
    x2 = BatchNormalization()(x2)
    x2 = Conv3D(n_kernels[1], (k_size, k_size, k_size), activation='relu', padding='same')(x2)
    x2 = BatchNormalization()(x2)
    pool2 = MaxPooling3D((2, 2, 2), padding='same')(x2)
    
    
    # Block 3
    x3 = Conv3D(n_kernels[2], (k_size, k_size, k_size), activation='relu', padding='same')(pool2)
    x3 = BatchNormalization()(x3)
    x3 = Conv3D(n_kernels[2], (k_size, k_size, k_size), activation='relu', padding='same')(x3)
    x3 = BatchNormalization()(x3)
    pool3 = MaxPooling3D((2, 2, 2), padding='same')(x3)
    
    
    # Block 4
    x4 = Conv3D(n_kernels[3], (k_size, k_size, k_size), activation='relu', padding='same')(pool3)
    x4 = BatchNormalization()(x4)
    x4 = Conv3D(n_kernels[3], (k_size, k_size, k_size), activation='relu', padding='same')(x4)
    x4 = BatchNormalization()(x4)
    pool4 = MaxPooling3D((2, 2, 2), padding='same')(x4)
    
    
    # Block 5
    x5 = Conv3D(n_kernels[4], (k_size, k_size, k_size), activation='relu', padding='same')(pool4)
    x5 = BatchNormalization()(x5)
    x5 = Conv3D(n_kernels[4], (k_size, k_size, k_size), activation='relu', padding='same')(x5)
    x5 = BatchNormalization()(x5)
    pool5 = MaxPooling3D((2, 2, 2), padding='same')(x5)


    #
    code = Conv3D(n_kernels[5], (k_size, k_size, k_size), activation='relu', padding='same')(pool5)
    code = Conv3D(n_kernels[5], (k_size, k_size, k_size), activation='relu', padding='same')(code)
    code = UpSampling3D(2)(code)
    
    
    #
    up5 = Concatenate()([code,x5])
    up5 = Conv3D(n_kernels[4], (k_size, k_size, k_size), activation='relu', padding='same')(up5)
    up5 = Conv3D(n_kernels[4], (k_size, k_size, k_size), activation='relu', padding='same')(up5)
    up5 = UpSampling3D(2)(up5)
    
    
    #
    up4 = Concatenate()([up5,x4])
    up4 = Conv3D(n_kernels[3], (k_size, k_size, k_size), activation='relu', padding='same')(up4)
    up4 = Conv3D(n_kernels[3], (k_size, k_size, k_size), activation='relu', padding='same')(up4)
    up4 = UpSampling3D(2)(up4)
    
    
    #
    up3 = Concatenate()([up4,x3])
    up3 = Conv3D(n_kernels[2], (k_size, k_size, k_size), activation='relu', padding='same')(up3)
    up3 = Conv3D(n_kernels[2], (k_size, k_size, k_size), activation='relu', padding='same')(up3)
    up3 = UpSampling3D(2)(up3)
    
    
    #
    up2 = Concatenate()([up3,x2])
    up2 = Conv3D(n_kernels[1], (k_size, k_size, k_size), activation='relu', padding='same')(up2)
    up2 = Conv3D(n_kernels[1], (k_size, k_size, k_size), activation='relu', padding='same')(up2)
    up2 = UpSampling3D(2)(up2)
    
    
    up1 = Concatenate()([up2,x1])
    up1 = Conv3D(n_kernels[0], (k_size, k_size, k_size), activation='relu', padding='same')(up1)
    up1 = Conv3D(n_kernels[0], (k_size, k_size, k_size), activation='relu', padding='same')(up1)
    out = Conv3D(1, (k_size, k_size, k_size), activation='sigmoid',  padding='same', name='phase')(up1)
    
    model = Model(inp, out)
    return model