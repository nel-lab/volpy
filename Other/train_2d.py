#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jul 30 17:49:08 2019

@author: hoss
"""
import sys
sys.path.append('/home/nel/Code/VoImAn')
import keras
import tensorflow as tf
from models import unet_2D
from SegDataGen import SegmentationDataGenerator
import numpy as np
from keras.models import *
from keras.layers import *
from keras.optimizers import Adam


#%%
batch_size = 32
epochs = 200
#in_shape = (256,256,2)
in_shape = (None,None,2)
nkernels = [8,16,32,64,128,256]

model = unet_2D(in_shape, nkernels)

print(model.summary())

#x = Xtrain
#y = Ytrain

n_steps = 10

#%%
datagen = SegmentationDataGenerator(rotation_range=10.,
                                     width_shift_range=0.1,
                                     height_shift_range=0.1,
                                     shear_range=0.1,
                                     zoom_range=0.0,
                                     horizontal_flip=True,
                                     vertical_flip=True,
                                     dim_ordering = 'tf',
                                     random_crops = (64,64))

iterator = datagen.flow(Xtrain, Ytrain, batch_size)

model.compile(optimizer = tf.keras.optimizers.Adam(lr = 2e-4), loss = 'mse', metrics = ['mse'])


#%%
for epoch in range(epochs):
    for step in range(n_steps):
        x,y = iterator.next()
        model.fit(x,y,verbose=0)
        
    #Ypred = model.predict(Xtest_crop)
    #print(get_metric(Ytest_crop,Ypred))
    model.evaluate(Xtest,Ytest)
    #model.evaluate(Xtest_crop, Ytest_crop)

#%%
#x,y = iterator.next()
    
#%%
    Ypred = model.predict(Xtest)
    Ypred = model.predict(Xtrain)

#%%
i=1
f, (ax1, ax2, ax3, ax4) = plt.subplots(1, 4, sharey=True)
ax1.imshow(Xtest[i,:,:,0])
ax1.set_title('Raw')
ax2.imshow(Xtest[i,:,:,1])
ax2.set_title('Corr')
ax3.imshow(Ytest[i,:,:,0])
ax3.set_title('Groundtruth')
ax4.imshow(Ypred[i,:,:,0])
ax4.set_title('Unet')

#%%
i=0
f, (ax1, ax2, ax3, ax4) = plt.subplots(1, 4, sharey=True)
ax1.imshow(Xtrain[i,:,:,0])
ax1.set_title('Raw')
ax2.imshow(Xtrain[i,:,:,1])
ax2.set_title('Corr')
ax3.imshow(Ytrain[i,:,:,0])
ax3.set_title('Groundtruth')
ax4.imshow(Ypred[i,:,:,0])
ax4.set_title('Unet')

#%%
nx = np.int(Xtest.shape[1]/in_shape[0])
ny = np.int(Xtest.shape[2]/in_shape[1])     
Xtest_crop = []
Ytest_crop = []
index = 0
for i in range(nx):
    for j in range(ny):
        xt = Xtest[:,i*in_shape[0]:(i+1)*in_shape[0],j*in_shape[1]:(j+1)*in_shape[1],:]
        yt = Ytest[:,i*in_shape[0]:(i+1)*in_shape[0],j*in_shape[1]:(j+1)*in_shape[1],:]
        Xtest_crop.append(xt)
        Ytest_crop.append(yt)
        index = index + 1
        
Xtest_crop = np.concatenate(Xtest_crop,axis=0)
Ytest_crop = np.concatenate(Ytest_crop,axis=0)


#%%
Ypred = model.predict(Xtest_crop)

Xr = np.zeros(Xtest.shape)
Yr = np.zeros(Ytest.shape)
Y_output = np.zeros(Ytest.shape)
index = 0
for i in range(nx):
    for j in range(ny):
        Xr[:,i*in_shape[0]:(i+1)*in_shape[0],j*in_shape[1]:(j+1)*in_shape[1],:] = Xtest_crop[index*2:(index*2+2),:,:,:]
        Yr[:,i*in_shape[0]:(i+1)*in_shape[0],j*in_shape[1]:(j+1)*in_shape[1],:] = Ytest_crop[index*2:(index*2+2),:,:,:]
        Y_output[:,i*in_shape[0]:(i+1)*in_shape[0],j*in_shape[1]:(j+1)*in_shape[1],:] = Ypred[index*2:(index*2+2),:,:,:]

        index = index + 1


#%%
i=0
f, (ax1, ax2, ax3, ax4) = plt.subplots(1, 4, sharey=True)
ax1.imshow(Xtest[i,:,:,0])
ax1.set_title('Raw')
ax2.imshow(Xtest[i,:,:,1])
ax2.set_title('Corr')
ax3.imshow(Ytest[i,:,:,0])
ax3.set_title('Groundtruth')
ax4.imshow(Ypred[i,:,:,0])
ax4.set_title('Unet')

#%%
i=1
f, (ax1, ax2, ax3, ax4) = plt.subplots(1, 4, sharey=True)
ax1.imshow(Xtest[i,:,:,0])
ax1.set_title('Raw')
ax2.imshow(Xtest[i,:,:,1])
ax2.set_title('Corr')
ax3.imshow(Ytest[i,:,:,0])
ax3.set_title('Groundtruth')
ax4.imshow(Y_output[i,:,:,0])
ax4.set_title('Unet')


    