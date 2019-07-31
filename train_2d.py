#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jul 30 17:49:08 2019

@author: hoss
"""

import keras
import tensorflow as tf
from models import unet_2D
from SegDataGen import SegmentationDataGenerator
import numpy as np

#%%
batch_size = 32
epochs = 100
in_shape = (256,256,2)

nkernels = [8,16,32,64,128,256]

model = unet_2D(in_shape, nkernels)

print(model.summary())

x = np.random.rand(121,200,256,2)
y = np.ones((121,200,256,2))

n_steps = len(x)/batch_size

#%%
datagen = SegmentationDataGenerator(rotation_range=10.,
                                     width_shift_range=0.1,
                                     height_shift_range=0.1,
                                     shear_range=0.1,
                                     zoom_range=0.0,
                                     horizontal_flip=True,
                                     vertical_flip=True,
                                     dim_ordering = 'tf')

iterator = datagen.flow(x, y, batch_size)

#%%
for epoch in range(epochs):
    for step in n_steps:
        x,y = iterator.next()
        model.fit(x,y,verbose=0)
        
    y_pred = model.predict(validation data)
    print(get_metric(y_true,y_pred))

#%%
