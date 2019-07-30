#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jul 30 17:49:08 2019

@author: hoss
"""

import keras
import tensorflow as tf
from models import unet_2D

in_shape = (256,256,2)

nkernels = [8,16,32,64,128,256]

model = unet_2D(in_shape, nkernels)

print(model.summary())

model.fit()