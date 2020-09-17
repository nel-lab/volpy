#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jul 14 21:28:15 2020

@author: nel
"""
import os
import caiman as cm
base_folder = '/home/nel/Code/VolPy_many_things/Mask_RCNN/videos & imgs/summary imgs/mean_npz'
os.listdir(base_folder)
for file in os.listdir(base_folder):
    m = np.load(os.path.join(base_folder, file), allow_pickle=True)['arr_0']
    print(m.shape)
    cm.movie(m).save(os.path.join(base_folder, file[:-4]+'.tif'))
    
#%%
from caiman.source_extraction.volpy.spikepursuit import signal_filter
import numpy as np
from skimage import io
import matplotlib.pyplot as plt
from sklearn.preprocessing import normalize

trace = io.imread('/home/nel/Code/volpy_test/invivo-imaging/test_data/09282017Fish1-1/output/temporal_traces.tif')
spatial = io.imread('/home/nel/Code/volpy_test/invivo-imaging/test_data/09282017Fish1-1/output/spatial_footprints.tif')

for i in range(len(trace)):

i=1
plt.plot(trace[i]);plt.show()
plt.plot(signal_filter(trace[i], freq=15, fr=400));plt.show()
plt.imshow(spatial[:, i].reshape((44, 120), order='F'));plt.show()

vpy = np.load('/home/nel/data/voltage_data/simul_electr/johannes/09282017Fish1-1/estimates.npz', allow_pickle=True)['arr_0'].item()
plt.plot(vpy['trace_processed'][0][100:] / vpy['trace_processed'][0].max())
plt.plot(signal_filter(trace[i], freq=15, fr=400) / signal_filter(trace[i], freq=15, fr=400).max());plt.show()

