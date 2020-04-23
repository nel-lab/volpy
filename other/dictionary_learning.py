#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Apr 20 09:34:51 2020
This file uses spams package for nmf and nnsc
@author: Andrea Giovannucci
"""

import caiman as cm
import pylab as plt
import numpy as np
from caiman.summary_images import local_correlations_movie_offline
import spams
#%%
m1 = cm.load('403106_3min_rois_mc_lp.hdf5') 
#%%#
mcr = cm.load('/home/andrea/NEL-LAB Dropbox/Andrea Giovannucci/Kaspar-Andrea/exampledata/Other/403106_3min_rois_mc_lp_crop.hdf5')
#%%
mcr = cm.load('/home/andrea/NEL-LAB Dropbox/Andrea Giovannucci/Kaspar-Andrea/exampledata/Other/403106_3min_rois_mc_lp_crop_2.hdf5')
#%%
fname = '/home/nel/data/voltage_data/Marton/454597/Cell_0/40x_patch1/movie/40x_patch1_000_mc_small.hdf5'

#%%
mcr = cm.load(fname)

ycr = mcr.to_2D() 

ycr = ycr - ycr.min()

mcr_lc = local_correlations_movie_offline(fname)
ycr_lc = mcr_lc.to_2D()
#%%
D_lc,tr_lc = spams.nmf(np.asfortranarray(ycr_lc.T), K=2, return_lasso=True)   
plt.figure();plt.plot(tr_lc.T.toarray()) 
plt.figure();plt.imshow(D_lc[:,0].reshape(mcr.shape[1:], order='F'), cmap='gray')
#%%
D,tr = spams.trainDL(np.asfortranarray(ycr.T), K=2, D=D_lc, lambda1=0)


#%%
D,tr = spams.nnsc(np.asfortranarray(ycr.T), K=2, return_lasso=True, lambda1=0)
#%%
D,tr = spams.nmf(np.asfortranarray(np.abs(ycr.T)), K=2, return_lasso=True) 
#%%
D,tr = spams.nnsc(np.asfortranarray(ycr.T), K=2, return_lasso=True, lambda1=1)
#%%
plt.figure();plt.plot(tr.T.toarray()) 
plt.figure();plt.imshow(D[:,1].reshape(mcr.shape[1:], order='F'), cmap='gray');plt.title('comp1') 
plt.figure();plt.imshow(D[:,0].reshape(mcr.shape[1:], order='F'), cmap='gray');plt.title('comp0')

#%%
plt.figure();plt.plot(tr.T.toarray()) 
plt.figure();plt.imshow(D[:,1].reshape(mcr.shape[1:], order='F'), cmap='gray') 