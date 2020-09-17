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
from sklearn.decomposition import NMF
#%%
c, dview, n_processes = cm.cluster.setup_cluster(
        backend='local', n_processes=None, single_thread=False)
#%%
m1 = cm.load('403106_3min_rois_mc_lp.hdf5') 
#%%#
mcr = cm.load('/home/andrea/NEL-LAB Dropbox/Andrea Giovannucci/Kaspar-Andrea/exampledata/Other/403106_3min_rois_mc_lp_crop.hdf5')
#%%
mcr = cm.load('/home/andrea/NEL-LAB Dropbox/Andrea Giovannucci/Kaspar-Andrea/exampledata/Other/403106_3min_rois_mc_lp_crop_2.hdf5')
#%%
fname = '/home/nel/data/voltage_data/Marton/454597/Cell_0/40x_patch1/movie/40x_patch1_000_mc_small.hdf5'
#%%
fname = '/Users/agiovann/NEL-LAB Dropbox/NEL/Papers/VolPy/Marton/456462_Cell_3_40x_1xtube_10A3.hdf5'
fname = '/Users/agiovann/NEL-LAB Dropbox/NEL/Papers/VolPy/Marton/456462_Cell_5_40x_1xtube_10A6.hdf5'
fname = '/Users/agiovann/NEL-LAB Dropbox/NEL/Papers/VolPy/Marton/video_small_region/462149_Cell_1_40x_1xtube_10A1.tif'
#%%
mcr = cm.load(fname)[:]

mcr = (-mcr).removeBL()

ycr = mcr.to_2D() 

ycr = ycr - ycr.min()
#%%
mcr_lc = local_correlations_movie_offline(fname, window=100, stride=10, dview=dview, Tot_frames=10000)
ycr_lc = mcr_lc.to_2D()
#%%
immg = mcr.mean(axis=(1,2))
immg = (immg - np.min(immg))/(np.max(immg)-np.min(immg))
plt.plot(mcr_lc.mean(axis=(1,2)));plt.plot(dict1['v_sg'][100:]*5); plt.plot(immg[100:])
#%%
D_lc,tr_lc = spams.nmf(np.asfortranarray(ycr_lc.T), K=2, return_lasso=True)   
plt.figure();plt.plot(tr_lc.T.toarray()) 
plt.figure();plt.imshow(D_lc[:,0].reshape(mcr.shape[1:], order='F'), cmap='gray')
plt.figure();plt.imshow(D_lc[:,1].reshape(mcr.shape[1:], order='F'), cmap='gray')
plt.figure();plt.imshow(D_lc[:,2].reshape(mcr.shape[1:], order='F'), cmap='gray')
#%%
n_components = 4
model = NMF(n_components=n_components, init='nndsvd', max_iter=1000, verbose=True)
W = model.fit_transform(np.maximum(ycr_lc,0))
H = model.components_
plt.figure();plt.plot(W + np.arange(n_components)/1000) 
for i in range(n_components):
    plt.figure();plt.imshow(H[i].reshape(mcr.shape[1:], order='F'), cmap='gray')
#%%
model = NMF(n_components=1, init='nndsvd', verbose=True)
W = model.fit_transform(np.maximum(ycr,0))#, H=H, W=W)
H = model.components_
plt.figure();plt.plot(W) 
plt.figure();plt.imshow(H[0].reshape(mcr.shape[1:], order='F'), cmap='gray')
plt.figure();plt.imshow(H[1].reshape(mcr.shape[1:], order='F'), cmap='gray')
#%%
#tr_lc, D_lc = spams.nmf(np.asfortranarray(ycr_lc), K=3, return_lasso=True)   
#plt.figure();plt.plot(tr_lc) 
#plt.figure();plt.imshow(D_lc.T.toarray()[:,0].reshape(mcr.shape[1:], order='F'), cmap='gray')
#plt.figure();plt.imshow(D_lc.T.toarray()[:,1].reshape(mcr.shape[1:], order='F'), cmap='gray')
#plt.figure();plt.imshow(D_lc.T.toarray()[:,2].reshape(mcr.shape[1:], order='F'), cmap='gray')
#%%
D_lc,tr_lc = spams.nmf(np.asfortranarray(ycr.T), K=2, return_lasso=True, )   
plt.figure();plt.plot(tr_lc.T.toarray()) 
plt.figure();plt.imshow(D_lc[:,0].reshape(mcr.shape[1:], order='F'), cmap='gray')
plt.figure();plt.imshow(D_lc[:,1].reshape(mcr.shape[1:], order='F'), cmap='gray')
plt.figure();plt.imshow(D_lc[:,2].reshape(mcr.shape[1:], order='F'), cmap='gray')
#%%
(D,model) = spams.trainDL(np.asfortranarray(ycr.T), K=3, D=D_lc, lambda1=10, return_model=True)
plt.figure();plt.imshow(D[:,0].reshape(mcr.shape[1:], order='F'), cmap='gray')
plt.figure();plt.imshow(D[:,1].reshape(mcr.shape[1:], order='F'), cmap='gray')
plt.figure();plt.imshow(D[:,2].reshape(mcr.shape[1:], order='F'), cmap='gray')

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
#%%
idx = np.round((dict1['v_sp']-dict1['v_t'][0])/np.median(np.diff(dict1['v_t']))).astype(np.int)
plt.figure(); plt.imshow(np.median(mcr_lc[idx[:1000]+3], axis=0))
#%%
plt.figure();plt.plot(dict1['v_t'],W)
plt.plot(dict1['e_sp'],[1]*len(dict1['e_sp']),'k|')
plt.plot(dict1['e_sp'],[2]*len(dict1['e_sp']),'k|')
plt.plot(dict1['e_sp'],[3]*len(dict1['e_sp']),'k|')
plt.plot(dict1['e_sp'],[4]*len(dict1['e_sp']),'k|')
plt.plot(dict1['e_sp'],[5]*len(dict1['e_sp']),'k|')
plt.plot(dict1['v_t'], dict1['v_sg']*100)

#%%
fe = slice(0,10000)
from scipy.optimize import nnls
Cf = np.array([nnls(H.T,y)[0] for y in ycr[fe]])
#%%
trr = Cf[:,:]
plt.plot(dict1['v_t'][fe],(trr-np.min(trr, axis=0))/(np.max(trr, axis=0)-np.min(trr, axis=0)))
trep = dict1['v_sg'][fe]
plt.plot(dict1['v_t'][fe], (trep-np.min(trep))/(np.max(trep)-np.min(trep)),'c')
eph = dict1['e_sg'][0:100000]
plt.plot(dict1['e_t'][00000:100000], (eph-np.min(eph))/(np.max(eph)-np.min(eph)),'k')
#plt.plot(dict1['e_sp'],[1]*len(dict1['e_sp']),'k|')
#%%
cm.movie(to_3D((H[[2,3],:].T@Cf[:,[2,3]].T).T,shape=mcr.shape, order='F')).play(magnification=4)
#%%
cm.movie(mcr-to_3D((H[[0,1],:].T@Cf[:,[0,1]].T).T,shape=mcr.shape, order='F')).play(magnification=4, fr=10)
    
