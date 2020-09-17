#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Apr  9 10:31:51 2020
This file is for reconstructed video
@author:caichangjia
"""

#%% Reconstructed video
scope = (2000,4000)
idx = np.where(np.array(vpy.estimates['locality']) > 0)[0]
est = np.load('/home/nel/data/voltage_data/volpy_paper/reconstructed/estimates.npz',allow_pickle=True)['arr_0'].item()
fnames = ['/home/nel/data/voltage_data/volpy_paper/memory/403106_3min_10000._rig__d1_512_d2_128_d3_1_order_F_frames_10000_.mmap']


def reconstructed_video(est, fnames, idx, scope):
    mv = cm.load(fnames, fr=400)[scope[0]:scope[1]]
    dims = (mv.shape[1], mv.shape[2])
    mv_bl = mv.computeDFF(secsWindow=0.1)[0]
    mv = (mv-mv.min())/(mv.max()-mv.min())
    mv_bl = -mv_bl
    mv_bl[mv_bl<np.percentile(mv_bl,3)] = np.percentile(mv_bl,3)
    mv_bl[mv_bl>np.percentile(mv_bl,98)] = np.percentile(mv_bl,98)
    mv_bl = (mv_bl - mv_bl.min())/(mv_bl.max()-mv_bl.min())
    
    for i in idx:
        est['weights'][i][est['weights'][i]<0] = 0
    
    A = np.array(est['weights'])[idx].transpose([1,2,0]).reshape((-1,len(idx)))
    C = np.array(est['t_rec'])[idx,scope[0]:scope[1]]
    mv_rec = np.dot(A, C).reshape((dims[0],dims[1],scope[1]-scope[0])).transpose((2,0,1))    
    mv_rec = cm.movie(mv_rec,fr=400)
    mv_rec = (mv_rec - mv_rec.min())/(mv_rec.max()-mv_rec.min())
    mv_all = cm.concatenate((mv,mv_bl,mv_rec),axis=2)
    
    return mv_all

 