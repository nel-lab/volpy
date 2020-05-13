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
from scipy.io import savemat
import scipy
#%% DATA FOR IDENT
file_list = ['/Users/agiovann/NEL-LAB Dropbox/NEL/Papers/VolPy/Marton/data_new/454597_Cell_0_40x_patch1_output.npz', 
             '/Users/agiovann/NEL-LAB Dropbox/NEL/Papers/VolPy/Marton/data_new/456462_Cell_3_40x_1xtube_10A2_output.npz',
             '/Users/agiovann/NEL-LAB Dropbox/NEL/Papers/VolPy/Marton/data_new/456462_Cell_3_40x_1xtube_10A3_output.npz',
             '/Users/agiovann/NEL-LAB Dropbox/NEL/Papers/VolPy/Marton/data_new/456462_Cell_5_40x_1xtube_10A5_output.npz',
             '/Users/agiovann/NEL-LAB Dropbox/NEL/Papers/VolPy/Marton/data_new/456462_Cell_5_40x_1xtube_10A7_output.npz',
             '/Users/agiovann/NEL-LAB Dropbox/NEL/Papers/VolPy/Marton/data_new/462149_Cell_1_40x_1xtube_10A1_output.npz', 
             '/Users/agiovann/NEL-LAB Dropbox/NEL/Papers/VolPy/Marton/data_new/462149_Cell_1_40x_1xtube_10A2_output.npz',
             '/Users/agiovann/NEL-LAB Dropbox/NEL/Papers/VolPy/Marton/data_new/456462_Cell_5_40x_1xtube_10A6_output.npz']

for file in file_list:
    with np.load(file, allow_pickle=True) as ld:
        locals().update(ld)
    #volt = dict1['sweep_time'], dict1['e_sg'],dict1['e_sp'], dict1['e_t'],dict1['e_sub'], dict1['v_sg'], dict1_v_sp_, dict1['v_t'], dict1['v_sub']
    plt.figure()
    sw = sweep_time[1]
    sw_0 = sweep_time[0]
    np.max(np.diff(sw_0))
    print(sw[-1] - sw[0])      
    idx_img = np.where((v_t<sw[-1]) & (v_t>sw[0]) )[0]
    img = v_sg[idx_img]
    img_sub = v_sub[idx_img]
    t_img = v_t[idx_img]
    t_img -= t_img[0]
    idx_e = np.where((e_t<sw[-1]) & (e_t>sw[0]))[0]
    e = e_sg[idx_e]
    t_e = e_t[idx_e]
#    t_e = sweep_time[1]
    t_e -= t_e[0]
    idx_sp = np.where((e_sp<sw[-1]) & (e_sp>sw[0]))[0]
    spikes = e_sp[idx_sp] - sw[0]
    spike_idx = [np.argmin(np.abs(t_img-sp)) for sp in spikes]
    spike_vec = np.zeros_like(t_img)
    spike_vec[spike_idx] = 1
    e_ds,t_ds = scipy.signal.resample(e, len(t_img),t=t_e)
#    scipy.signal.decimate(x, q, n=None, ftype='iir', axis=-1, zero_phase=True)[source]
    if np.median(np.diff(e_t))<0.0025:
        print('slow spike')
        img_no_sub=np.roll(img-img_sub,1)
    else:
        img_no_sub=np.roll(img-img_sub,1)
        
    savemat(file[:-3]+'mat', {'t_img':t_img, 'img':img, 'img_sub':img_sub, 'spike_vec':spike_vec, 'spike_idx':spike_idx, 'e_ds':e_ds, 'img_no_sub':np.roll(img_no_sub,1)})
#    plt.plot(e)
    plt.plot(t_img, img)
    plt.plot(t_img, img_sub)
#    plt.plot(t_e, e)
    plt.plot(t_img, e_ds)
    plt.plot(t_img[spike_idx],[np.max(img)]*len(spike_idx),'r|')
    
