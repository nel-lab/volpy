#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
This file creates figure for spikepursuit pipeline, 
visualization of subthreshold traces
@author: caichangjia & andrea
"""
#%%
import os
import numpy as np
import matplotlib
matplotlib.rcParams['pdf.fonttype'] = 42
matplotlib.rcParams['ps.fonttype'] = 42
import matplotlib.pyplot as plt

#%% Pipeline figure
root_dir = '/home/nel/Code/NEL_LAB/volpy/figures/figure2_method'
sg = np.load(os.path.join(root_dir, 'high_pass.npz'))['arr_0']
sg0 = np.load(os.path.join(root_dir, 'bg_remove.npz'))['arr_0']
sg1 = np.load(os.path.join(root_dir, 'first_trace.npz'))['arr_0']
sp1 = np.load(os.path.join(root_dir, 'first_spike.npz'))['arr_0']
sg2 = np.load(os.path.join(root_dir, 'second_trace.npz'))['arr_0']
sp2 = np.load(os.path.join(root_dir, 'second_spike.npz'))['arr_0']
t1, t2 = np.load(os.path.join(root_dir, 'thresh.npz'))['arr_0']
sg_sub = np.load(os.path.join(root_dir, 'sub.npz'))['arr_0']
spatial = np.load(os.path.join(root_dir, 'spatial.npz'))['arr_0']

#%%
scope = [10500,11000]
sg = sg[scope[0]:scope[1]]
sg0 = sg0[scope[0]:scope[1]]
sg1 = sg1[scope[0]:scope[1]]
sp1 = [i-scope[0] for i in sp1 if i>scope[0] and i<scope[1]]
sg2 = sg2[scope[0]:scope[1]]
sp2 = [i-scope[0] for i in sp2 if i>scope[0] and i<scope[1]]
sg_sub = sg_sub[scope[0]:scope[1]]

plt.figure(); plt.imshow(summary_image[45:85,28:67,2],vmax=np.percentile(summary_image[:,:,2],98),cmap='gray');
plt.savefig(os.path.join(root_dir, 'one.pdf'))
plt.figure(); plt.plot(sg);plt.savefig(os.path.join(root_dir, 'two.pdf'))
plt.figure(); plt.plot(sg0);plt.savefig(os.path.join(root_dir, 'three.pdf'))
plt.figure(); plt.plot(sg1);
plt.scatter(sp1, np.ones(len(sp1))*1.1*sg1.max());
plt.hlines(t1, 0, scope[1]-scope[0],'gray','dashed');plt.savefig(os.path.join(root_dir, 'four.pdf'))
plt.figure(); plt.plot(sg2);
plt.scatter(sp2, np.ones(len(sp2))*1.1*sg2.max());
plt.hlines(t2, 0, scope[1]-scope[0],'gray','dashed');plt.savefig(os.path.join(root_dir, 'six.pdf'))
plt.figure();plt.imshow(spatial[45:85,28:67]);plt.colorbar();plt.savefig(os.path.join(root_dir, 'seven.pdf'))
plt.figure(); plt.plot(sg_sub);plt.savefig(os.path.join(root_dir, 'eight.pdf'))

#%% Various traces
import matplotlib
matplotlib.rcParams['pdf.fonttype'] = 42
matplotlib.rcParams['ps.fonttype'] = 42
#%%
idxs = 5,6,8
time_ = 1/400*np.arange(len(vpy.estimates['trace_raw'][0]))
counter=0
for idx in idxs:
    counter+=1
    plt.subplot(len(idxs),1,counter)
    plt.plot(time_,vpy.estimates['trace_raw'][idx])
    plt.plot(time_,vpy.estimates['trace_processed'][idx])
    plt.plot(time_,vpy.estimates['trace_recons'][idx])
    plt.plot(time_,vpy.estimates['trace_sub'][idx])
    peaks =  vpy.estimates['trace_recons'][idx][vpy.estimates['spikes'][idx]]
    max_peaks = np.max(vpy.estimates['trace_raw'][idx])
    plt.plot(time_[vpy.estimates['spikes'][idx]],np.maximum(peaks, max_peaks)*1.1, 'r|')    
    plt.xlim([0,4])
    if counter !=3:
        plt.axis('off')
​
plt.xlabel('time(s)')
plt.legend(['t_0','t','reconstructted','t_{sub}','spikes'])


