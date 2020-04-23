#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Created on Fri Apr 17 17:08:11 2020
Files for analyzing Marton's GT data
@author: caichangjia
"""

import numpy as np
import os
import json
movie_dir = '/home/nel/data/voltage_data/Marton/454597/Cell_0/40x_patch1/'
movie_dir = '/home/nel/data/voltage_data/Marton/456462/Cell_3/40x_1xtube_10A2'

frame_times = np.load(os.path.join(movie_dir, 'frame_times.npy'))
voltage = np.load(os.path.join(movie_dir,'ephys/sweep_31.npz'))['voltage']
np.load(os.path.join(movie_dir,'ephys/sweep_31.npz'))['time']

#%% Ephys data
ephys_files_dir = os.path.join(movie_dir,'ephys')
ephys_files = sorted(os.listdir(ephys_files_dir))
#ephys_files

#%%
sweep_time = list()
sweep_response = list()
sweep_stimulus = list()
sweep_metadata = list()
for ephys_file in ephys_files:
    if ephys_file[-3:]=='npz':
        data_dict = np.load(os.path.join(ephys_files_dir,ephys_file))
        sweep_time.append(data_dict['time'])
        sweep_response.append(data_dict['voltage'])
        sweep_stimulus.append(data_dict['stimulus'])
        with open(os.path.join(ephys_files_dir,ephys_file[:-3]+'json')) as json_file:
            sweep_metadata.append(json.load(json_file))     
sweep_metadata[0] # here is an example of a sweep metadata

#%%
%matplotlib inline
xlimits = [frame_times[0],frame_times[-1]]
#xlimits = [400, 405]
fig=plt.figure()
ax_ephys = fig.add_axes([0,0,2,.8])
ax_stim = fig.add_axes([0,-.5,2,.4])
for time,response,stimulus in zip(sweep_time,sweep_response,sweep_stimulus):
    ax_ephys.plot(time,response,'k-')
    ax_stim.plot(time,stimulus,'k-')
if dff is not None:
    ax_ophys = fig.add_axes([0,1,2,.8])

    ax_ophys.plot(frame_times,dff,'g-')
    ax_ophys.autoscale(tight = True)
    #ax_ophys.invert_yaxis()
    ax_ophys.set_xlim(xlimits)
    ax_ophys.set_ylabel('dF/F')
    vals = ax_ophys.get_yticks()
    ax_ophys.set_yticklabels(['{:,.2%}'.format(x) for x in vals])
    
ax_ephys.set_xlim(xlimits)
ax_ephys.set_ylabel('Membrane potential (mV)')
ax_stim.set_xlim(xlimits)
ax_stim.set_xlabel('Time from obtaining whole cell (s)')
ax_stim.set_ylabel('Stimulus (pA)')

#%% For one video
%matplotlib auto
dff = vpy.estimates['dFF'][0]
#dff = None
xlimits = [frame_times[0],frame_times[len(dff)-1]]
#xlimits = [400, 405]
fig=plt.figure()
ax_ephys = fig.add_axes([0,0,2,.8])
ax_stim = fig.add_axes([0,-.5,2,.4])
for time,response,stimulus in zip(sweep_time,sweep_response,sweep_stimulus):
    ax_ephys.plot(time,response,'k-')
    ax_stim.plot(time,stimulus,'k-')

if dff is not None:
    ax_ophys = fig.add_axes([0,1,2,.8])

    ax_ophys.plot(frame_times[0:len(dff)], dff,'g-')
    ax_ophys.autoscale(tight = True)
    #ax_ophys.invert_yaxis()
    ax_ophys.set_xlim(xlimits)
    ax_ophys.set_ylabel('dF/F')
    vals = ax_ophys.get_yticks()
    ax_ophys.set_yticklabels(['{:,.2%}'.format(x) for x in vals])
    
ax_ephys.set_xlim(xlimits)
ax_ephys.set_ylabel('Membrane potential (mV)')
ax_stim.set_xlim(xlimits)
ax_stim.set_xlabel('Time from obtaining whole cell (s)')
ax_stim.set_ylabel('Stimulus (pA)')

#%% Some functions
def compare_with_ephys_match(sg_gt, sp_gt, sg, sp, timepoint, max_dist=None, scope=None, hline=None):
    """ Match spikes from ground truth data and spikes from inference.
    Args:
        timepoint: map between gt signal and inference signal (normally not one to one)
        scope: the scope of signal need matching
        hline: threshold of gt signal
    return: precision, recall and F1 score
    """
    # Adjust signal and spikes to the scope
    height = np.max(np.array(sg_gt.max(), sg.max()))
    sg_gt = sg_gt[scope[0]:scope[1]]
    sp_gt = sp_gt[np.where(np.logical_and(sp_gt>scope[0], sp_gt<scope[1]))]-scope[0]    
    sg = sg[np.where(np.multiply(timepoint>=scope[0],timepoint<scope[1]))[0]]
    sp = np.array([timepoint[i] - scope[0] for i in sp if timepoint[i]>=scope[0] and timepoint[i]<scope[1]])
    time = [i-scope[0] for i in timepoint if i < scope[1] and i >=scope[0]]
    
    # Plot signals and spikes
    plt.plot(sg_gt, color='b', label='ephys')
    plt.plot(sp_gt, 1.2*height*np.ones(sp_gt.shape),color='b', marker='.', ms=2, fillstyle='full', linestyle='none')
    plt.plot(time, sg, color='orange', label='VolPy')
    plt.plot(sp, 1*height*np.ones(len(sp)),color='orange', marker='.', ms=2, fillstyle='full', linestyle='none')
    plt.hlines(hline, 0, len(sg_gt), linestyles='dashed', color='gray')
    ax = plt.gca()
    ax.locator_params(nbins=7)
    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)
    
    # Distance matrix and find matches
    D = distance_spikes(s1=sp_gt, s2=sp, max_dist=max_dist)
    index_gt, index_method = find_matches(D)
    for i in range(len(index_gt)):
        plt.plot((sp_gt[index_gt[i]], sp[index_method[i]]),(1.15*height, 1.05*height), color='gray',alpha=0.5, linewidth=1)

    # Calculate measures
    TP = len(index_gt)
    FP = len(sp) - TP
    FN = len(sp_gt) - TP
    precision = TP / (TP + FP)
    recall = TP / (TP + FN)
    F1 = 2 * (precision * recall) / (precision + recall) 
    print('precision:',precision)
    print('recall:',recall)
    print('F1:',F1)      
    plt.legend(prop={'size': 6})
    plt.tight_layout()
    
    return precision, recall, F1    

from scipy.optimize import linear_sum_assignment
def distance_spikes(s1, s2, max_dist):
    """ Define distance matrix between two spike train.
    Distance greater than maximum distance is assigned one.
    """    
    D = np.ones((len(s1), len(s2)))
    for i in range(len(s1)):
        for j in range(len(s2)):
            if np.abs(s1[i] - s2[j]) > max_dist:
                D[i, j] = 1
            else:
                D[i, j] = (np.abs(s1[i] - s2[j]))/5/max_dist
    return D

def find_matches(D):
    """ Find matches between two spike train by solving linear assigment problem.
    Delete matches where their distance is greater than maximum distance
    """
    index_gt, index_method = linear_sum_assignment(D)
    del_list = []
    for i in range(len(index_gt)):
        if D[index_gt[i], index_method[i]] == 1:
            del_list.append(i)
    index_gt = np.delete(index_gt, del_list)
    index_method = np.delete(index_method, del_list)
    return index_gt, index_method

#%%
from scipy.signal import find_peaks
e_sg =  np.array(sweep_response).reshape(-1, order='C')
hline=-40
e_t = np.array(sweep_time).reshape(-1, order='C')
e_sp = e_t[find_peaks(e_sg, hline, distance=10)[0]]
plt.plot(e_t, e_sg, label='ephys', color='blue')
plt.hlines(hline, e_t[0], e_t[-1], linestyles='dashed', color='gray')
plt.plot(e_sp, np.max(e_sg)*1.1*np.ones(e_sp.shape),color='b', marker='.', ms=2, fillstyle='full', linestyle='none')

#%%    
v_sg = vpy.estimates['dFF'][0]
v_sp = vpy.estimates['spikes'][0]
#plt.plot(dff)
v_t = frame_times
v_sp = v_t[vpy.estimates['spikes'][0]]
plt.plot(v_t, v_sg, label='ephys', color='blue')
plt.plot(v_sp, np.max(v_sg)*1.1*np.ones(v_sp.shape),color='b', marker='.', ms=2, fillstyle='full', linestyle='none')

#%%
scope = [255, 325]

def spike_comparison(e_sg, e_sp, e_t, v_sg, v_sp, v_t, scope):
    e_sg = e_sg[np.where(np.multiply(e_t>=scope[0], e_t<=scope[1]))[0]]
    e_sg = (e_sg - np.mean(e_sg))/(np.max(e_sg)-np.min(e_sg))*np.max(v_sg)
    e_sp = e_sp[np.where(np.multiply(e_sp>=scope[0], e_sp<=scope[1]))[0]]
    e_t = e_t[np.where(np.multiply(e_t>=scope[0], e_t<=scope[1]))[0]]
    #plt.plot(e_t, e_sg, label='ephys', color='blue')
    #plt.plot(e_sp, np.max(e_sg)*1.1*np.ones(e_sp.shape),color='b', marker='.', ms=2, fillstyle='full', linestyle='none')
    
    v_sg = v_sg[np.where(np.multiply(v_t>=scope[0], v_t<=scope[1]))[0]]
    v_sp = v_sp[np.where(np.multiply(v_sp>=scope[0], v_sp<=scope[1]))[0]]
    v_t = v_t[np.where(np.multiply(v_t>=scope[0], v_t<=scope[1]))[0]]
    #plt.plot(v_t, v_sg, label='ephys', color='blue')
    #plt.plot(v_sp, np.max(v_sg)*1.1*np.ones(v_sp.shape),color='b', marker='.', ms=2, fillstyle='full', linestyle='none')
    
    # Distance matrix and find matches
    D = distance_spikes(s1=e_sp, s2=v_sp, max_dist=0.05)
    index_gt, index_method = find_matches(D)
    
    height = np.max(np.array(e_sg.max(), v_sg.max()))
    plt.plot(e_t, e_sg, color='b', label='ephys')
    plt.plot(e_sp, 1.2*height*np.ones(e_sp.shape),color='b', marker='.', ms=2, fillstyle='full', linestyle='none')
    plt.plot(v_t, v_sg, color='orange', label='VolPy')
    plt.plot(v_sp, 1.4*height*np.ones(len(v_sp)),color='orange', marker='.', ms=2, fillstyle='full', linestyle='none')
    ax = plt.gca()
    ax.locator_params(nbins=7)
    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)
    
    for i in range(len(index_gt)):
        plt.plot((e_sp[index_gt[i]], v_sp[index_method[i]]),(1.25*height, 1.35*height), color='gray',alpha=0.5, linewidth=1)
    
    # Calculate measures
    TP = len(index_gt)
    FP = len(v_sp) - TP
    FN = len(e_sp) - TP
    precision = TP / (TP + FP)
    recall = TP / (TP + FN)
    F1 = 2 * (precision * recall) / (precision + recall) 
    print('precision:',precision)
    print('recall:',recall)
    print('F1:',F1)      
    plt.legend(prop={'size': 6})
    plt.tight_layout()
    
    return precision, recall, F1
    
#%%
precision = []
recall = []
F1 = []
for i in range(len(sweep_time)):
    scope = [sweep_time[i][0], sweep_time[i][-1]]
    plt.figure()
    pr, re, F = spike_comparison(e_sg, e_sp, e_t, v_sg, v_sp, v_t, scope)
    precision.append(pr)
    recall.append(re)
    F1.append(F)
    #plt.savefig(f'/home/nel/NEL-LAB Dropbox/NEL/Papers/VolPy/Marton/spike_sweep{i}_{vpy.params.volspike["threshold_method"]}.pdf')

#%%
plt.figure()    
plt.title(f'Method: {vpy.params.volspike["threshold_method"]}')
plt.plot(precision, label=f'precision:{round(np.array(precision).mean(),2)}')
plt.plot(recall, label=f'recall: {round(np.array(recall).mean(),2)}')
plt.plot(F1, label=f'F1: {round(np.array(F1).mean(),2)}')
plt.legend()
#plt.savefig(f'/home/nel/NEL-LAB Dropbox/NEL/Papers/VolPy/Marton/F1_score_{vpy.params.volspike["threshold_method"]}.pdf')
#plt.text(0.1, 0.1, f'precision:{round(np.array(precision).mean(),2)}', horizontalalignment='center', verticalalignment='center')
#plt.text(0.1, 0.05, f'recall: {round(np.array(recall).mean(),2)}', horizontalalignment='center', verticalalignment='center')            
#plt.text(0.1, 0.00, f'F1: {round(np.array(F1).mean(),2)}', horizontalalignment='center', verticalalignment='center')            


#%%
e_sg =  np.array(sweep_response).reshape(-1, order='C')
plt.plot(e_sg)
spikes = find_peaks(e_sg, hline, distance=10)[0]

window_length = 30
window = np.int64(np.arange(-window_length, window_length + 1, 1))
data = e_sg - np.percentile(e_sg,3)
locs = spikes.copy()
locs = locs[np.logical_and(locs > (-window[0]), locs < (len(data) - window[-1]))]

for i in locs:
    xp = [i+window[0]-1, i+window[-1]+1]
    fp = [data[i+window[0]-1], data[i+window[-1]+1]]
    x = list(range(i+window[0],i+window[-1]+1))
    data[x] = np.interp(x, xp, fp)


"""
window = np.int64(np.arange(-window_length, window_length + 1, 1))
locs = locs[np.logical_and(locs > (-window[0]), locs < (len(data) - window[-1]))]
PTD = data[(locs[:, np.newaxis] + window)]
PTA = np.mean(PTD, 0)
templates = PTA

t_rec = np.zeros(data.shape)
t_rec[spikes] = 1
t_rec = np.convolve(t_rec, PTA, 'same')   

plt.plot(data)
plt.plot(t_rec)
"""
#%%
from caiman.source_extraction.volpy.spikepursuit import signal_filter
t_sub = signal_filter(data,20, fr=10000, order=5, mode='low') 

#%%
plt.plot(e_sg - np.percentile(e_sg,3))
plt.plot(data)
plt.plot(t_sub)










