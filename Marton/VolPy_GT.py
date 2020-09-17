#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Created on Fri Apr 17 17:08:11 2020
Files for analyzing Marton's GT data
@author: caichangjia
"""
import caiman as cm
import numpy as np
import os
import json
from scipy.optimize import linear_sum_assignment
from scipy.signal import find_peaks
import matplotlib.pyplot as plt

#%%
#movie_dir = '/home/nel/data/voltage_data/Marton/454597/Cell_0/40x_patch1/'
#movie_dir = '/home/nel/data/voltage_data/Marton/456462/Cell_3/40x_1xtube_10A2'
#movie_dir = '/home/nel/data/voltage_data/Marton/456462/Cell_3/40x_1xtube_10A3'
#movie_dir = '/home/nel/data/voltage_data/Marton/462149/Cell_1/40x_1xtube_10A1'
#movie_dir =  '/home/nel/data/voltage_data/Marton/462149/Cell_1/40x_1xtube_10A2'
#movie_dir = '/home/nel/data/voltage_data/Marton/456462/Cell_5/40x_1xtube_10A5'
#movie_dir = '/home/nel/data/voltage_data/Marton/456462/Cell_5/40x_1xtube_10A6'
#movie_dir = '/home/nel/data/voltage_data/Marton/456462/Cell_5/40x_1xtube_10A7'
#movie_dir = '/home/nel/data/voltage_data/Marton/456462/Cell_4/40x_1xtube_10A4'
#movie_dir = '/home/nel/data/voltage_data/Marton/456462/Cell_6/40x_1xtube_10A10'
#movie_dir = '/home/nel/data/voltage_data/Marton/456462/Cell_6/40x_1xtube_10A11'
#movie_dir = '/home/nel/data/voltage_data/Marton/466769/Cell_0/40x_1xtube_10A_1'
#movie_dir = '/home/nel/data/voltage_data/Marton/456462/Cell_5/40x_1xtube_10A8' # EPSP
#movie_dir = '/home/nel/data/voltage_data/Marton/456462/Cell_5/40x_1xtube_10A9' # EPSP
#movie_dir = '/home/nel/data/voltage_data/Marton/462149/Cell_3/40x_1xtube_10A3' # 1KHZ
#movie_dir = '/home/nel/data/voltage_data/Marton/462149/Cell_3/40x_1xtube_10A4' # 1KHZ
movie_dir = '/home/nel/data/voltage_data/Marton/466769/Cell_2/40x_1xtube_10A_6'
movie_dir = '/home/nel/data/voltage_data/Marton/466769/Cell_2/40x_1xtube_10A_4'
movie_dir = '/home/nel/data/voltage_data/Marton/466769/Cell_3/40x_1xtube_10A_8'
idx = 0
volpy_path = os.path.join(movie_dir, 'volpy')
if not os.path.isdir(volpy_path):
   os.makedirs(volpy_path)



#%%
np.save(os.path.join(os.path.dirname(os.path.dirname(fnames[0])), 'volpy')+'/estimates.npy', vpy.estimates)
#%% load voltage data
frame_times = np.load(os.path.join(movie_dir, 'frame_times.npy'))
ephys_files_dir = os.path.join(movie_dir,'ephys')
ephys_files = sorted(os.listdir(ephys_files_dir))

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

estimates = np.load(volpy_path+'/estimates.npy', allow_pickle=True).item()
v_sg = estimates['t'][idx]
v_sp = estimates['spikes'][idx]
#plt.plot(dff)
v_t = frame_times
v_sp = v_t[estimates['spikes'][idx]]

#for i in range(len(sweep_time) - 1):
#    v_sp = np.delete(v_sp, np.where([np.logical_and(v_sp>sweep_time[i][-1], v_sp<sweep_time[i+1][0])])[1])
#v_sp = np.delete(v_sp, np.where([v_sp>sweep_time[i+1][-1]])[1])

plt.plot(v_t, v_sg, label='ephys', color='blue')
plt.plot(v_sp, np.max(v_sg)*1.1*np.ones(v_sp.shape),color='b', marker='.', ms=2, fillstyle='full', linestyle='none')

#%% load ephys data
e_sg =  np.array(sweep_response).reshape(-1, order='C')
e_t = np.array(sweep_time).reshape(-1, order='C')
e_sg = np.delete(e_sg, np.where([np.logical_or(e_t<v_t[0], e_t>v_t[-1])])[1])
e_t = np.delete(e_t, np.where([np.logical_or(e_t<v_t[0], e_t>v_t[-1])])[1])
hline= 2/3 * e_sg.max() + 1/3 * e_sg.min()
e_sp = e_t[find_peaks(e_sg, hline, distance=10)[0]]
plt.figure()
plt.plot(e_t, e_sg, label='ephys', color='blue')
plt.hlines(hline, e_t[0], e_t[-1], linestyles='dashed', color='gray')
plt.plot(e_sp, np.max(e_sg)*1.1*np.ones(e_sp.shape),color='b', marker='.', ms=2, fillstyle='full', linestyle='none')

#%% subthreshold of ephys data
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

from caiman.source_extraction.volpy.spikepursuit import signal_filter
e_sub = signal_filter(data,20, fr=sweep_metadata[0]['sample_rate'], order=5, mode='low') 

v_sub = estimates['t_sub'][idx]
plt.plot(e_t, e_sg - np.percentile(e_sg,3))
plt.plot(e_t, data)
plt.plot(e_t, e_sub)
plt.plot(v_t, v_sub)

e_sub = np.interp(v_t, e_t, e_sub)
e_sub = e_sub - np.median(e_sub) - 20
v_sub = v_sub - np.median(v_sub)
plt.plot(v_t, e_sub)
plt.plot(v_t, v_sub)


#%% function for compare spikes
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

def spike_comparison(i, e_sg, e_sp, e_t, v_sg, v_sp, v_t, scope, max_dist, save=False):
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
    D = distance_spikes(s1=e_sp, s2=v_sp, max_dist=max_dist)
    index_gt, index_method = find_matches(D)
    match = [e_sp[index_gt], v_sp[index_method]]
    height = np.max(np.array(e_sg.max(), v_sg.max()))
    
    # Calculate measures
    TP = len(index_gt)
    FP = len(v_sp) - TP
    FN = len(e_sp) - TP
    try:    
        precision = TP / (TP + FP)
    except ZeroDivisionError:
        if v_sp == 0:
            precision = 1
        else:    
            precision = 0

    try:    
        recall = TP / (TP + FN)
    except ZeroDivisionError:
        recall = 1

    try:
        F1 = 2 * (precision * recall) / (precision + recall) 
    except ZeroDivisionError:
        F1 = 0

    print('precision:',precision)
    print('recall:',recall)
    print('F1:',F1)      
    if save:
        plt.figure()
        plt.plot(e_t, e_sg, color='b', label='ephys')
        plt.plot(e_sp, 1.2*height*np.ones(e_sp.shape),color='b', marker='.', ms=2, fillstyle='full', linestyle='none')
        plt.plot(v_t, v_sg, color='orange', label='VolPy')
        plt.plot(v_sp, 1.4*height*np.ones(len(v_sp)),color='orange', marker='.', ms=2, fillstyle='full', linestyle='none')
        for j in range(len(index_gt)):
            plt.plot((e_sp[index_gt[j]], v_sp[index_method[j]]),(1.25*height, 1.35*height), color='gray',alpha=0.5, linewidth=1)
        ax = plt.gca()
        ax.locator_params(nbins=7)
        ax.spines['right'].set_visible(False)
        ax.spines['top'].set_visible(False)
        plt.legend(prop={'size': 6})
        plt.tight_layout()
        plt.savefig(f'{volpy_path}/spike_sweep{i}_{vpy.params.volspike["threshold_method"]}.pdf')
    
    return precision, recall, F1, match

def sub_correlation(i, v_t, e_sub, v_sub, scope, save=False):
    e_sub = e_sub[np.where(np.multiply(v_t>=scope[0], v_t<=scope[1]))[0]]
    v_sub = v_sub[np.where(np.multiply(v_t>=scope[0], v_t<=scope[1]))[0]]
    v_t = v_t[np.where(np.multiply(v_t>=scope[0], v_t<=scope[1]))[0]]
    corr = np.corrcoef(e_sub, v_sub)[0,1]
    if save:
        plt.figure()
        plt.plot(v_t, e_sub)
        plt.plot(v_t, v_sub)   
        plt.savefig(f'{volpy_path}/spike_sweep{i}_subthreshold.pdf')
    return corr

def metric(sweep_time, e_sg, e_sp, e_t, e_sub, v_sg, v_sp, v_t, v_sub, save=False):
    precision = []
    recall = []
    F1 = []
    sub_corr = []
    mean_time = []
    e_match = []
    v_match = []
    
    for i in range(len(sweep_time)):
        print(f'sweep{i}')
        if i == 0:
            scope = [np.ceil(max([e_t.min(), v_t.min()])), sweep_time[i][-1]]
            m_t = 1 / 2 * (scope[0] + scope[-1])
        elif i == len(sweep_time) - 1:
            scope = [sweep_time[i][0], np.floor(min([e_t.max(), v_t.max()]))]
        else:
            scope = [sweep_time[i][0], sweep_time[i][-1]]
        mean_time.append(1 / 2 * (scope[0] + scope[-1]))
        
        pr, re, F, match = spike_comparison(i, e_sg, e_sp, e_t, v_sg, v_sp, v_t, scope, max_dist=0.05, save=save)
        corr = sub_correlation(i, v_t, e_sub, v_sub, scope, save=save)
        precision.append(pr)
        recall.append(re)
        F1.append(F)
        sub_corr.append(corr)
        e_match.append(match[0])
        v_match.append(match[1])
    
    e_match = np.concatenate(e_match)
    v_match = np.concatenate(v_match)

    return precision, recall, F1, sub_corr, e_match, v_match, mean_time
    
#%% save
temp = movie_dir.split('/')

save_name = f'{volpy_path}/{temp[-3]}_{temp[-2]}_{temp[-1]}_output.npz' 

np.savez(save_name, sweep_time=sweep_time, e_sg=e_sg, v_sg=v_sg, 
         e_t=e_t, v_t=v_t, e_sp=e_sp, v_sp=v_sp, e_sub=e_sub, v_sub=v_sub)


#%%
mm = np.array(m)
mm.shape
plt.imshow(m[0])
plt.imshow(mm[0,80:110, 180:280])
plt.imshow(mm[0,80:110, 250:280])
plt.imshow(mm[0,85:115, 255:285])
plt.imshow(mm[0,75:105, 255:285])
mmm = mm[:, 75:105, 255:285]
mmm.shape
mmm
save_name = f'{volpy_path}/{temp[-3]}_{temp[-2]}_{temp[-1]}.tif'
save_name
mmm.shape
cm.movie(mmm).save(save_name)

#%% firing rate
"""
file_list = ['/home/nel/data/voltage_data/Marton/456462/Cell_3/40x_1xtube_10A2/volpy/output_simple.npz',
             '/home/nel/data/voltage_data/Marton/456462/Cell_3/40x_1xtube_10A3/volpy/output_simple.npz']
file_list = [f'/home/nel/data/voltage_data/Marton/462149/Cell_1/40x_1xtube_10A1/volpy/output_{vpy.params.volspike["threshold_method"]}.npz', 
                                                                                              f'/home/nel/data/voltage_data/Marton/462149/Cell_1/40x_1xtube_10A2/volpy/output_{vpy.params.volspike["threshold_method"]}.npz']
file_list = ['/home/nel/data/voltage_data/Marton/456462/Cell_5/40x_1xtube_10A5/volpy/output_simple.npz',
             '/home/nel/data/voltage_data/Marton/456462/Cell_5/40x_1xtube_10A6/volpy/output_simple.npz',
             '/home/nel/data/voltage_data/Marton/456462/Cell_5/40x_1xtube_10A7/volpy/output_simple.npz']

file_list = ['/home/nel/data/voltage_data/Marton/454597/Cell_0/40x_patch1/volpy/Cell_0_40x_patch1__output_simple.npz']

file_list = ['/home/nel/data/voltage_data/Marton/456462/Cell_5/40x_1xtube_10A5/volpy/456462_Cell_5_40x_1xtube_10A5_output_simple.npz',
             '/home/nel/data/voltage_data/Marton/456462/Cell_5/40x_1xtube_10A6/volpy/456462_Cell_5_40x_1xtube_10A6_output_simple.npz',
             '/home/nel/data/voltage_data/Marton/456462/Cell_5/40x_1xtube_10A7/volpy/456462_Cell_5_40x_1xtube_10A7_output_simple.npz']
"""
#file_list = ['/home/nel/NEL-LAB Dropbox/NEL/Papers/VolPy/Marton/data/454597_Cell_0_40x_patch1_output_simple.npz']
"""
base_folder = os.path.dirname(os.path.dirname(file_list[0]))
base_name = os.path.basename(file_list[0]).split('_')[:-2]
base_name = [n+'_' for n in base_name]
base_name = ''.join(base_name)
"""

#file_list = ['/home/nel/data/voltage_data/Marton/456462/Cell_4/40x_1xtube_10A4/volpy/456462_Cell_4_40x_1xtube_10A4_output.npz']
file_list = [save_name]
fig = plt.figure(figsize=(12,12))
fig.suptitle(f'subject_id:{movie_dir.split("/")[-3]}  Cell number:{movie_dir.split("/")[-2]}')

pr= []
re = []
F = []
sub = []
for file in file_list:
    dict1 = np.load(file, allow_pickle=True)
    precision, recall, F1, sub_corr, e_match, v_match, mean_time = metric(dict1['sweep_time'], dict1['e_sg'], 
                                                                          dict1['e_sp'], dict1['e_t'], dict1['e_sub'],
                                                                          dict1['v_sg'], dict1['v_sp'], 
                                                                          dict1['v_t'], dict1['v_sub'], save=False)
    pr.append(np.array(precision).mean().round(2))
    re.append(np.array(recall).mean().round(2))
    F.append(np.array(F1).mean().round(2))
    sub.append(np.array(sub_corr).mean().round(2))
    ax1 = fig.add_axes([0.05, 0.8, 0.9, 0.15])
    #xlimits = [frame_times[0],frame_times[-1]]
    e_fr = np.unique(np.floor(dict1['e_sp']), return_counts=True)
    v_fr = np.unique(np.floor(dict1['v_sp']), return_counts=True)
    ax1.plot(e_fr[0], e_fr[1], color='black')
    ax1.plot(v_fr[0], v_fr[1], color='g')
    ax1.legend(['ephys','voltage'])
    ax1.set_ylabel('Firing Rate (Hz)')
    #ax1.set_xticklabels([])
    
    ax2 = fig.add_axes([0.05, 0.6, 0.9, 0.15])
    ax2.vlines(list(set(dict1['v_sp'])-set(v_match)), 2.75,3.25, color='red')
    ax2.vlines(v_sp, 1.75,2.25, color='green')
    ax2.vlines(dict1['e_sp'], 0.75,1.25, color='black')
    ax2.vlines(list(set(dict1['e_sp'])-set(e_match)), -0.25,0.25, color='red')
    plt.yticks(np.arange(4), ['False Negative', 'Ephys', 'Voltage', 'False Positive'])
    #ax2.set_xticklabels([])
    
    ax3 = fig.add_axes([0.05, 0.2, 0.9, 0.35])
    ax3.plot(mean_time, precision, 'o-', c='blue')
    ax3.plot(mean_time, recall, 'o-', c='orange')
    ax3.plot(mean_time, F1, 'o-', c='green')
    
    #ax3.set_xticklabels([])
    
    ax4 = fig.add_axes([0.05, 0, 0.9, 0.15])
    ax4.plot(mean_time, sub_corr, 'o-', c='blue')
    #plt.savefig(f'{volpy_path}/metric_{vpy.params.volspike["threshold_method"]}.pdf', bbox_inches='tight')
ax3.legend([f'precision:{pr}', f'recall: {re}', f'F1: {F}'])
ax4.legend([f'corr:{sub}'])
#plt.savefig(f'{base_folder}/output/{base_name}.pdf', bbox_inches='tight')

#%%
locs = signal.find_peaks(datafilt, height=thresh2)[0]



#%%
#F1_orig = [0.96, 0.94, 0.7, 0.28, 0.91, 0.72, 0.97, 0.79]
file_list = ['/home/nel/NEL-LAB Dropbox/NEL/Papers/VolPy/Marton/data_new/454597_Cell_0_40x_patch1_output.npz',
             '/home/nel/NEL-LAB Dropbox/NEL/Papers/VolPy/Marton/data_new/456462_Cell_3_40x_1xtube_10A2_output.npz',
             '/home/nel/NEL-LAB Dropbox/NEL/Papers/VolPy/Marton/data_new/456462_Cell_3_40x_1xtube_10A3_output.npz',
             '/home/nel/NEL-LAB Dropbox/NEL/Papers/VolPy/Marton/data_new/456462_Cell_5_40x_1xtube_10A5_output.npz',
             '/home/nel/NEL-LAB Dropbox/NEL/Papers/VolPy/Marton/data_new/456462_Cell_5_40x_1xtube_10A6_output.npz',
             '/home/nel/NEL-LAB Dropbox/NEL/Papers/VolPy/Marton/data_new/456462_Cell_5_40x_1xtube_10A7_output.npz',
             '/home/nel/NEL-LAB Dropbox/NEL/Papers/VolPy/Marton/data_new/462149_Cell_1_40x_1xtube_10A1_output.npz',
             '/home/nel/NEL-LAB Dropbox/NEL/Papers/VolPy/Marton/data_new/462149_Cell_1_40x_1xtube_10A2_output.npz']

pr= []
re = []
F = []
sub = []
update = False

for file in file_list:
    dict1 = np.load(file, allow_pickle=True)
    v_sp = dict1['v_sp']

    if update:
        datafilt, v_sp, t_rec, templates, _, thresh2 = denoise_spikes(dict1['v_sg'], window_length=8, fr=400, hp_freq=1, threshold_method='simple',
                      min_spikes=5, threshold=3.5, last_round=False, do_plot=False)
        v_sp = dict1['v_t'][v_sp]
        
        for i in range(len(dict1['sweep_time']) - 1):
            v_sp = np.delete(v_sp, np.where([np.logical_and(v_sp>dict1['sweep_time'][i][-1], v_sp<dict1['sweep_time'][i+1][0])])[1])
        v_sp = np.delete(v_sp, np.where([v_sp>dict1['sweep_time'][i+1][-1]])[1])

    precision, recall, F1, sub_corr, e_match, v_match, mean_time = metric(dict1['sweep_time'], dict1['e_sg'], 
                                                                          dict1['e_sp'], dict1['e_t'], dict1['e_sub'],
                                                                          dict1['v_sg'], v_sp, 
                                                                          dict1['v_t'], dict1['v_sub'], save=False)
    pr.append(np.array(precision).mean().round(2))
    re.append(np.array(recall).mean().round(2))
    F.append(np.array(F1).mean().round(2))
    sub.append(np.array(sub_corr).mean().round(2))


#%%
from caiman.source_extraction.volpy.spikepursuit import denoise_spikes
from caiman.source_extraction.volpy.spikepursuit import signal_filter


#%%
window = 20000
stride = 4000
T = len(v_sg)
n = int((T - window) / stride+1)
mv_mean = []
mv_std = []
mv_peak = []
v_sg = signal_filter(v_sg, 1, 400, order=5)
#v_sg = v_sg - np.median(v_sg)

for i in range(n):
    data = datafilt[stride * i: stride * i + window]
    ff1 = -data * (data < 0)
    Ns = np.sum(ff1 > 0)
    mv_std.append(np.sqrt(np.divide(np.sum(ff1**2), Ns)))
    spike = v_sp[np.where(np.logical_and(v_sp >= stride * i, v_sp < stride * i + window))[0]]
    mv_peak.append(v_sg[spike].mean())
    
plt.plot(mv_std/max(mv_std))
plt.plot(mv_peak/max(mv_peak))
plt.legend(['std', 'peak height'])
#%%
window = 10000
T = len(v_sg)
n = int(np.ceil(T / window))
mv_mean = []
mv_std = []
mv_peak = []
mv_diff = []
#v_sg = signal_filter(v_sg, 1, 400, order=5)
#v_sg = v_sg - np.median(v_sg)

for i in range(n):
    data = datafilt[i * window: (i + 1) * window]
    ff1 = -data * (data < 0)
    Ns = np.sum(ff1 > 0)
    mv_std.append(np.sqrt(np.divide(np.sum(ff1**2), Ns)))
    spike = v_sp[np.where(np.logical_and(v_sp >= window * i, v_sp < (i + 1) * window))[0]]
    mv_peak.append(datafilt[spike].mean())
    mv_diff.append(np.percentile(data,99) - np.percentile(data,1))
    
mv_std = mv_std / max(mv_std)
mv_diff = mv_diff / max(mv_diff)
mv_peak = mv_peak / max(mv_peak)
z = np.polyfit(list(range(n)), mv_diff, deg=3)
p = np.poly1d(z)
decay = p(list(range(n)))
decay = decay / max(decay)

plt.plot(mv_std)
plt.plot(mv_peak)
plt.plot(decay)
plt.plot(mv_diff)   

plt.legend(['std', 'peak height', 'decay'])

data = datafilt.copy()
ff1 = -data * (data < 0)
Ns = np.sum(ff1 > 0)
std = np.sqrt(np.divide(np.sum(ff1**2), Ns))

peaks = []
for i in range(n):
    data = datafilt[i * window: (i + 1) * window]
    thresh = 3.5 * std * decay[i]
    peak = signal.find_peaks(data, height=thresh)[0]
    peaks.append(peak + i * window)
    
peaks= np.concatenate(peaks)

v_sp = peaks
v_sp = dict1['v_t'][v_sp]

for i in range(len(dict1['sweep_time']) - 1):
    v_sp = np.delete(v_sp, np.where([np.logical_and(v_sp>dict1['sweep_time'][i][-1], v_sp<dict1['sweep_time'][i+1][0])])[1])
v_sp = np.delete(v_sp, np.where([v_sp>dict1['sweep_time'][i+1][-1]])[1])





#%%
from scipy import signal
data = dict1['v_sg']
data = data - np.median(data)
data = signal_filter(data, 1, 400, order=5)
#ff1 = -data * (data < 0)
#Ns = np.sum(ff1 > 0)
#std = np.sqrt(np.divide(np.sum(ff1**2), Ns)) 
#thresh = 3.5 * std
locs = signal.find_peaks(data, height=8)[0]
v_sp = v_t[locs]



#%%
plt.plot(dict1['v_t'], dict1['v_sg'])
plt.plot(v_sp, np.ones(v_sp.shape) * 1.1 * np.max(dict1['v_sg']), 'o', c='blue')
plt.plot(dict1['e_sp'], np.ones(dict1['e_sp'].shape) * 1.2 * np.max(dict1['v_sg']), 'o', c='red')
#plt.plot(dict1['v_t'], t_rec)
    
#%%
from caiman.source_extraction.volpy.utils import view_components
utils.view_components(estimates, img_corr, [idx], frame_times=frame_times, gt_times=e_sp)


#%% import ephys data


#%% visualize ephys and voltage data
dff = estimates['dFF'][idx]
%matplotlib inline
xlimits = [frame_times[0],frame_times[-1]]
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



#%%
ax1.set_axis_off()
ax1.imshow(x.reshape((-1, 1)), **barprops)

# a horizontal barcode
ax2 = fig.add_axes([0.3, 0.4, 0.6, 0.2])
ax2.set_axis_off()
ax2.imshow(x.reshape((1, -1)), **barprops)

plt.show()


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

#%% For one video
dff = estimates['dFF'][0]
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


plt.figure()    
plt.title(f'Method: {vpy.params.volspike["threshold_method"]}')
plt.plot(precision, label=f'precision:{round(np.array(precision).mean(),2)}')
plt.plot(recall, label=f'recall: {round(np.array(recall).mean(),2)}')
plt.plot(F1, label=f'F1: {round(np.array(F1).mean(),2)}')
plt.legend()
plt.savefig(f'{volpy_path}/F1_score_{vpy.params.volspike["threshold_method"]}.pdf')

plt.figure()    
plt.title(f'Subthreshold correlation')
plt.plot(sub_corr, label=f'corr:{round(np.array(sub_corr).mean(),2)}')
plt.legend()
plt.savefig(f'{volpy_path}/subthreshold_{vpy.params.volspike["threshold_method"]}.pdf')












