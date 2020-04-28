#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Apr 28 18:58:55 2020
files for loading and analyzing proccessed Marton's data
@author: caichangjia
"""
#%% import library
import matplotlib.pyplot as plt
import numpy as np
from scipy.optimize import linear_sum_assignment
from scipy.signal import find_peaks

#%% relevant functions
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

# Compute subthreshold correlation coefficents
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


#%%
file_list = ['/home/nel/NEL-LAB Dropbox/NEL/Papers/VolPy/Marton/data/454597_Cell_0_40x_patch1_output_simple.npz']
fig = plt.figure(figsize=(12,12))
temp = file_list[0].split('/')[-1].split('_')
fig.suptitle(f'subject_id:{temp[0]}  Cell number:{temp[2]}')

pr= []
re = []
F = []
sub = []
for file in file_list:
    dict1 = np.load(file, allow_pickle=True)
    precision, recall, F1, sub_corr, e_match, v_match, mean_time = metric(dict1['sweep_time'], dict1['e_sg'], 
                                                                          dict1['e_sp'], dict1['e_t'],dict1['e_sub'], 
                                                                          dict1['v_sg'], dict1['v_sp'], 
                                                                          dict1['v_t'], dict1['v_sub'],save=False)
    pr.append(np.array(precision).mean().round(2))
    re.append(np.array(recall).mean().round(2))
    F.append(np.array(F1).mean().round(2))
    sub.append(np.array(sub_corr).mean().round(2))
    ax1 = fig.add_axes([0.05, 0.8, 0.9, 0.15])
    
    e_fr = np.unique(np.floor(dict1['e_sp']), return_counts=True)
    v_fr = np.unique(np.floor(dict1['v_sp']), return_counts=True)
    ax1.plot(e_fr[0], e_fr[1], color='black')
    ax1.plot(v_fr[0], v_fr[1], color='g')
    ax1.legend(['ephys','voltage'])
    ax1.set_ylabel('Firing Rate (Hz)')
    
    ax2 = fig.add_axes([0.05, 0.6, 0.9, 0.15])
    ax2.vlines(list(set(dict1['v_sp'])-set(v_match)), 2.75,3.25, color='red')
    ax2.vlines(dict1['v_sp'], 1.75,2.25, color='green')
    ax2.vlines(dict1['e_sp'], 0.75,1.25, color='black')
    ax2.vlines(list(set(dict1['e_sp'])-set(e_match)), -0.25,0.25, color='red')
    plt.yticks(np.arange(4), ['False Negative', 'Ephys', 'Voltage', 'False Positive'])
    
    ax3 = fig.add_axes([0.05, 0.2, 0.9, 0.35])
    ax3.plot(mean_time, precision, 'o-', c='blue')
    ax3.plot(mean_time, recall, 'o-', c='orange')
    ax3.plot(mean_time, F1, 'o-', c='green')
    
    
    ax4 = fig.add_axes([0.05, 0, 0.9, 0.15])
    ax4.plot(mean_time, sub_corr, 'o-', c='blue')
    #plt.savefig(f'{volpy_path}/metric_{vpy.params.volspike["threshold_method"]}.pdf', bbox_inches='tight')
ax3.legend([f'precision:{pr}', f'recall: {re}', f'F1: {F}'])
ax4.legend([f'corr:{sub}'])
