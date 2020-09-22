#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Sep  7 22:36:12 2020

@author: nel
"""

import numpy as np
import os
import sys
sys.path.append('/home/nel/Code/NEL_LAB/volpy/figures/figure3_performance')

import scipy.io
import shutil

import caiman as cm
from caiman.source_extraction.volpy.spikepursuit import signal_filter
from caiman.base.rois import nf_match_neurons_in_binary_masks
from demo_voltage_simulation import run_volpy
from caiman_on_voltage import run_caiman
from pca_ica import run_pca_ica
from utils import normalize, flip_movie, load_gt, extract_spikes
import matplotlib
matplotlib.rcParams['pdf.fonttype'] = 42
matplotlib.rcParams['ps.fonttype'] = 42
import matplotlib.pyplot as plt
from match_spikes import match_spikes_greedy, compute_F1
from scipy.signal import find_peaks
#from simulation_sgpmd import run_sgpmd_demixing

ROOT_FOLDER = '/home/nel/NEL-LAB Dropbox/NEL/Papers/VolPy/test_data/simulation'
SAVE_FOLDER = '/home/nel/NEL-LAB Dropbox/NEL/Papers/VolPy/Figures/plos/original_files/figure3/simulation'

#%% Generate simulation movie
plt.imshow(m[0]);plt.colorbar()
plt.imshow(m[0, 170:220, 75:125]); plt.colorbar()
mm = m[5000:10000, 170:220, 77:127]
plt.imshow(mm.mean(0)); plt.colorbar()
mm.play()
mm = mm.transpose([1,2,0])
scipy.io.savemat('/home/nel/NEL-LAB Dropbox/NEL/Papers/VolPy/test_data/simulation/sim3/sim3_raw.mat', 
                 {'data':mm, "sampleRate":400.0})

#%% Move file in folders
names = [f'sim4_{i}' for i in range(13, 16)]
for name in names:
    try:
        os.makedirs(os.path.join(ROOT_FOLDER, name))
        print('make folder')
    except:
        print('already exist')
    files = [file for file in os.listdir(ROOT_FOLDER) if '.mat' in file and name[4:] in file]
    for file in files:
        shutil.move(os.path.join(ROOT_FOLDER, file), os.path.join(ROOT_FOLDER, name, file))

#%% save in .hdf5
#names = [f'sim3_{i}' for i in range(10, 17)]
#names = [f'sim4_{i}' for i in range(1, 4)]
for name in names:
    fnames_mat = f'/home/nel/NEL-LAB Dropbox/NEL/Papers/VolPy/test_data/simulation/{name}/{name}.mat'
    m = scipy.io.loadmat(fnames_mat)
    m = cm.movie(m['dataAll'].transpose([2, 0, 1]))
    fnames = fnames_mat[:-4] + '.hdf5'
    m.save(fnames)
    
#%% 
for name in names:
    try:
        os.makedirs(os.path.join(ROOT_FOLDER, name, 'caiman'))
        os.makedirs(os.path.join(ROOT_FOLDER, name, 'volpy'))
        os.makedirs(os.path.join(ROOT_FOLDER, name, 'pca-ica'))
        os.makedirs(os.path.join(ROOT_FOLDER, name, 'sgpmd'))
        os.makedirs(os.path.join(ROOT_FOLDER, name, 'spikepursuit'))
        print('make folder')
    except:
        print('already exist')
    
for name in names:
    folder = os.path.join(ROOT_FOLDER, name)
    fnames = os.path.join(folder, f'{name}.hdf5')
    m  = cm.load(fnames)
    mm = flip_movie(m)
    mm.save(os.path.join(folder, 'caiman', name+'_flip.hdf5'))
    
#%%
for name in names:
    folder = os.path.join(ROOT_FOLDER, name)
    fnames = os.path.join(folder, f'{name}.hdf5')
    m  = cm.load(fnames)
    mm = m.transpose([1, 2, 0])    
    scipy.io.savemat(os.path.join(ROOT_FOLDER, name, 'spikepursuit', f'{name}.mat'), 
                 {'data':mm, "sampleRate":400.0})

    
#%% move to volpy folder
for name in names:
    folder = os.path.join(ROOT_FOLDER, name)
    volpy_folder = os.path.join(folder, 'volpy')
    files = os.listdir(folder)
    files = [file for file in files if '.hdf5' not in file and '.mat' not in file]
    files = [file for file in files if os.path.isfile(os.path.join(folder, file))]
    for file in files:
        shutil.move(os.path.join(folder, file), os.path.join(volpy_folder, file))
        
#%% 
for name in names:
    folder = os.path.join(ROOT_FOLDER, name)
    volpy_folder = os.path.join(folder, 'volpy')
    files = os.listdir(folder)
    files = [file for file in files if '.hdf5' in file and 'flip' not in file]
    files = [file for file in files if os.path.isfile(os.path.join(folder, file))]
    for file in files:
        shutil.copyfile(os.path.join(folder, file), os.path.join(volpy_folder, file))
        
#%%
for name in names:
    folder = os.path.join(ROOT_FOLDER, name)
    s_folder = os.path.join(ROOT_FOLDER, name, 'sgpmd')
    file = f'{name}.hdf5'
    m = cm.load(os.path.join(folder, file))
    m.save(os.path.join(s_folder, name+'.tif'))
    
#%%
for name in names:
    folder = os.path.join(ROOT_FOLDER, name)
    spatial, temporal, spikes = load_gt(folder)  
    #ROIs = spatial.transpose([1,2,0])
    ROIs = spatial.copy()
    
    volpy_folder = os.path.join(folder, 'volpy')
    np.save(os.path.join(volpy_folder, 'ROIs_gt'), ROIs)
    
    
    
#%% volpy params
#for ridge_bg in [0.5, 0.1, 0.01, 0.001, 0]:
    context_size = 35                             # number of pixels surrounding the ROI to censor from the background PCA
    flip_signal = True                            # Important!! Flip signal or not, True for Voltron indicator, False for others
    hp_freq_pb = 1 / 3                            # parameter for high-pass filter to remove photobleaching
    threshold_method = 'simple'                   # 'simple' or 'adaptive_threshold'
    min_spikes= 30                                # minimal spikes to be found
    threshold = 3.0                               # threshold for finding spikes, increase threshold to find less spikes
    do_plot = False                               # plot detail of spikes, template for the last iteration
    ridge_bg= 0.1                                 # ridge regression regularizer strength for background removement, larger value specifies stronger regularization 
    sub_freq = 20                                 # frequency for subthreshold extraction
    weight_update = 'NMF'                       # 'ridge' or 'NMF' for weight update
    n_iter = 2
    
    options={'context_size': context_size,
               'flip_signal': flip_signal,
               'hp_freq_pb': hp_freq_pb,
               'threshold_method': threshold_method,
               'min_spikes':min_spikes,
               'threshold': threshold,
               'do_plot':do_plot,
               'ridge_bg':ridge_bg,
               'sub_freq': sub_freq,
               'weight_update': weight_update,
               'n_iter': n_iter}

    #%% run volpy
    #names = [f'sim3_{i}' for i in range(10, 17)]
    for name in names:
        folder = os.path.join(ROOT_FOLDER, name)
        volpy_folder = os.path.join(folder, 'volpy')
        fnames = os.path.join(volpy_folder, f'{name}.hdf5')
        run_volpy(fnames, options=options, do_motion_correction=False, do_memory_mapping=False)
    
#%% run caiman
#names = [f'sim3_{i}' for i in range(10, 17)]
for name in names:
    folder = os.path.join(ROOT_FOLDER, name)
    fnames = os.path.join(folder, 'caiman', f'{name}_flip.hdf5')
    run_caiman(fnames)
    
#%% run pca-ica
names = [f'sim3_{i}' for i in range(10, 17)]
for name in names:
    folder = os.path.join(ROOT_FOLDER, name)
    fnames = os.path.join(folder, f'{name}.hdf5')
    run_pca_ica(fnames)
    
#%% run sgpmd need invivo environment
names = [f'sim3_{i}' for i in range(10, 17)]
for name in names:
    folder = os.path.join(ROOT_FOLDER, name)
    run_sgpmd_demixing(folder)


#%%
#for ridge_bg in [0.5, 0.1, 0.01, 0.001, 0]:
names = [f'sim4_{i}' for i in range(1, 16)]
#distance = [f'dist_{i}' for i in [5, 6, 7, 10, 15]]
distance = [f'dist_{i}' for i in [5, 6, 7, 10, 15]]

for idx, dist in enumerate(distance):  
    select = np.arange(idx * 3, (idx + 1) * 3)
    v_result_all = []
    #names = [f'sim3_{i}' for i in range(10, 17)]
    for name in np.array(names)[select]:
        folder = os.path.join(ROOT_FOLDER, name)
        spatial, temporal, spikes = load_gt(folder)
        summary_file = os.listdir(os.path.join(folder, 'volpy'))
        summary_file = [file for file in summary_file if 'summary' in file][0]
        summary = cm.load(os.path.join(folder, 'volpy', summary_file))
        
        v_folder = os.path.join(folder, 'volpy')
        v_files = sorted([file for file in os.listdir(v_folder) if f'simple_3.0' in file and f'bg_{ridge_bg}.npy' in file]) # 'simple_3.0' #'adaptive' in file and '0.1'
        #v_files = sorted([file for file in os.listdir(v_folder) if 'adaptive' in file])
        #v_files = sorted([file for file in os.listdir(v_folder) if 'NMF' in file])
        v_file = v_files[0]
        v = np.load(os.path.join(v_folder, v_file), allow_pickle=True).item()
        
        
        v_spatial = v['weights'].copy()
        v_temporal = v['t'].copy()
        v_ROIs = v['ROIs'].copy()
        v_ROIs = v_ROIs * 1.0
        v_templates = v['templates'].copy()
        v_spikes = v['spikes'].copy()
        
        #plt.figure(); plt.suptitle(f'distance:{dist}');plt.subplot(1,2,1);plt.imshow(v_spatial[0]); plt.subplot(1,2,2);plt.imshow(v_spatial[1]);
        #plt.figure(); plt.suptitle(f'distance:{dist}');plt.subplot(1,2,1);plt.imshow(v_spatial[0]); plt.subplot(1,2,2);plt.imshow(v_spatial[1]);
        
#%%        
        tp_gt, tp_comp, fn_gt, fp_comp, performance_cons_off = nf_match_neurons_in_binary_masks(
                spatial, v_ROIs, thresh_cost=1, min_dist=10, print_assignment=True,
                plot_results=True, Cn=summary[2], labels=['gt', 'volpy'])   
        
        plt.savefig(os.path.join(SAVE_FOLDER, 'simulation_Mask_RCNN_result_spikeamp_0.1_corr.pdf'))
        
        v_temporal = v_temporal[tp_comp]
        v_templates = v_templates[tp_comp]
        v_spikes = v_spikes[tp_comp]
        v_spatial = v_spatial[tp_comp]
        
        
        n_cells = len(tp_comp)
        v_result = {'F1':[], 'precision':[], 'recall':[]}
        
        for idx in range(n_cells):
            s1 = spikes[idx].flatten()
            s2 = v_spikes[idx]
            idx1_greedy, idx2_greedy = match_spikes_greedy(s1, s2, max_dist=3)
            F1, precision, recall = compute_F1(s1, s2, idx1_greedy, idx2_greedy)
            v_result['F1'].append(F1)
            v_result['precision'].append(precision)
            v_result['recall'].append(recall)
            
        v_result_all.append(v_result)
        if len(tp_comp) < 10:
            print(f'missing {10-tp_comp} neurons')
        print(f"volpy average 10 neurons:{np.array(v_result['F1']).sum()/10}")    
        
        #np.save(os.path.join(ROOT_FOLDER, 'result', 'volpy_threshold', f'volpy_adaptive_thresh'), v_result_all)
        #np.save(os.path.join(ROOT_FOLDER, 'result', 'volpy_threshold', f'volpy_thresh_{threshold}'), v_result_all)
        #np.save(os.path.join(ROOT_FOLDER, 'result', 'volpy_ridge_bg', f'volpy_bg_{ridge_bg}'), v_result_all)
    np.save(os.path.join(ROOT_FOLDER, 'result_overlap', dist, f'volpy_{dist}'), v_result_all)
    
    #np.save(os.path.join(folder, 'vpy_F1.npy'), v_result)
    #np.save(os.path.join(ROOT_FOLDER, 'result', 'thresh_3.0', f'volpy_thresh_{np.round(threshold, 2)}'), v_result_all)
    #np.save(os.path.join(ROOT_FOLDER, 'result', 'thresh_3.0', f'volpy_thresh_adaptive'), v_result_all)


#%%
#for threshold in np.arange(2.5, 4.1, 0.1):
#    threshold = np.round(threshold, 2)
names = [f'sim4_{i}' for i in range(1, 16)]
distance = [f'dist_{i}' for i in [5, 6, 7, 10, 15]]

for idx, dist in enumerate(distance):  
    select = np.arange(idx * 3, (idx + 1) * 3)
    c_result_all = []
    #names = [f'sim3_{i}' for i in range(10, 17)]
    for name in np.array(names)[select]:
        folder = os.path.join(ROOT_FOLDER, name)
        spatial, temporal, spikes = load_gt(folder)
        summary_file = os.listdir(os.path.join(folder, 'volpy'))
        summary_file = [file for file in summary_file if 'summary' in file][0]
        summary = cm.load(os.path.join(folder, 'volpy', summary_file))
    
    #%%
        c_folder = os.path.join(folder, 'caiman')
        caiman_files = [file for file in os.listdir(c_folder) if 'caiman_sim' in file][0]
        c = np.load(os.path.join(c_folder, caiman_files), allow_pickle=True).item()
        c_spatial = c.A.toarray().copy()
        c_spatial = c_spatial.reshape([50, 50, c_spatial.shape[1]], order='F').transpose([2, 0, 1])
        c_spatial_p = c_spatial.copy()
        for idx in range(len(c_spatial_p)):
            c_spatial_p[idx][c_spatial_p[idx] < c_spatial_p[idx].max() * 0.15] = 0
            c_spatial_p[idx][c_spatial_p[idx] >= c_spatial_p[idx].max() * 0.15] = 1
        c_temporal = c.C.copy()    
       
        #%%
        plt.figure()
        tp_gt, tp_comp, fn_gt, fp_comp, performance_cons_off = nf_match_neurons_in_binary_masks(
                spatial, c_spatial_p, thresh_cost=0.95, min_dist=10, print_assignment=True,
                plot_results=True, Cn=summary[0], labels=['gt', 'caiman'])    
        
        #%%
        c_temporal = c_temporal[tp_comp]
        c_spatial = c_spatial[tp_comp]   
        plt.figure(); plt.suptitle(f'distance:{dist}');plt.subplot(1,2,1);plt.imshow(c_spatial[0]); plt.subplot(1,2,2);plt.imshow(c_spatial[1]);
        
        #c_temporal_p = c_temporal_p[tp_comp]
        
        #%%
        idx=0
        plt.plot(c_temporal[idx])
        plt.plot(signal_filter(c_temporal, freq=15, fr=400)[idx])
        c_temporal_p = signal_filter(c_temporal, freq=15, fr=400)
        c_spikes = extract_spikes(c_temporal_p, threshold=3.0)
        
        #%%    
        c_result = {'F1':[], 'precision':[], 'recall':[]}
        n_cells = len(tp_comp)
        for idx in range(n_cells):
            s1 = spikes[idx].flatten()
            s2 = c_spikes[idx]
            idx1_greedy, idx2_greedy = match_spikes_greedy(s1, s2, max_dist=3)
            F1, precision, recall = compute_F1(s1, s2, idx1_greedy, idx2_greedy)
            c_result['F1'].append(F1)
            c_result['precision'].append(precision)
            c_result['recall'].append(recall)
            
        if len(tp_comp) < 10:
            print(f'missing {10-tp_comp} neurons')
        print(f"caiman average 10 neurons:{np.array(c_result['F1']).sum()/10}")
        
        c_result_all.append(c_result)
        #np.save(os.path.join(ROOT_FOLDER, 'result', 'caiman_threshold', f'caiman_thresh_{np.round(threshold, 2)}'), c_result_all)
    
        #np.save(os.path.join(folder, 'caiman_F1.npy'), c_result)
        #np.save(os.path.join(ROOT_FOLDER, 'result', 'thresh_3.0', f'caiman_thresh_{np.round(threshold, 2)}'), c_result_all)
    np.save(os.path.join(ROOT_FOLDER, 'result_overlap', dist, f'caiman_{dist}'), c_result_all)

#%%
m_result_all = []
names = [f'sim3_{i}' for i in range(10, 17)]
for name in names:
    folder = os.path.join(ROOT_FOLDER, name)
    spatial, temporal, spikes = load_gt(folder)    

    movie_file = [file for file in os.listdir(folder) if 'hdf5' in file and 'flip' not in file][0]
    mov = cm.load(os.path.join(folder, movie_file))
    mov = mov.reshape([mov.shape[0], -1], order='F')
    spatial_F = [np.where(sp.reshape(-1, order='F')>0) for sp in spatial]
    m_temporal = np.array([-mov[:, sp].mean((1,2)) for sp in spatial_F])
    m_temporal_p = signal_filter(m_temporal, freq=15, fr=400)
    m_spikes = extract_spikes(m_temporal_p, threshold=3.0)
    
#%%
    idx = 2
    plt.plot(m_temporal[idx,:1000])
    spk = spikes[idx][spikes[idx]<1000]
    plt.vlines(spk, ymin=m_temporal[idx,:1000].max(), ymax=m_temporal[idx,:1000].max()+10 )
    plt.savefig(os.path.join(SAVE_FOLDER, 'eg_trace_amplitude_0.15.pdf'))
    
#%%    
        
    m_result = {'F1':[], 'precision':[], 'recall':[]}
    n_cells = len(tp_comp)
    for idx in range(n_cells):
        s1 = spikes[idx].flatten()
        s2 = m_spikes[idx]
        idx1_greedy, idx2_greedy = match_spikes_greedy(s1, s2, max_dist=3)
        F1, precision, recall = compute_F1(s1, s2, idx1_greedy, idx2_greedy)
        m_result['F1'].append(F1)
        m_result['precision'].append(precision)
        m_result['recall'].append(recall)
        
    if len(tp_comp) < 10:
        print(f'missing {10-tp_comp} neurons')
    print(f"mean roi average 10 neurons:{np.array(m_result['F1']).sum()/10}")
    
    m_result_all.append(m_result)
    
    np.save(os.path.join(ROOT_FOLDER, 'result', 'thresh_3.0', f'mean_roi_thresh_{np.round(threshold, 2)}'), m_result_all)

    
#%%
p_result_all = []
names = [f'sim3_{i}' for i in range(10, 17)]
for name in names:
    folder = os.path.join(ROOT_FOLDER, name)
    p_folder = os.path.join(ROOT_FOLDER, name, 'pca-ica')
    
    movie_file = [file for file in os.listdir(folder) if 'hdf5' in file and 'flip' not in file][0]
    mov = cm.load(os.path.join(folder, movie_file))
    spatial, temporal, spikes = load_gt(folder)    

    file = os.listdir(p_folder)[0]
    p = np.load(os.path.join(p_folder, file), allow_pickle=True).item()

    p_spatial = p['spatial'].copy()   
    p_temporal = p['temporal'].copy().T
    
    p_spatial_p = p_spatial.copy()
    for idx in range(len(p_spatial_p)):
        p_spatial_p[idx][p_spatial_p[idx] < p_spatial_p[idx].max() * 0.15] = 0
        p_spatial_p[idx][p_spatial_p[idx] >= p_spatial_p[idx].max() * 0.15] = 1
    
    tp_gt, tp_comp, fn_gt, fp_comp, performance_cons_off = nf_match_neurons_in_binary_masks(
        spatial, p_spatial_p, thresh_cost=0.95, min_dist=10, print_assignment=True,
        plot_results=True, Cn=mov[0], labels=['gt', 'pca_ica'])    

    p_temporal = p_temporal[tp_comp]
    p_spatial = p_spatial[tp_comp]   

    p_spikes = extract_spikes(p_temporal, threshold=3.0)
        
    p_result = {'F1':[], 'precision':[], 'recall':[]}
    n_cells = len(tp_comp)
    for idx in range(n_cells):
        s1 = spikes[idx].flatten()
        s2 = p_spikes[idx]
        idx1_greedy, idx2_greedy = match_spikes_greedy(s1, s2, max_dist=3)
        F1, precision, recall = compute_F1(s1, s2, idx1_greedy, idx2_greedy)
        p_result['F1'].append(F1)
        p_result['precision'].append(precision)
        p_result['recall'].append(recall)

    if len(tp_comp) < 10:
        print(f'missing {10-tp_comp} neurons')
    print(f"pca-ica average 10 neurons:{np.array(p_result['F1']).sum()/10}")

    p_result_all.append(p_result)
    
    np.save(os.path.join(ROOT_FOLDER, 'result', 'thresh_3.0', f'pca_ica_thresh_{np.round(threshold, 2)}'), p_result_all)



#%%
#for threshold in np.arange(2.5, 4.1, 0.1):
#    print(threshold)
    
s_result_all = []
names = [f'sim3_{i}' for i in range(10, 17)]
for name in names:
    folder = os.path.join(ROOT_FOLDER, name)
    s_folder = os.path.join(folder, 'sgpmd', 'output')
    s_spatial = cm.load(os.path.join(s_folder, 'cell_spatial_footprints.tif'))
    s_spatial = s_spatial.reshape([48, 48, -1], order='F').transpose([2, 0, 1])
    s_spatial1 = np.zeros((s_spatial.shape[0], 50, 50))
    s_spatial1[:, 1:49, 1:49] = s_spatial    
    s_spatial = s_spatial1.copy()
    
    s_spatial_p = s_spatial.copy()
    for idx in range(len(s_spatial_p)):
        s_spatial_p[idx][s_spatial_p[idx] < s_spatial_p[idx].max() * 0.15] = 0
        s_spatial_p[idx][s_spatial_p[idx] >= s_spatial_p[idx].max() * 0.15] = 1
    s_temporal = cm.load(os.path.join(s_folder, 'cell_traces.tif'))
    
    # load gt and mov
    spatial, temporal, spikes = load_gt(folder)
    movie_file = [file for file in os.listdir(folder) if 'hdf5' in file and 'flip' not in file][0]
    mov = cm.load(os.path.join(folder, movie_file))
    
    #
    tp_gt, tp_comp, fn_gt, fp_comp, performance_cons_off = nf_match_neurons_in_binary_masks(
            spatial, s_spatial_p, thresh_cost=0.7, min_dist=10, print_assignment=True,
            plot_results=True, Cn=mov[0], labels=['gt', 'sgpmd'])    
    
    #
    s_temporal = s_temporal[tp_comp]
    s_spatial = s_spatial[tp_comp]   
    #s_temporal_p = s_temporal_p[tp_comp]
    
    #
    threshold= 3.0
    s_temporal_p = signal_filter(s_temporal, freq=15, fr=400)
    s_spikes = extract_spikes(s_temporal_p, threshold=threshold)
    
    #    
    s_result = {'F1':[], 'precision':[], 'recall':[]}
    n_cells = len(tp_comp)
    for idx in range(n_cells):
        s1 = spikes[idx].flatten()
        s1 = np.delete(s1, np.where([s1<100])[0]) 
        s2 = s_spikes[idx] + 100
        idx1_greedy, idx2_greedy = match_spikes_greedy(s1, s2, max_dist=3)
        F1, precision, recall = compute_F1(s1, s2, idx1_greedy, idx2_greedy)
        s_result['F1'].append(F1)
        s_result['precision'].append(precision)
        s_result['recall'].append(recall)
    
    if len(tp_comp) < 10:
        print(f'missing {10-tp_comp} neurons')
    print(f"sgpmd average 10 neurons:{np.array(s_result['F1']).sum()/10}")
    
    
    s_result_all.append(s_result)

    np.save(os.path.join(ROOT_FOLDER, 'result', 'thresh_3.0', f'sgpmd_thresh_{np.round(threshold, 2)}'), s_result_all)
    
#%% spikepursuit result
sp_result_all = []
names = [f'sim3_{i}' for i in range(10, 17)]
for name in names:
    folder = os.path.join(ROOT_FOLDER, name)
    sp_folder = os.path.join(folder, 'spikepursuit')
    output = scipy.io.loadmat(os.path.join(sp_folder, f'spikepursuit_{name}.mat'))['output'][0]
    sp_spikes = []
    for i in range(10):
        sp_spikes.append(output[i]['spikeTimes'][0][0].flatten().astype(np.int16))

    # load gt and mov
    spatial, temporal, spikes = load_gt(folder)
    movie_file = [file for file in os.listdir(folder) if 'hdf5' in file and 'flip' not in file][0]
    mov = cm.load(os.path.join(folder, movie_file))
        
    sp_result = {'F1':[], 'precision':[], 'recall':[]}
    n_cells = len(tp_comp)
    for idx in range(n_cells):
        s1 = spikes[idx].flatten()
        s2 = sp_spikes[idx]
        idx1_greedy, idx2_greedy = match_spikes_greedy(s1, s2, max_dist=3)
        F1, precision, recall = compute_F1(s1, s2, idx1_greedy, idx2_greedy)
        sp_result['F1'].append(F1)
        sp_result['precision'].append(precision)
        sp_result['recall'].append(recall)
    
    if len(tp_comp) < 10:
        print(f'missing {10-tp_comp} neurons')
    print(f"spikepursuit average 10 neurons:{np.array(sp_result['F1']).sum()/10}")
    
    
    sp_result_all.append(sp_result)

    np.save(os.path.join(ROOT_FOLDER, 'result', 'thresh_3.0', f'spikepursuit'), sp_result_all)

#%%
plt.plot(normalize(s_temporal[0])); plt.plot(normalize(s_temporal_p[0]));


#%%
x = [round(0.05 + 0.025 * i, 3) for i in range(7)]    
result_all = [v_result_all, c_result_all, m_result_all, p_result_all, s_result_all]
for results in result_all:
    #plt.plot(x, [np.array(result['F1']).sum()/10 for result in results])
    plt.errorbar(x, [np.array(result['F1']).sum()/10 for result in results], 
                     [np.std(np.array(result['F1'])) for result in results], 
                     solid_capstyle='projecting', capsize=3)
    
    
    plt.legend(['volpy', 'caiman', 'mean_roi', 'pca_ica', 'sgpmd'])
    plt.xlabel('spike amplitude')
    plt.ylabel('F1 score')
    plt.title('F1 score for different methods with error bar')
    
#%%
x = [round(0.05 + 0.025 * i, 3) for i in range(7)]    
result_all = [v_result_all, c_result_all, m_result_all, p_result_all, s_result_all]
colors = ['blue', 'orange', 'green', 'red', 'purple']
for idx, results in enumerate(result_all):
    plt.plot(x, [np.array(result['F1']).sum()/10 for result in results], color=colors[idx])
    plt.plot(x, [np.array(result['precision']).sum()/10 for result in results], color=colors[idx], alpha=0.7, linestyle=':')
    plt.plot(x, [np.array(result['recall']).sum()/10 for result in results], color=colors[idx], alpha=0.7, linestyle='-.')
    
    li = ['volpy', 'caiman', 'mean_roi', 'pca_ica', 'sgpmd']
    li = sum([[i, i+'_pre', i+'_rec'] for i in li], [])
    plt.legend(li)
    plt.xlabel('spike amplitude')
    plt.ylabel('F1/precision/recall')
    plt.title('F1/precision/recall of different methods')
    
#%% new F1 score
x = [round(0.05 + 0.025 * i, 3) for i in range(7)]    
result_folder = '/home/nel/NEL-LAB Dropbox/NEL/Papers/VolPy/test_data/simulation/result/thresh_3.0'
files = np.array(sorted(os.listdir(result_folder)))[np.array([5, 0, 1, 2, 3, 4, 6])]
result_all = [np.load(os.path.join(result_folder, file), allow_pickle=True) for file in files]
for results in result_all:
    plt.plot(x, [np.array(result['F1']).sum()/10 for result in results], marker='.')
    #plt.errorbar(x, [np.array(result['F1']).sum()/10 for result in results], 
    #                 [np.std(np.array(result['F1'])) for result in results], 
    #                 solid_capstyle='projecting', capsize=3)
    
    
    plt.legend([file for file in files])
    plt.xlabel('spike amplitude')
    plt.ylabel('F1 score')
    plt.title('F1 score of all methods')

plt.savefig(os.path.join(SAVE_FOLDER, 'VolPy_F1_scores_comparison.pdf'))

    
#%% F1 score for ridge bg
x = [round(0.05 + 0.025 * i, 3) for i in range(7)]    
result_folder = '/home/nel/NEL-LAB Dropbox/NEL/Papers/VolPy/test_data/simulation/result/volpy_ridge_bg'
files = np.array(os.listdir(result_folder))[np.array([1, 2,0,3,  4])]
result_all = [np.load(os.path.join(result_folder, file), allow_pickle=True) for file in files]
for results in result_all:
    plt.plot(x, [np.array(result['F1']).sum()/10 for result in results], marker='.', linewidth=1)
    #plt.errorbar(x, [np.array(result['F1']).sum()/10 for result in results], 
    #                 [np.std(np.array(result['F1'])) for result in results], 
    #                 solid_capstyle='projecting', capsize=3)
    
    
    plt.legend([file for file in files])
    plt.xlabel('spike amplitude')
    plt.ylabel('F1 score')
    plt.title('F1 score at different bg')

plt.savefig(os.path.join(SAVE_FOLDER, 'VolPy_ridge_vs_linear_regression.pdf'))


#%% F1 with precision/recall
x = [round(0.05 + 0.025 * i, 3) for i in range(7)]    
result_folder = '/home/nel/NEL-LAB Dropbox/NEL/Papers/VolPy/test_data/simulation/result/thresh_3.0'
files = np.array(sorted(os.listdir(result_folder)))[np.array([4,0,3,2,1,-1])]
result_all = [np.load(os.path.join(result_folder, file), allow_pickle=True) for file in files]
colors = ['blue', 'orange', 'green', 'red', 'purple', 'black']
for idx, results in enumerate(result_all):
    plt.plot(x, [np.array(result['F1']).sum()/10 for result in results], color=colors[idx])
    plt.plot(x, [np.array(result['precision']).sum()/10 for result in results], color=colors[idx], alpha=0.7, linestyle=':')
    plt.plot(x, [np.array(result['recall']).sum()/10 for result in results], color=colors[idx], alpha=0.7, linestyle='-.')
    
    li = ['volpy_3.0', 'caiman', 'mean_roi', 'pca_ica', 'sgpmd', 'volpy_adaptive']
    li = sum([[i, i+'_pre', i+'_rec'] for i in li], [])
    plt.legend(li)
    plt.xlabel('spike amplitude')
    plt.ylabel('F1/precision/recall')
    plt.title('F1/precision/recall of different methods')

#%% Overlapping neurons for volpy with different overlapping areas
names = [f'sim4_{i}' for i in range(1, 16)]
distance = [f'dist_{i}' for i in [5, 6, 7, 10, 15]]
area = []


for idx, dist in enumerate(distance):  
    select = np.arange(idx * 3, (idx + 1) * 3)
    v_result_all = []
    #names = [f'sim3_{i}' for i in range(10, 17)]
    for name in np.array(names)[select]:
        folder = os.path.join(ROOT_FOLDER, name)
        spatial, temporal, spikes = load_gt(folder)
        percent = (np.where(spatial.sum(axis=0) > 1)[0].shape[0])/(np.where(spatial.sum(axis=0) > 0)[0].shape[0])
        area.append(percent)   
        
area = np.round(np.unique(area), 2)[::-1]

#%%
result_all = {}
distance = [f'dist_{i}' for i in [5, 6, 7, 10, 15]]
x = [round(0.075 + 0.05 * i, 3) for i in range(3)] 
for dist in distance:   
    result_folder = f'/home/nel/NEL-LAB Dropbox/NEL/Papers/VolPy/test_data/simulation/result_overlap/{dist}'
    files = np.array(sorted(os.listdir(result_folder)))#[0]#[np.array([5, 0, 1, 2, 3, 4, 6])]
    files = [file for file in files if 'volpy' in file]
    result_all[files[0]] = np.load(os.path.join(result_folder, files[0]), allow_pickle=True)
    #result_all[files[1]] = np.load(os.path.join(result_folder, files[1]), allow_pickle=True)
    
    
for idx, key in enumerate(result_all.keys()):
    results = result_all[key]
    plt.plot(x, [np.array(result['F1']).sum()/2 for result in results], marker='.', label=f'{area[idx]:.0%}')

    plt.legend()
    plt.xlabel('spike amplitude')
    plt.ylabel('F1 score')
    plt.title('VolPy F1 score with different overlapping areas')
    
plt.savefig(os.path.join(SAVE_FOLDER, 'VolPy_overlapping_neurons.pdf'))
    
#%% VolPy vs CaImAn
x = [round(0.075 + 0.05 * i, 3) for i in range(3)] 
result_folder = '/home/nel/NEL-LAB Dropbox/NEL/Papers/VolPy/test_data/simulation/result_overlap/dist_5'
files = np.array(sorted(os.listdir(result_folder)))[np.array([1,0])]
result_all = [np.load(os.path.join(result_folder, file), allow_pickle=True) for file in files]
for results in result_all:
    plt.plot(x, [np.array(result['F1']).sum()/2 for result in results])
    #plt.errorbar(x, [np.array(result['F1']).sum()/10 for result in results], 
    #                 [np.std(np.array(result['F1'])) for result in results], 
    #                 solid_capstyle='projecting', capsize=3)
    
    plt.legend([file for file in files])
    plt.xlabel('spike amplitude')
    plt.ylabel('F1 score')
    plt.title('F1 score of all methods')   
    
#%%
    
    
#%% sgpmd
result_folder = '/home/nel/NEL-LAB Dropbox/NEL/Papers/VolPy/test_data/simulation/result/sgpmd_threshold'
files = sorted(os.listdir(result_folder))
files = np.array(files)[np.array([0, 3, 5, 10, 15])]
x = [round(0.05 + 0.025 * i, 3) for i in range(7)]    
result_all = [np.load(os.path.join(result_folder, file), allow_pickle=True) for file in files]


colors = ['blue', 'orange', 'green', 'red', 'purple']
for idx, results in enumerate(result_all):
    plt.plot(x, [np.array(result['F1']).sum()/10 for result in results], color=colors[idx%5])
    #plt.plot(x, [np.array(result['precision']).sum()/10 for result in results], color=colors[idx%5], alpha=0.7, linestyle=':')
    #plt.plot(x, [np.array(result['recall']).sum()/10 for result in results], color=colors[idx%5], alpha=0.7, linestyle='-.')
    
    li = [file for file in files]
    #li = sum([[i, i+'_pre', i+'_rec'] for i in li], [])
    plt.legend(li)
    plt.xlabel('spike amplitude')
    plt.ylabel('F1/precision/recall')
    plt.title('sgpmd F1 with different threshold')
    
#%% volpy
result_folder = '/home/nel/NEL-LAB Dropbox/NEL/Papers/VolPy/test_data/simulation/result/volpy_threshold'
files = sorted(os.listdir(result_folder))
files = np.array(files)[np.array([0, 1, 4, 6, 11])]
x = [round(0.05 + 0.025 * i, 3) for i in range(7)]    
result_all = [np.load(os.path.join(result_folder, file), allow_pickle=True) for file in files]


colors = ['blue', 'orange', 'green', 'red', 'purple']
for idx, results in enumerate(result_all):
    plt.plot(x, [np.array(result['F1']).sum()/10 for result in results], color=colors[idx%5])
    #plt.plot(x, [np.array(result['precision']).sum()/10 for result in results], color=colors[idx%5], alpha=0.7, linestyle=':')
    #plt.plot(x, [np.array(result['recall']).sum()/10 for result in results], color=colors[idx%5], alpha=0.7, linestyle='-.')
    
    li = [file for file in files]
    #li = sum([[i, i+'_pre', i+'_rec'] for i in li], [])
    plt.legend(li)
    plt.xlabel('spike amplitude')
    plt.ylabel('F1/precision/recall')
    plt.title('VolPy F1 with different threshold')
    
#%% caiman
result_folder = '/home/nel/NEL-LAB Dropbox/NEL/Papers/VolPy/test_data/simulation/result/caiman_threshold'
files = sorted(os.listdir(result_folder))
files = np.array(files)[np.array([0, 3, 5, 10, 15])]
x = [round(0.05 + 0.025 * i, 3) for i in range(7)]    
result_all = [np.load(os.path.join(result_folder, file), allow_pickle=True) for file in files]

colors = ['blue', 'orange', 'green', 'red', 'purple']
for idx, results in enumerate(result_all):
    plt.plot(x, [np.array(result['F1']).sum()/10 for result in results], color=colors[idx%5])
    #plt.plot(x, [np.array(result['precision']).sum()/10 for result in results], color=colors[idx%5], alpha=0.7, linestyle=':')
    #plt.plot(x, [np.array(result['recall']).sum()/10 for result in results], color=colors[idx%5], alpha=0.7, linestyle='-.')
    
    li = [file for file in files]
    #li = sum([[i, i+'_pre', i+'_rec'] for i in li], [])
    plt.legend(li)
    plt.xlabel('spike amplitude')
    plt.ylabel('F1/precision/recall')
    plt.title('CaImAn F1 with different threshold')
    
    
#%%
result_folder = '/home/nel/NEL-LAB Dropbox/NEL/Papers/VolPy/test_data/simulation/result/caiman_threshold'
files = sorted(os.listdir(result_folder))
files = np.array(files)[np.array([0, 3, 5, 10, 15])]
x = [round(0.05 + 0.025 * i, 3) for i in range(7)]    
result_all = [np.load(os.path.join(result_folder, file), allow_pickle=True) for file in files]

colors = ['blue', 'orange', 'green', 'red', 'purple']
for idx, results in enumerate(result_all):
    plt.plot(x, [np.array(result['F1']).sum()/10 for result in results], color=colors[idx%5])
    #plt.plot(x, [np.array(result['precision']).sum()/10 for result in results], color=colors[idx%5], alpha=0.7, linestyle=':')
    #plt.plot(x, [np.array(result['recall']).sum()/10 for result in results], color=colors[idx%5], alpha=0.7, linestyle='-.')
    
    li = [file for file in files]
    #li = sum([[i, i+'_pre', i+'_rec'] for i in li], [])
    plt.legend(li)
    plt.xlabel('spike amplitude')
    plt.ylabel('F1/precision/recall')
    plt.title('CaImAn F1 with different threshold')


#%%
F1_average = []    
for idx, results in enumerate(result_all):
    F1_average.append(np.array([np.array(result['F1']).sum()/10 for result in results]).mean())
    
 
#%%    plt.plot(x, [np.array(result['F1']).sum()/10 for result in v_result_all])
    #plt.plot(x, [np.array(result['F1']).mean() for result in v1])
    #plt.plot(x, [np.array(result['F1']).mean() for result in v2])
    
    #plt.plot([np.array(result['precision']).mean() for result in v_result_all])
    #plt.plot([np.array(result['recall']).mean() for result in v_result_all])
        
    plt.plot(x, [np.array(result['F1']).sum()/10 for result in c_result_all])
    #plt.plot(x, [np.array(result['precision']).mean() for result in c_result_all])
    #plt.plot(x, [np.array(result['recall']).mean() for result in c_result_all])
    plt.plot(x, [np.array(result['F1']).sum()/10 for result in m_result_all])
    plt.plot(x, [np.array(result['F1']).sum()/10 for result in p_result_all])
    plt.plot(x, [np.array(result['F1']).sum()/10 for result in s_result_all])
    
    #plt.legend(['volpy', 'caiman', 'mean roi'])
    plt.legend(['volpy', 'caiman', 'mean_roi', 'pca_ica', 'sgpmd'])
    plt.xlabel('spike amplitude')
    plt.ylabel('F1 score')
    plt.title('F1 score for three methods')
    
#%%
def plot_components(data, mode='temporal', row=2, col=5, scope=[0, 1000]):
    fig, ax = plt.subplots(row, col)
    for idx in range(data.shape[0]):
        h = int(idx/col)
        w = idx - h * col
        if mode == 'spatial':
            ax[h,w].imshow(data[idx])
        elif mode == 'temporal':
            ax[h,w].plot(data[idx][scope[0]:scope[1]])
        ax[h,w].get_yaxis().set_visible(False)
        ax[h,w].get_xaxis().set_visible(False)
        ax[h,w].spines['right'].set_visible(False)
        ax[h,w].spines['top'].set_visible(False)  
        ax[h,w].spines['left'].set_visible(False)
        ax[h,w].spines['bottom'].set_visible(False)

#%%
plot_components(c_spatial, mode='spatial')
plot_components(c_temporal_p, mode='temporal')

#%%
plot_components(v_spatial, mode='spatial')
plot_components(v_temporal, mode='temporal')

#%%
plt.figure()
ax = plt.gca()
width = 0.35
#labels = ['1', '2', '4', '6']
labels = [str(i) for i in range(10)]
x = np.arange(len(labels))
rects1 = ax.bar(x - width/2, v_result['F1'], width, label='volpy')
rects2 = ax.bar(x + width/2, c_result['F1'], width, label='caiman')

# Add some text for labels, title and custom x-axis tick labels, etc.
ax.set_ylabel('F1 Score')
ax.set_xticks(x)
ax.set_xticklabels(labels)
ax.spines['right'].set_visible(False)
ax.spines['top'].set_visible(False)
ax.legend()

#%%
print(f"volpy average 10 neurons:{np.array(v_result['F1']).mean()}")
print(f"caiman average 10 neurons:{np.array(c_result['F1']).mean()}")

#print(f"volpy average 8 neurons without bad example: {np.array(sorted(v_result['F1'])[2:]).mean()}")
print(f"caiman average 9 neurons without bad example: {np.array(sorted(c_result['F1'])[1:]).mean()}")


#%%
plt.plot(vpy.estimates['rawROI'][0]['t'])

#%%
#%%
    """    
    m.fr= 300
    #mm = (-m).computeDFF(0.1)[0] #!!!!!
    #mm = mm - mm.min()
    #plt.plot(mm[:, 30:50, 65:85].mean((1,2)))
    #plt.plot(mm[:, 5:25, 45:65].mean((1,2)))
    ylim = [260, 280]
    xlim = [70, 80]
    plt.plot(m[:, ylim[0]:ylim[1], xlim[0]:xlim[1]].mean((1,2)))
    plt.imshow(m[0, ylim[0]:ylim[1], xlim[0]:xlim[1]])
    plt.plot(mm[:, ylim[0]:ylim[1], xlim[0]:xlim[1]].mean((1,2)))
    #plt.plot(signal_filter(mm[:, 5:25, 45:65].mean((1,2))[np.newaxis,:], fr=400, freq=15)[0])
    plt.plot(signal_filter(mm[:, ylim[0]:ylim[1], xlim[0]:xlim[1]].mean((1,2))[np.newaxis,:], fr=300, freq=15)[0])
    
#%%
    #mm.save('/home/nel/data/voltage_data/simul_electr/kaspar/Ground truth data/Session1/Session1_DFF.tif')
    mm.save(fnames[0][:-5]+'_DFF.hdf5')
    """
    fnames = ['/home/nel/caiman_data/example_movies/volpy/demo_voltage_imaging.hdf5']
    #fnames = ['/home/nel/caiman_data/example_movies/volpy/demo_voltage_imaging_flip.hdf5']
    #fnames = ['/home/nel/caiman_data/example_movies/volpy/demo_voltage_imaging_inverse.hdf5']
    #fnames = ['/home/nel/caiman_data/example_movies/volpy/demo_voltage_imaging_highpass.hdf5']
    fnames = ['/home/nel/caiman_data/example_movies/volpy/demo_voltage_imaging_DFF.hdf5']
    fnames = ['/home/nel/data/voltage_data/simul_electr/kaspar/Ground truth data/Session1/Session1_DFF.tif']
    fnames = ['/home/nel/data/voltage_data/simul_electr/johannes/10052017Fish2-2/registered_DFF.tif']
    fnames = ['/home/nel/data/voltage_data/simul_electr/johannes/10052017Fish2-2/registered_flip.tif']
    fnames = ['/home/nel/NEL-LAB Dropbox/NEL/Papers/VolPy_online/test_data/multiple_neurons/06152017Fish1-2_portion.hdf5']
    fnames = ['/home/nel/NEL-LAB Dropbox/NEL/Papers/VolPy_online/test_data/multiple_neurons/06152017Fish1-2_portion_DFF.hdf5']
    #fnames = ['/home/nel/NEL-LAB Dropbox/NEL/Papers/VolPy_online/test_data/multiple_neurons/06152017Fish1-2_portion_flip.hdf5']
    fnames = ['/home/nel/NEL-LAB Dropbox/NEL/Papers/VolPy_online/test_data/multiple_neurons/IVQ32_S2_FOV1_processed_mc.hdf5']
    fnames = ['/home/nel/NEL-LAB Dropbox/NEL/Papers/VolPy_online/test_data/multiple_neurons/IVQ32_S2_FOV1_processed_mc_DFF.hdf5']
    fnames = ['/home/nel/NEL-LAB Dropbox/NEL/Papers/VolPy_online/test_data/multiple_neurons/FOV4_50um.hdf5']
    fnames = ['/home/nel/NEL-LAB Dropbox/NEL/Papers/VolPy_online/test_data/multiple_neurons/FOV4_50um_DFF.hdf5']
    fnames = ['/home/nel/caiman_data/example_movies/volpy/demo_voltage_imaging_remove_pixelwise.hdf5']
    fnames = ['/home/nel/caiman_data/example_movies/volpy/demo_voltage_imaging_remove_pixelwise_add500.hdf5']
    fnames = ['/home/nel/NEL-LAB Dropbox/NEL/Papers/VolPy/test_data/06152017Fish1-2.hdf5']
    fnames = ['/home/nel/NEL-LAB Dropbox/NEL/Papers/VolPy/test_data/06152017Fish1-2_flip.hdf5']
    fnames = ['/home/nel/NEL-LAB Dropbox/NEL/Papers/VolPy/test_data/multiple_neurons/IVQ32_S2_FOV1_processed.hdf5']
    fnames = ['/home/nel/Code/volpy_test/invivo-imaging/demo_data/raw_data.tif']
    fnames = ['/home/nel/NEL-LAB Dropbox/NEL/Papers/VolPy/test_data/multiple_neurons/FOV4_50um_flip.hdf5']
    fnames = ['/home/nel/NEL-LAB Dropbox/NEL/Papers/VolPy/test_data/simulation/demo_simulated_flip.hdf5']
    fnames = ['/home/nel/NEL-LAB Dropbox/NEL/Papers/VolPy/test_data/simulation/sim3_4/sim3_4_flip.hdf5']
    fnames = ['/home/nel/NEL-LAB Dropbox/NEL/Papers/VolPy/test_data/simulation/sim3_8/sim3_8_flip.hdf5']
    fnames = ['/home/nel/NEL-LAB Dropbox/NEL/Papers/VolPy/test_data/simulation/sim3_9/sim3_9_flip.hdf5']
