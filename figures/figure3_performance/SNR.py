#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Aug 18 10:29:14 2020

@author: agiovann
"""
import numpy as np
import caiman as cm
from caiman.source_extraction.cnmf.cnmf import load_CNMF
from caiman.source_extraction.volpy.spikepursuit import signal_filter
from scipy.signal import find_peaks
from skimage import measure
import matplotlib
matplotlib.rcParams['pdf.fonttype'] = 42
matplotlib.rcParams['ps.fonttype'] = 42
import matplotlib.pyplot as plt

#%%
def normalize(data):
    data = data - np.median(data)
    ff1 = -data * (data < 0)
    Ns = np.sum(ff1 > 0)
    std = np.sqrt(np.divide(np.sum(ff1**2), Ns))
    data_norm = data/std 
    return data_norm

def compute_SNR(s1, s2, height=3.5, do_plot=True):
    s1 = normalize(s1)
    s2 = normalize(s2)
    pks1 = find_peaks(s1, height)[0]
    pks2 = find_peaks(s2, height)[0]
    both_found = np.intersect1d(pks1,pks2)

    if do_plot:
        plt.figure()
        plt.plot(s1)
        plt.plot(s2)
        plt.plot(both_found, s1[both_found],'o')
        plt.plot(both_found, s2[both_found],'o')
        plt.legend(['volpy','caiman','volpy_peak','caiman_peak'])
    
    print(len(both_found))
    print([np.mean(s1[both_found]),np.mean(s2[both_found])])
    snr = [np.mean(s1[both_found]),np.mean(s2[both_found])]
    if len(both_found) < 10:
        snr = [np.nan, np.nan]
    #print([np.std(s1[both_found]),np.std(s2[both_found])])
    return snr

def compute_SNR_S(S, height=3.5, do_plot=True, volpy_peaks=None):
    pks = {}
    dims = S.shape
    for idx, s in enumerate(S):
        S[idx] = normalize(s)
        pks[idx] = set(find_peaks(s, height)[0])
    
    if volpy_peaks is not None:
        pks[0] = set(volpy_peaks)
        print('use_volpy_peaks')
        
    found = np.array(list(set.intersection(pks[0], pks[1], pks[2], pks[3])))    

    if do_plot:
        plt.figure()
        plt.plot(S.T)
        plt.plot(found, S[0][found],'o')
        plt.legend(['volpy','caiman','volpy_peak','caiman_peak'])
    
    print(len(found))
    snr = np.mean(S[:, found], 1)
    print(np.round(snr, 2))
    if len(found) < 10:
        snr = [np.nan] * dims[0]
    #print([np.std(s1[both_found]),np.std(s2[both_found])])
    return snr
#%% VolPy & caiman file
fr = 400
#mov = cm.load('/home/nel/NEL-LAB Dropbox/NEL/Papers/VolPy_online/test_data/multiple_neurons/IVQ32_S2_FOV1_processed.hdf5')
#mask = cm.load('/home/nel/NEL-LAB Dropbox/NEL/Papers/VolPy_online/test_data/multiple_neurons/IVQ32_S2_FOV1_ROIs.hdf5')
#mov = cm.load('/home/nel/NEL-LAB Dropbox/NEL/Papers/VolPy_online/test_data/multiple_neurons/06152017Fish1-2_portion.hdf5')
#mask = cm.load('/home/nel/NEL-LAB Dropbox/NEL/Papers/VolPy_online/test_data/multiple_neurons/06152017Fish1-2_portion_ROIs.hdf5')
mov = cm.load('/home/nel/NEL-LAB Dropbox/NEL/Papers/VolPy_online/test_data/multiple_neurons/FOV4_50um.hdf5')
mask = cm.load('/home/nel/NEL-LAB Dropbox/NEL/Papers/VolPy_online/test_data/multiple_neurons/FOV4_50um_ROIs.hdf5')


#%%
#fl = '/Users/agiovann/NEL-LAB Dropbox/NEL/Papers/VolPy_online/result/test_multiple_neurons/volpy_IVQ32_S2_FOV1.npy'
#fl = '/home/nel/NEL-LAB Dropbox/NEL/Papers/VolPy_online/result/test_multiple_neurons/volpy_IVQ32_S2_FOV1.npy'
fl = '/home/nel/NEL-LAB Dropbox/NEL/Papers/VolPy_online/result/test_multiple_neurons/volpy_FOV4_35um.npy'
#fl = '/home/nel/NEL-LAB Dropbox/NEL/Papers/VolPy_online/result/test_multiple_neurons/volpy_06152017Fish1-2.npy'
#fl2 = '/Users/agiovann/NEL-LAB Dropbox/NEL/Papers/VolPy/test_result/multiple_neurons/caiman_IVQ32_S2_FOV1.npy'
#fl2 = '/home/nel/NEL-LAB Dropbox/NEL/Papers/VolPy/test_result/multiple_neurons/caiman_IVQ32_S2_FOV1.npy'
#fl2 ='/home/nel/NEL-LAB Dropbox/NEL/Papers/VolPy_online/result/test_multiple_neurons/caiman_06152017Fish1-2.npy'
fl2 = '/home/nel/NEL-LAB Dropbox/NEL/Papers/VolPy/test_result/multiple_neurons/caiman_FOV4_50um.npy'
#fl3 = '/home/nel/Code/volpy_test/invivo-imaging/test_data/HPC/output/nmf_traces.tif'
#fl3_spatial =  '/home/nel/Code/volpy_test/invivo-imaging/test_data/HPC/output/spatial_footprints.tif'
#fl3 = '/home/nel/Code/volpy_test/invivo-imaging/test_data/TEG_small/output/nmf_traces.tif'
#fl3_spatial = '/home/nel/Code/volpy_test/invivo-imaging/test_data/TEG_small/output/spatial_footprints.tif'
fl3 = '/home/nel/Code/volpy_test/invivo-imaging/test_data/L1_all/output/nmf_traces.tif'
fl3_spatial = '/home/nel/Code/volpy_test/invivo-imaging/test_data/L1_all/output/spatial_footprints.tif'


#%%
vpy  = np.load(fl, allow_pickle=True)[()]
idx_list = np.where(np.array([vpy['spikes'][i].shape[0] for i in range(len(vpy['spikes']))])>50)[0]
s1 = vpy['ts'][1]
plt.figure(); plt.plot(s1) # after whitened matched filter
plt.figure(); plt.imshow(vpy['weights'][1])


caiman_estimates = np.load(fl2, allow_pickle=True)[()]
idx = 2
plt.figure();plt.plot(caiman_estimates.C[idx])    
s2 = signal_filter(caiman_estimates.C, freq=15, fr=fr)[idx]
plt.plot(s2);
plt.figure();plt.imshow(caiman_estimates.A[:,idx].toarray().reshape(caiman_estimates.dims, order='F'))

idx = 1
s3 = signal_filter(mov[:,mask[idx]>0].mean(1), freq=15, fr=fr)
plt.plot(s3)

idx = 1
s4 = signal_filter(cm.load(fl3).T, freq=15, fr=fr)[idx]
s4_spatial = cm.load(fl3_spatial)
plt.plot(s4)

#%%
sf = 101
s4 = np.append(np.array([0]*sf), s4)
S = np.vstack([s1, s2, s3, s4])

snr_all = compute_SNR_S(S, volpy_peaks=vpy['spikes'][1])
np.save('/home/nel/NEL-LAB Dropbox/NEL/Papers/VolPy/test_result/multiple_neurons/IVQ32_S2_FOV1_snr.npy', snr_all)

#%%
fr = 300
vpy  = np.load(fl, allow_pickle=True)[()]
idx_list = np.where(np.array([vpy['spikes'][i].shape[0] for i in range(len(vpy['spikes']))])>30)[0]
idx_list = [0, 5, 6]
for idx in idx_list:
    #idx = idx_list[2]
    s1 = vpy['ts'][idx]
    plt.figure(); plt.plot(s1) # after whitened matched filter
    plt.figure(); plt.imshow(vpy['weights'][idx])
idx_volpy = [0,5]
s1 = vpy['ts'][np.array([0,5])]

caiman_estimates = np.load(fl2, allow_pickle=True)[()]
idx = 9
plt.figure();plt.plot(caiman_estimates.C[idx])    
s2 = signal_filter(caiman_estimates.C, freq=15, fr=fr)[idx]
plt.plot(s2);
plt.figure();plt.imshow(caiman_estimates.A[:,idx].toarray().reshape(caiman_estimates.dims, order='F'))
s2 = signal_filter(caiman_estimates.C[np.array([13, 9])], freq=15, fr=fr)

idx = 0
s3 = -signal_filter(mov[:,mask[idx]>0].mean(1), freq=15, fr=fr)
plt.plot(s3)

s3 = np.array([-signal_filter(mov[:,mask[idx]>0].mean(1), freq=15, fr=fr) for idx in [0, 5]])

idx = 1
s4 = signal_filter(cm.load(fl3)[:,np.array([1,0])].T, freq=15, fr=fr)
s4_spatial = cm.load(fl3_spatial)
plt.plot(s4.T)

#%%
sf = 101
s4 = np.hstack((np.zeros((2, 101)), s4))

snr_all = []
#idx = 0
for idx in [0,1]:
    S = np.vstack([s1[idx], s2[idx], s3[idx], s4[idx]])
    snr_all.append(compute_SNR_S(S, volpy_peaks=vpy['spikes'][idx_volpy[idx]]))
np.save('/home/nel/NEL-LAB Dropbox/NEL/Papers/VolPy/test_result/multiple_neurons/06152017Fish1-2_snr.npy', snr_all)


#%% L1
fr = 400
vpy  = np.load(fl, allow_pickle=True)[()]
idx_list = np.where(np.array([vpy['spikes'][i].shape[0] for i in range(len(vpy['spikes']))])>30)[0]
#idx_list = [0, 5, 6]
idx_list = np.array([ 0,  3, 12, 16, 18, 20, 26, 27])
for idx in idx_list:
    #idx = idx_list[2]
    print(idx)
    s1 = vpy['ts'][idx]
    plt.figure(); plt.plot(s1);plt.show() # after whitened matched filter
    plt.figure(); plt.imshow(vpy['weights'][idx]);plt.show()
#plt.imshow(vpy['weights'][idx_list].sum(0), cmap='gray')
idx_volpy = np.array([ 0,  3, 12, 16, 18, 20, 26, 27])
s1 = vpy['ts'][np.array(idx_list)]

caiman_estimates = np.load(fl2, allow_pickle=True)[()]
idx = 9
plt.figure();plt.plot(caiman_estimates.C[idx])    
s2 = signal_filter(caiman_estimates.C, freq=15, fr=fr)[idx]
plt.plot(s2);
plt.figure();plt.imshow(caiman_estimates.A[:,idx].toarray().reshape(caiman_estimates.dims, order='F'))
idx_caiman = np.array([161,  41, 168, 167, 172,  96, 101, 163])
s2 = signal_filter(caiman_estimates.C[idx_caiman], freq=15, fr=fr)

idx = 0
s3 = -signal_filter(mov[:,mask[idx]>0].mean(1), freq=15, fr=fr)
plt.plot(s3)
s3 = np.array([-signal_filter(mov[:,mask[idx]>0].mean(1), freq=15, fr=fr) for idx in idx_list])

idx = 1
idx_sgpmd = np.array([ 9, 13,  3,  1,  7,  4,  8,  5])
s4 = signal_filter(cm.load(fl3)[:,idx_sgpmd].T, freq=15, fr=fr)
s4_spatial = cm.load(fl3_spatial)
s4_spatial = s4_spatial.reshape([512,128, 20], order='F')
s4_spatial = s4_spatial / s4_spatial.max((0,1))
s4_spatial = s4_spatial.transpose([2, 0, 1])
plt.plot(s4.T)



#%%
sf = 101
s4 = np.hstack((np.zeros((s4.shape[0], sf)), s4))
s4 = s4[:,:10000]
s1 = s1[:,:10000]
s2 = s2[:,:10000]
s3 = s3[:,:10000]

snr_all = []
#idx = 0
for idx in range(s1.shape[0]):
    print(idx)
    S = np.vstack([s1[idx], s2[idx], s3[idx], s4[idx]])
    snr_all.append(compute_SNR_S(S, volpy_peaks=vpy['spikes'][idx_volpy[idx]]))
np.save('/home/nel/NEL-LAB Dropbox/NEL/Papers/VolPy/test_result/multiple_neurons/FOV4_50um_snr.npy', snr_all)




#%%
#idx_list = [0]
#idx_volpy = [1]
#idx_list = [0, 1]
#idx_volpy = [0, 5]
idx_list = list(range(s1.shape[0]))

colorsets = plt.cm.tab10(np.linspace(0,1,10))
colorsets = colorsets[[0,1,2,3,4,5,6,8,9],:]

scope = [0, 10000]
if s1.shape[0] > 300:
    s1 = s1[np.newaxis,:]
    s2 = s2[np.newaxis,:]
    s3 = s3[np.newaxis,:]
    s4 = s4[np.newaxis,:]

dims = s1.shape

fig, ax = plt.subplots(dims[0],1)
#ax = [ax]
for idx in idx_list:
    ax[idx].plot(normalize(s1[idx]), 'c', linewidth=0.5, color='black')
    ax[idx].plot(normalize(s2[idx]), 'c', linewidth=0.5, color='red')
    ax[idx].plot(normalize(s3[idx]), 'c', linewidth=0.5, color='orange')
    ax[idx].plot(normalize(s4[idx]), 'c', linewidth=0.5, color='green')
    ax[idx].legend(['volpy', 'caiman', 'mean', 'sgpmd'])
    
    height = 5
    pks1 = vpy['spikes'][idx_volpy][idx]
    pks2 = find_peaks(normalize(s2[idx]), height)[0]
    pks3 = find_peaks(normalize(s3[idx]), height)[0]
    pks4 = find_peaks(normalize(s4[idx]), height)[0]
    
    add = 15
    
    ax[idx].vlines(pks1, add+3.5, add+4, color='black')
    ax[idx].vlines(pks2, add+2.5, add+3, color='red')
    ax[idx].vlines(pks3, add+1.5, add+2, color='orange')
    ax[idx].vlines(pks4, add+0.5, add+1, color='green')
    
    ax[idx].set_xlim([3000, 5000])
    #ax[idx].set_xlim([8000, 9000])
    
    
    if idx<dims[0]-1:
        ax[idx].get_xaxis().set_visible(False)
        ax[idx].spines['right'].set_visible(False)
        ax[idx].spines['top'].set_visible(False) 
        ax[idx].spines['bottom'].set_visible(False) 
        ax[idx].spines['left'].set_visible(False) 
        ax[idx].set_yticks([])
    
    if idx==dims[0]-1:
        ax[idx].legend()
        ax[idx].spines['right'].set_visible(False)
        ax[idx].spines['top'].set_visible(False)  
        ax[idx].spines['left'].set_visible(True) 
        ax[idx].set_xlabel('Frames')
    
    ax[idx].set_ylabel('o')
    ax[idx].get_yaxis().set_visible(True)
    ax[idx].yaxis.label.set_color(colorsets[np.mod(idx,9)])
    #plt.savefig('/home/nel/NEL-LAB Dropbox/NEL/Papers/VolPy/Figures/plos/original_files/figure3/IVQ32_S2_FOV1_temporal.pdf')
    #plt.savefig('/home/nel/NEL-LAB Dropbox/NEL/Papers/VolPy/Figures/plos/original_files/figure3/06152017_FIsh1-2_temporal.pdf')
    plt.savefig('/home/nel/NEL-LAB Dropbox/NEL/Papers/VolPy/Figures/plos/original_files/figure3/FOV4_50um_temporal.pdf')

#%%
Cn = mov[0]
vmax = np.percentile(Cn, 99)
vmin = np.percentile(Cn, 5)
plt.figure()
plt.imshow(Cn, interpolation='None', vmax=vmax, vmin=vmin, cmap=plt.cm.gray)
plt.title('Neurons location')
d1, d2 = Cn.shape
#cm1 = com(mask.copy().reshape((N,-1), order='F').transpose(), d1, d2)
colors='yellow'
for n, idx in enumerate(idx_volpy):
    contours = measure.find_contours(mask[idx], 0.5)[0]
    plt.plot(contours[:, 1], contours[:, 0], linewidth=1, color=colorsets[np.mod(n,9)])
    #plt.savefig('/home/nel/NEL-LAB Dropbox/NEL/Papers/VolPy_online/picture/Figures/multiple_neurons/FOV4_50um_footprints.pdf')
    #plt.savefig('/home/nel/NEL-LAB Dropbox/NEL/Papers/VolPy_online/picture/Figures/multiple_neurons/06152017Fish1-2_footprints.pdf')   
    #plt.savefig('/home/nel/NEL-LAB Dropbox/NEL/Papers/VolPy_online/picture/Figures/multiple_neurons/IVQ32_S2_FOV1_footprints.pdf')   
    #plt.savefig('/home/nel/NEL-LAB Dropbox/NEL/Papers/VolPy/Figures/plos/original_files/figure3/IVQ32_S2_FOV1_spatial.pdf')
    #plt.savefig('/home/nel/NEL-LAB Dropbox/NEL/Papers/VolPy/Figures/plos/original_files/figure3/06152017_FIsh1-2_spatial.pdf')
    plt.savefig('/home/nel/NEL-LAB Dropbox/NEL/Papers/VolPy/Figures/plos/original_files/figure3/FOV4_50um_spatial.pdf')




#%%
idx = 0
plt.figure();plt.plot(signal_filter(caiman_estimates.C, freq=15, fr=400)[idx])
plt.figure();plt.imshow(caiman_estimates.A[:,idx].toarray().reshape((512, 128), order='F'))
    
mask0 = np.array(mask[idx_list])
mask1 = caiman_estimates.A.toarray()[:, caiman_estimates.idx].reshape((512, 128, -1), order='F').transpose([2, 0, 1])
mask1[mask1>0.02] = 1
mask1 = np.float32(mask1)
plt.figure();plt.imshow(mask0.sum(0));plt.colorbar();plt.show()
        
from caiman.base.rois import nf_match_neurons_in_binary_masks
tp_gt, tp_comp, fn_gt, fp_comp, performance_cons_off = nf_match_neurons_in_binary_masks(
        mask0, mask1, thresh_cost=1, min_dist=10, print_assignment=True,
        plot_results=True, Cn=mov[0], labels=['viola', 'cm'])    


#%%


idx_list = 


mask0 = np.array(mask[idx_list])
mask1 = s4_spatial
mask1 = np.array(mask1)
        
from caiman.base.rois import nf_match_neurons_in_binary_masks
tp_gt, tp_comp, fn_gt, fp_comp, performance_cons_off = nf_match_neurons_in_binary_masks(
        mask0, mask1, thresh_cost=1, min_dist=10, print_assignment=True,
        plot_results=True, Cn=mov[0], labels=['viola', 'cm'])    







#%%
vpy  = np.load(fl, allow_pickle=True)[()]
idx_list = np.where(np.array([vpy['spikes'][i].shape[0] for i in range(len(vpy['spikes']))])>50)[0]
#idx_list = [1]
#tp_comp = [2]
caiman_estimates = np.load(fl2, allow_pickle=True)[()]
do_plot = True

snr_list = []
for idx in range(len(idx_list)):
    s1 = vpy['ts'][idx_list][idx]
    #s1 = signal_filter(s1[np.newaxis, :], freq=15, fr=fr)[0]
    s2 = signal_filter(caiman_estimates.C, freq=15, fr=fr)[tp_comp][idx]

    
    #plt.figure(); plt.plot(s1) # after whitened matched filter
    #plt.figure();plt.plot(caiman_estimates.C[tp_comp][idx]); plt.plot(s2);
    if do_plot:
        plt.figure();
        plt.subplot(1,2,1); plt.imshow(vpy['weights'][idx_list][idx])
        plt.subplot(1,2,2); plt.imshow(caiman_estimates.A[:,tp_comp[idx]].toarray().reshape(caiman_estimates.dims, order='F'))
        plt.show()
    
    snr_list.append(compute_SNR(s1, s2, height=3.5, do_plot=do_plot))

snr_list = np.round(np.nanmean(np.array(snr_list), 0), 2)
print(f'volpy:{snr_list[0]},  caiman:{snr_list[1]}')






