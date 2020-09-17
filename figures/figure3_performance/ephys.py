# -*- coding: utf-8 -*-
"""
File for processing simultaneous electrophysiology with voltage imaging data
@author: caichangjia
"""
#%%
import numpy as np
import matplotlib
matplotlib.rcParams['pdf.fonttype'] = 42
matplotlib.rcParams['ps.fonttype'] = 42
import matplotlib.pyplot as plt
import os
import scipy.signal

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

#%% Johannes's data
#root_dir = '/home/nel/data/voltage_data/simul_electr/johannes/09282017Fish1-1'
root_dir = '/home/nel/data/voltage_data/simul_electr/johannes/10052017Fish2-2'
#root_dir = '/home/nel/data/voltage_data/simul_electr/johannes/09212017Fish1-1'

#%% Load signal
ephys = np.load(os.path.join(root_dir, 'ephys.npy'))
frame_timing = np.load(os.path.join(root_dir,'frame_timing.npy'))
imaging_signal = np.load(os.path.join(root_dir, 'imaging_signal.npy'))
plt.plot(ephys);plt.hlines(0.5, 0, len(ephys), linestyles='dashed', color='gray')
plt.plot(frame_timing,(-imaging_signal+1)*20)

#%% Pre-processing
n = 0
hline = 0.5
estimates = np.load(os.path.join(root_dir, 'estimates.npz'), allow_pickle=True)['arr_0'].item()
trace = estimates['trace_processed'][n][:len(frame_timing)]
trace = trace - np.mean(trace)
spikes = estimates['spikes'][n].copy()
spikes = np.delete(spikes, np.where(spikes>len(frame_timing))).astype(np.int32)
timepoint = frame_timing
scope = [0, len(ephys)]
ephys = ephys/np.max(ephys)
trace = trace/np.max(trace)

plt.plot(frame_timing, trace, label='volpy',color='orange')
plt.plot(frame_timing[spikes], 1.1 * np.max(trace) * np.ones(spikes.shape),
         color='orange', marker='.', fillstyle='none', linestyle='none')
plt.plot(ephys, label='ephys', color='blue')
#plt.hlines(hline, 0, len(ephys), linestyles='dashed', color='gray')
etime = scipy.signal.find_peaks(ephys, hline, distance=200)[0]
plt.plot(etime,  np.max(ephys) * 1.2 * np.ones(etime.shape),
         color='b', marker='.', fillstyle='none', linestyle='none')
plt.legend()

"""
#SGPMD-NMF
trace_pmd = io.imread('/home/nel/Code/volpy_test/invivo-imaging/test_data/09282017Fish1-1/output/temporal_traces.tif')[1]
trace_pmd = np.concatenate((np.array([0]*100), trace_pmd))[:len(frame_timing)]
trace_pmd = signal_filter(trace_pmd, freq=15, fr=400)
trace_pmd = trace_pmd / np.max(trace_pmd)
plt.plot(frame_timing, trace_pmd, label='pmd',color='red')
spikes_pmd = scipy.signal.find_peaks(trace_pmd, 0.45, distance=5)[0]
plt.plot(frame_timing[spikes_pmd],  np.max(ephys) * 1.3 * np.ones(spikes_pmd.shape),
         color='red', marker='.', fillstyle='none', linestyle='none')
plt.legend()
"""

#%%
import caiman as cm
from caiman.base.rois import nf_read_roi_zip
masks = nf_read_roi_zip(os.path.join(root_dir, 'mask.zip'), dims=(44, 128))
masks_m = masks[0].reshape(-1, order='F')
m = cm.load('/home/nel/data/voltage_data/simul_electr/johannes/09282017Fish1-1/memmap__d1_44_d2_128_d3_1_order_C_frames_37950_.mmap')
mm = m.reshape((m.shape[0], -1), order='F')
trace_mean = (mm[:, masks_m>0]).mean(1)
trace_mean = signal_filter(-trace_mean[np.newaxis, :], freq=1/3, fr=300)[0]
trace_mean = signal_filter(trace_mean[np.newaxis, :], freq=10, fr=300)[0]
trace_mean = trace_mean / trace_mean.max()
plt.plot(trace_mean)
spikes_mean = scipy.signal.find_peaks(trace_mean, 0.3, distance=5)[0]
spikes_mean = np.delete(spikes_mean, np.where(spikes_mean>len(frame_timing))).astype(np.int32)

#%%
import caiman as cm
from caiman.base.rois import nf_read_roi_zip
masks = nf_read_roi_zip(os.path.join(root_dir, 'mask.zip'), dims=(32, 64))
masks_m = masks[0].reshape(-1, order='F')
m = cm.load('/home/nel/data/voltage_data/simul_electr/johannes/10052017Fish2-2/memmap__d1_32_d2_64_d3_1_order_C_frames_37950_.mmap')
mm = m.reshape((m.shape[0], -1), order='F')
trace_mean = (mm[:, masks_m>0]).mean(1)
trace_mean = signal_filter(trace_mean[np.newaxis, :], freq=1/3, fr=300)[0]
trace_mean = signal_filter(trace_mean[np.newaxis, :], freq=10, fr=300)[0]
trace_mean = trace_mean / trace_mean.max()
plt.plot(trace_mean)
spikes_mean = scipy.signal.find_peaks(trace_mean, 0.5, distance=5)[0]
spikes_mean = np.delete(spikes_mean, np.where(spikes_mean>len(frame_timing))).astype(np.int32)


#%%
est = np.load('/home/nel/data/voltage_data/simul_electr/johannes/10052017Fish2-2/estimates_caiman.npz', allow_pickle=True)['arr_0'].item()
trace_cm = signal_filter(est.C, freq=10, fr=400)[est.idx]
trace_cm = trace_cm / trace_cm.max()
plt.plot(trace_cm)
spikes_cm = scipy.signal.find_peaks(trace_cm, 0.65, distance=5)[0]
spikes_cm = np.delete(spikes_cm, np.where(spikes_cm>len(frame_timing))).astype(np.int32)
#%% Comparison of signal
scope = [0, 600000]
max_dist = 500
precision, recall, F1 = compare_with_ephys_match(sg_gt=ephys, sp_gt=etime, sg=trace, 
                                                 sp=spikes, timepoint=timepoint, 
                                                 max_dist=max_dist, scope=scope, hline=hline)  
#plt.savefig(os.path.join(root_dir, 'result_new.pdf'))
#%%
scope = [0, 600000]
max_dist = 500
precision, recall, F1 = compare_with_ephys_match(sg_gt=ephys, sp_gt=etime, sg=trace_mean, 
                                                 sp=spikes_mean, timepoint=timepoint, 
                                                 max_dist=max_dist, scope=scope, hline=hline)  

#%%
scope = [0, 600000]
max_dist = 500
precision, recall, F1 = compare_with_ephys_match(sg_gt=ephys, sp_gt=etime, sg=trace_cm, 
                                                 sp=spikes_cm, timepoint=timepoint, 
                                                 max_dist=max_dist, scope=scope, hline=hline)  

#%%



#%%
sweep_time=None
e_t = np.arange(0, scope[1])
e_sg = ephys[:scope[1]]
e_sp = etime[etime<scope[1]]
e_sub = None
v_t = timepoint[timepoint<scope[1]]
v_sg = trace[timepoint<scope[1]]
v_sp = v_t[spikes[timepoint[spikes]<scope[1]]]
v_sub = None
save_name = '/home/nel/NEL-LAB Dropbox/NEL/Papers/VolPy/Marton/data_new/09282017Fish1-1_output.npz'
np.savez(save_name, sweep_time=sweep_time, e_sg=e_sg, v_sg=v_sg, 
         e_t=e_t, v_t=v_t, e_sp=e_sp, v_sp=v_sp, e_sub=e_sub, v_sub=v_sub)

#%%
plt.plot(e_sg);plt.vlines(e_sp, -14,-10)
plt.plot(v_t, v_sg); plt.vlines(v_t[v_sp], -20, -16)




#%% Spatial footprint
Xinds = estimates['ROI'][n][:,0] 
Yinds = estimates['ROI'][n][:,1]
plt.figure(); plt.imshow(estimates['spatialFilter'][n][Xinds[0] : Xinds[-1] + 1, Yinds[0] : Yinds[-1] + 1] ) 
plt.savefig(os.path.join(root_dir, 'spatial_new.pdf'))

#%% Kaspar's data 
root_dir = '/home/nel/data/voltage_data/simul_electr/kaspar/Ground truth data/Session1'
estimates = np.load(os.path.join(root_dir, 'estimates.npz'), allow_pickle=True)['arr_0'].item()
from scipy import io
n = 0
fr = 20000
scope = [2000000,3000000]
#scope = [2500000,3500000]
fname = '/home/nel/Dropbox_old/Kaspar-Andrea/Ground truth data/Ephys_data_session6_s2.mat'
fname = '/home/nel/data/voltage_data/simul_electr/kaspar/Ground truth data/Ephys_data_session1.mat'
f = io.loadmat(fname)
frames = np.where(np.logical_and(f['read_starts'][0]>scope[0], f['read_starts'][0]<scope[1]))[0]
timepoint = np.array([f['read_starts'][0][frames[i]]-scope[0] for i in range(frames.shape[0])])

ephys = f['v'][0][scope[0]:scope[-1]]
#plot_signal_and_spike(ephys, etime, scope)
trace = estimates['trace_processed'][n]
spikes = estimates['spikes'][n]
ephys = ephys - np.mean(ephys)
ephys = ephys/np.max(ephys)
trace = trace/np.max(trace)
etime = scipy.signal.find_peaks(ephys, 0.5, distance=200)[0]


#%%
sweep_time=None
e_t = np.arange(0, 1000000)
e_sg = ephys
e_sp = etime
e_sub = None
v_t = timepoint
v_sg = trace
v_sp = v_t[spikes[timepoint[spikes]<scope[1]]]
v_sub = None
save_name = '/home/nel/NEL-LAB Dropbox/NEL/Papers/VolPy/Marton/data_new/Mouse_Session_1.npz'
np.savez(save_name, sweep_time=sweep_time, e_sg=e_sg, v_sg=v_sg, 
         e_t=e_t, v_t=v_t, e_sp=e_sp, v_sp=v_sp, e_sub=e_sub, v_sub=v_sub)

#%%
import numpy as np
import caiman as cm
from caiman.source_extraction.volpy.spikepursuit import signal_filter
from caiman.base.rois import nf_read_roi_zip
import scipy
masks = nf_read_roi_zip(os.path.join(root_dir, 'mask.zip'), dims=(80, 128))
masks_m = masks[0].reshape(-1, order='F')
m = cm.load('/home/nel/data/voltage_data/simul_electr/kaspar/Ground truth data/Session1/memmap__d1_80_d2_128_d3_1_order_C_frames_24908_.mmap')
mm = m.reshape((m.shape[0], -1), order='F')
trace_mean = (mm[:, masks_m>0]).mean(1)
#trace_mean = signal_filter(-trace_mean[np.newaxis, :], freq=1/3, fr=500)[0]
trace_mean = signal_filter(trace_mean[np.newaxis, :], freq=10, fr=500)[0]
trace_mean = trace_mean
trace_mean = trace_mean / trace_mean.max()
plt.plot(trace_mean)
spikes_mean = scipy.signal.find_peaks(trace_mean, 0.5, distance=5)[0]
#spikes_mean = np.delete(spikes_mean, np.where(spikes_mean>len(frame_timing))).astype(np.int32)

#%%
est = np.load('/home/nel/data/voltage_data/simul_electr/kaspar/Ground truth data/Session1/estimates_caiman.npz', allow_pickle=True)['arr_0'].item()
trace_cm = signal_filter(est.C, freq=10, fr=400)[est.idx]
trace_cm = trace_cm / trace_cm.max()
plt.plot(trace_cm)
spikes_cm = scipy.signal.find_peaks(trace_cm, 0.55, distance=5)[0]
#spikes_cm = np.delete(spikes_cm, np.where(spikes_cm>len(frame_timing))).astype(np.int32)

#%%
#plt.plot(trace)
#plt.plot(trace_mean)
plt.figure();plt.plot(trace_cm)
#%%
plt.figure();plt.imshow(est.A[:,idx].toarray().reshape((80, 128), order='F'))



#%% Comparison
precision, recall, F1 = compare_with_ephys_match(sg_gt=ephys, sp_gt=etime, sg=trace, 
                                                 sp=spikes, timepoint=timepoint, 
                                                 max_dist=500, scope=[0,1000000], hline=0.4)  

plt.savefig(os.path.join(root_dir, 'result_new.pdf'))

#%%
precision, recall, F1 = compare_with_ephys_match(sg_gt=ephys, sp_gt=etime, sg=trace_mean, 
                                                 sp=spikes_mean, timepoint=timepoint, 
                                                 max_dist=500, scope=[0,1000000], hline=0.4)  

#%%
precision, recall, F1 = compare_with_ephys_match(sg_gt=ephys, sp_gt=etime, sg=trace_cm, 
                                                 sp=spikes_cm, timepoint=timepoint, 
                                                 max_dist=500, scope=[0,1000000], hline=0.4)  



#%% Spatial footprint
Xinds = estimates['ROI'][n][:,0] 
Yinds = estimates['ROI'][n][:,1]
plt.figure(); plt.imshow(estimates['spatialFilter'][n][Xinds[0] : Xinds[-1] + 1, Yinds[0] : Yinds[-1] + 1] ) 
plt.savefig(os.path.join(root_dir, 'spatial_new.pdf'))

#%% Visualize F1 score
width = 0.2  # the width of the bars
m = [0.991, 0.934, 0.918]
fig, ax = plt.subplots()
x = 0
rects2 = ax.bar(x - width/2, m[2], width/2, label='Fish1',color='r')
rects3 = ax.bar(x , m[1], width/2, label='Fish2',color='gray')
rects1 = ax.bar(x + width/2, m[0], width/2, label='Mouse',color='b')

# Add some text for labels, title and custom x-axis tick labels, etc.
ax.set_ylabel('Scores')
ax.set_ylabel('F1 score')
ax.set_ylim([0.60,1.00])
ax.spines['right'].set_visible(False)
ax.spines['top'].set_visible(False)
plt.xticks([])
ax.legend(loc=1)
plt.savefig(os.path.join('/home/nel/data/voltage_data/simul_electr', 'F1_ephys.pdf'))


#%% Subthreshold event of eletrophysiology and voltage
plt.figure(); plt.plot(timepoint, vpy.estimates['t'][1], label='t'); plt.plot(timepoint, vpy.estimates['t_sub'][1]+30, label='t_sub'); plt.plot(ephys, label='ephys');plt.legend()
plt.savefig('/home/nel/NEL-LAB Dropbox/NEL/Papers/VolPy/Backup/Sub_10Hz.pdf')
#%% Backup

#%% Others
import numpy as np
import matplotlib
matplotlib.rcParams['pdf.fonttype'] = 42
matplotlib.rcParams['ps.fonttype'] = 42
import matplotlib.pyplot as plt
import caiman as cm
import scipy.io

def plot_signal_and_spike(sg, sp, scope=None):
    if scope is not None:
        sg = sg[scope[0]:scope[1]]
        sp = sp[np.where(np.logical_and(sp>scope[0], sp<scope[1]))]
        plt.plot(sg)
        plt.plot(sp-scope[0], np.max(sg)*np.ones(sp.shape),color='g', marker='o', fillstyle='none', linestyle='none')
    else:
        plt.plot(sg)
        plt.plot(sp, np.max(sg)*np.ones(sp.shape),color='g', marker='o', fillstyle='none', linestyle='none')
    
def compare_with_ephys(sg_gt, sp_gt, sg, sp, timepoint, scope=None):
    height = np.max(np.array([np.max(sg_gt), np.max(sg)]))
    sg_gt = sg_gt[scope[0]:scope[1]]
    sp_gt = sp_gt[np.where(np.logical_and(sp_gt>scope[0], sp_gt<scope[1]))]
    plt.plot(sg_gt, color='b', label='ephys')
    plt.plot(sp_gt-scope[0], (height+2)*np.ones(sp_gt.shape),color='b', marker='o', fillstyle='none', linestyle='none')
    timepoint = [i-scope[0] for i in timepoint]
    plt.plot(timepoint, sg, color='orange', label='VolPy')
    sp = [timepoint[i] for i in sp]
    plt.plot(sp, (height+5)*np.ones(len(sp)),color='orange', marker='o', fillstyle='none', linestyle='none')
    plt.legend()


#%%
import scipy.io
scipy.io.savemat('ephys.mat',{'data':ephys})
fname = 'times_ephys.mat'
f = scipy.io.loadmat(fname)
ff = f['cluster_class'][np.where(np.logical_or(f['cluster_class'][:,0] == 0, f['cluster_class'][:,0] == 0))[0]]
#ff = f['cluster_class']
etime = (ff[:,1]/1000*30000).astype(np.int32)
#etime = np.array(np.where(ephys>0.6))
#%%
cd '/home/nel/anaconda3/envs/ephys/etc/mountainlab/packages/ml_pyms/mlpy'
from mdaio import writemda16i,writemda32,readmda
from mountainlab_pytools import mdaio
cd '/home/nel/Dropbox_old/SimultaneousEphys/MountainSort/dataset'
writemda32(ephys,'raw.mda')

# csv file
a = np.array([[1,0],[2,0]]).astype(np.int16)
np.savetxt("geom.csv", a, delimiter=",")
#np.readtxt('/home/nel/Code/VolPy/mountainsort_examples-master/bash_examples/001_ms4_bash_example/dataset/geom.csv',delimiter=',')


name = '/home/nel/Dropbox_old/SimultaneousEphys/09212017Fish1-1/RoiSet.zip'
dims=(44,96)
from caiman.base.rois import nf_read_roi_zip
img = nf_read_roi_zip(name,dims)
plt.figure();plt.imshow(img.sum(axis=0))

path='/tmp/mountainlab-tmp/output_47670870cb9afee72b43d0c03d477d36581a055e_timeseries_out.mda'

path = '/home/nel/Dropbox_old/SimultaneousEphys/MountainSort/dataset/raw.mda'
path = '/tmp/mountainlab-tmp/output_47670870cb9afee72b43d0c03d477d36581a055e_timeseries_out.mda'
X = readmda(path)

path = '/home/nel/Dropbox_old/SimultaneousEphys/MountainSort/output/firings.mda'
Y = readmda(path)

plt.figure()
plt.plot(X[0])
plt.scatter(Y[1],np.repeat(0.3, len(Y[1])))

#%% Yichun
fname = '/home/nel/Dropbox_old/Voltron_fly_data/PPL1_ap2a2/axon/180319-1voltron/Data3/Data3_mini.mat'
import h5py
import numpy as np
arrays = {}
f = h5py.File(fname)
gt = f['wholeCell']['voltage'][0]
#%%
name = '/home/nel/Dropbox_old/Voltron_fly_data/PPL1_ap2a2/axon/180319-1voltron/Data3/Data3_timeStamp.txt'
file = open(name,'r')
a = []
for line in file:
    n = np.int(np.float(line[:-3])*10000)
    a.append(n)



#%%
sp = vpy.estimates['trace'][1]
plt.figure();plt.plot(gt+50);plt.plot(a,-sp);
plt.plot( np.array(a)[vpy.estimates['spikeTimes'][0]], np.max(vpy.estimates['trace'][0]) * 1 * np.ones(vpy.estimates['spikeTimes'][0].shape),color='g', marker='o', fillstyle='none', linestyle='none')

#%%
plt.imshow(-vpy.estimates['spatialFilter'][0])


#%% Kaspar & Amrita


#%%
fr = 20000
scope = [3000000,4000000]
fname = '/home/nel/Dropbox_old/Kaspar-Andrea/Ground truth data/Ephys_data_session6_s2.mat'
f = scipy.io.loadmat(fname)
signal = f['v'][0]
signal = signal / np.max(signal)
spike = f['spike_times'][0]
plot_signal_and_spike(signal, spike, None)
plt.plot(np.arange(0,len(signal))/20000, signal);
plt.plot(spike,np.max(signal)*np.ones(spike.shape),color='g', marker='o', fillstyle='none', linestyle='none')

frames = np.where(np.logical_and(f['read_starts'][0]>scope[0], f['read_starts'][0]<scope[1]))[0]
timepoint = [f['read_starts'][0][frames[i]] for i in range(frames.shape[0])]

f['curr_inj_stops']


#%%


#%%
compare_dict = {}
threshold = 10000
sp_gt = np.array(sp_gt[np.where(np.logical_and(sp_gt>scope[0], sp_gt<scope[1]))])
sp = np.array(sp)
for i in range(len(sp_gt)):
    #print(i)
    j = np.where(np.abs((sp-sp_gt[i]))==np.min(np.abs(sp-sp_gt[i])))[0][0]
    distance = np.abs((sp-sp_gt[i]))[j]
    if distance < threshold:
        if j not in compare_dict.values():
            compare_dict[i] = j
    TP = len(compare_dict)
    FP = len(sp) - TP
    FN = len(sp_gt) - TP
    precision = TP / (TP + FP)
    recall = TP / (TP + FN)
    F1 = 2 * (precision * recall) / (precision + recall) 





#%%

sys.path.insert(0, "/home/nel/Code/VolPy/PySpike/pyspike")
sys.path.insert(0, "/home/nel/Code/VolPy/PySpike")
sys.path.insert(0, "/home/nel/Code/VolPy/PySpike/examples")
import matplotlib.pyplot as plt
import pyspike as spk

spike_trains = spk.load_spike_trains_from_txt('/home/nel/Code/VolPy/PySpike/examples/PySpike_testdata.txt',
                                              edges=(0, 4000))
isi_profile = spk.isi_profile(spike_trains[0], spike_trains[1])
x, y = isi_profile.get_plottable_data()
plt.plot(x, y, '--k')
print("ISI distance: %.8f" % isi_profile.avrg())
plt.show()

spike_profile = spk.spike_profile(spike_trains[0], spike_trains[1])
x, y = spike_profile.get_plottable_data()
plt.plot(x, y, '--k')
print("SPIKE distance: %.8f" % spike_profile.avrg())
plt.show()

import numpy as np
from matplotlib import pyplot as plt
import pyspike as spk


st1 = spk.generate_poisson_spikes(1.0, [0, 20])
st2 = spk.generate_poisson_spikes(1.0, [0, 20])

d = spk.spike_directionality(st1, st2)

#print "Spike Directionality of two Poissonian spike trains:", d

E = spk.spike_train_order_profile(st1, st2)

plt.figure()
x, y = E.get_plottable_data()
plt.plot(x, y, '-ob')
plt.ylim(-1.1, 1.1)
plt.xlabel("t")
plt.ylabel("E")
plt.title("Spike Train Order Profile")

plt.show()



#%% Michael
cd '/home/nel/Dropbox_old/toshare_slice/QPP4-1_slice1_FOV3/'
a = np.fromfile('Sq_camera.bin', dtype=np.uint16)
f = open('experimental_parameters.txt','r')
Horizontal = np.int(f.readline().split()[-1])
Vertical = np.int(f.readline().split()[-1])
f.close()
dims = [Horizontal, Vertical]
m = cm.movie(a.reshape((dims[0],dims[1],-1), order='F')).transpose([2,0,1])
m[2000:32000].save('data.hdf5')

from scipy.io import loadmat
d = loadmat('Patch.fig', squeeze_me=True, struct_as_record=False)

#%%
Mouse 0.947
Fish2 0.976
Fish1 0.892







