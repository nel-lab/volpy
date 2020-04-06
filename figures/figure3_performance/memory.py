#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
This file is used to measure computational performance of VolPy.
Note that if testing with 1 CPU, make sure dview=None.
@author: caichangjia
"""

from memory_profiler import profile

def test_computational_performance(fnames, path_ROIs, n_processes):
    import os
    import cv2
    import glob
    import logging
    import matplotlib.pyplot as plt
    import numpy as np
    import tensorflow as tf
    import h5py
    from time import time
    
    try:
        cv2.setNumThreads(0)
    except:
        pass
    
    try:
        if __IPYTHON__:
            # this is used for debugging purposes only. allows to reload classes
            # when changed
            get_ipython().magic('load_ext autoreload')
            get_ipython().magic('autoreload 2')
    except NameError:
        pass
    
    import caiman as cm
    from caiman.motion_correction import MotionCorrect
    from caiman.utils.utils import download_demo, download_model
    from caiman.source_extraction.volpy.volparams import volparams
    from caiman.source_extraction.volpy.volpy import VOLPY
    from caiman.source_extraction.volpy.mrcnn import visualize, neurons
    import caiman.source_extraction.volpy.mrcnn.model as modellib
    from caiman.paths import caiman_datadir
    from caiman.summary_images import local_correlations_movie_offline
    from caiman.summary_images import mean_image
    from caiman.source_extraction.volpy.utils import quick_annotation
    from multiprocessing import Pool
    
    time_start = time()
    print('Start MOTION CORRECTION')

    # %%  Load demo movie and ROIs
    fnames = fnames
    path_ROIs = path_ROIs    

#%% dataset dependent parameters
    # dataset dependent parameters
    fr = 400                                        # sample rate of the movie
                                                   
    # motion correction parameters
    pw_rigid = False                                # flag for pw-rigid motion correction
    gSig_filt = (3, 3)                              # size of filter, in general gSig (see below),
                                                    # change this one if algorithm does not work
    max_shifts = (5, 5)                             # maximum allowed rigid shift
    strides = (48, 48)                              # start a new patch for pw-rigid motion correction every x pixels
    overlaps = (24, 24)                             # overlap between pathes (size of patch strides+overlaps)
    max_deviation_rigid = 3                         # maximum deviation allowed for patch with respect to rigid shifts
    border_nan = 'copy'

    opts_dict = {
        'fnames': fnames,
        'fr': fr,
        'pw_rigid': pw_rigid,
        'max_shifts': max_shifts,
        'gSig_filt': gSig_filt,
        'strides': strides,
        'overlaps': overlaps,
        'max_deviation_rigid': max_deviation_rigid,
        'border_nan': border_nan
    }

    opts = volparams(params_dict=opts_dict)

# %% start a cluster for parallel processing
    dview = Pool(n_processes)
    #dview = None
# %%% MOTION CORRECTION
    # first we create a motion correction object with the specified parameters
    mc = MotionCorrect(fnames, dview=dview, **opts.get_group('motion'))
    # Run correction
    mc.motion_correct(save_movie=True)

    time_mc = time() - time_start
    print(time_mc)
    print('START MEMORY MAPPING')
        
    # %% restart cluster to clean up memory
    dview.terminate()
    dview = Pool(n_processes)
    
# %% MEMORY MAPPING
    border_to_0 = 0 if mc.border_nan == 'copy' else mc.border_to_0
    # you can include the boundaries of the FOV if you used the 'copy' option
    # during motion correction, although be careful about the components near
    # the boundaries
    
    # memory map the file in order 'C'
    fname_new = cm.save_memmap_join(mc.mmap_file, base_name='memmap_',
                               add_to_mov=border_to_0, dview=dview, n_chunks=1000)  # exclude border
    
    time_mmap = time() - time_start - time_mc
    print('Start Segmentation')
# %% SEGMENTATION
    # create summary images    
    img = mean_image(mc.mmap_file[0], window = 1000, dview=dview)
    img = (img-np.mean(img))/np.std(img)
    Cn = local_correlations_movie_offline(mc.mmap_file[0], fr=fr, window=1500, 
                                          stride=1500, winSize_baseline=400, remove_baseline=True, dview=dview).max(axis=0)
    img_corr = (Cn-np.mean(Cn))/np.std(Cn)
    summary_image = np.stack([img, img, img_corr], axis=2).astype(np.float32) 

    #%% three methods for segmentation
    methods_list = ['manual_annotation',        # manual annotation needs user to prepare annotated datasets same format as demo ROIs 
                    'quick_annotation',         # quick annotation annotates data with simple interface in python
                    'maskrcnn' ]                # maskrcnn is a convolutional network trained for finding neurons using summary images
    method = methods_list[0]
    if method == 'manual_annotation':                
        with h5py.File(path_ROIs, 'r') as fl:
            ROIs = fl['mov'][()]  # load ROIs

    elif method == 'quick_annotation': 
        ROIs = quick_annotation(img_corr, min_radius=4, max_radius=10)

    elif method == 'maskrcnn':
        config = neurons.NeuronsConfig()
        class InferenceConfig(config.__class__):
            # Run detection on one image at a time
            GPU_COUNT = 1
            IMAGES_PER_GPU = 1
            DETECTION_MIN_CONFIDENCE = 0.7
            IMAGE_RESIZE_MODE = "pad64"
            IMAGE_MAX_DIM = 512
            RPN_NMS_THRESHOLD = 0.7
            POST_NMS_ROIS_INFERENCE = 1000
        config = InferenceConfig()
        config.display()
        model_dir = os.path.join(caiman_datadir(), 'model')
        DEVICE = "/cpu:0"  # /cpu:0 or /gpu:0
        with tf.device(DEVICE):
            model = modellib.MaskRCNN(mode="inference", model_dir=model_dir,
                                      config=config)
        weights_path = download_model('mask_rcnn')
        model.load_weights(weights_path, by_name=True)
        results = model.detect([summary_image], verbose=1)
        r = results[0]
        ROIs = r['masks'].transpose([2, 0, 1])

        display_result = False
        if display_result:
            _, ax = plt.subplots(1,1, figsize=(16,16))
            visualize.display_instances(summary_image, r['rois'], r['masks'], r['class_ids'], 
                                    ['BG', 'neurons'], r['scores'], ax=ax,
                                    title="Predictions")

    time_seg = time() - time_mmap - time_mc - time_start
    print('Start SPIKE EXTRACTION')

# %% restart cluster to clean up memory
    dview.terminate()
    dview = Pool(n_processes, maxtasksperchild=1)

# %% parameters for trace denoising and spike extraction
    fnames = fname_new                            # change file
    ROIs = ROIs                                   # region of interests
    index = list(range(len(ROIs)))                 # index of neurons
    weights = None                                # reuse spatial weights 

    tau_lp = 5                                    # parameter for high-pass filter to remove photobleaching
    threshold = 4                                 # threshold for finding spikes, increase threshold to find less spikes
    contextSize = 35                              # number of pixels surrounding the ROI to censor from the background PCA
    flip_signal = True                            # Important! Flip signal or not, True for Voltron indicator, False for others

    opts_dict={'fnames': fnames,
               'ROIs': ROIs,
               'index': index,
               'weights': weights,
               'tau_lp': tau_lp,
               'threshold': threshold,
               'contextSize': contextSize,
               'flip_signal': flip_signal}

    opts.change_params(params_dict=opts_dict);          

#%% Trace Denoising and Spike Extraction
    vpy = VOLPY(n_processes=n_processes, dview=dview, params=opts)
    vpy.fit(n_processes=n_processes, dview=dview)
    
    # %% STOP CLUSTER and clean up log files
    #dview.terminate()
    log_files = glob.glob('*_LOG_*')
    for log_file in log_files:
        os.remove(log_file)
        
    time_ext = time() - time_mmap - time_mc - time_start - time_seg
    
    #%%
    print('file:'+fnames)
    print('number of processes'+str(n_processes))
    print(time_mc)
    print(time_mmap)
    print(time_seg)
    print(time_ext)
    time_list = [time_mc, time_mmap, time_seg, time_ext]
    
    return time_list
    
    
#%%
if __name__ == '__main__':
    import numpy as np
    from memory_profiler import memory_usage
    time_all = []
    results = {}
    fnames = '/home/nel/data/voltage_data/volpy_paper/memory/403106_3min_40000.hdf5'
    #fnames = '/home/nel/data/voltage_data/volpy_paper/memory/403106_3min_36000.hdf5'
    path_ROIs = '/home/nel/data/voltage_data/volpy_paper/memory/ROIs.hdf5'
    """
    n_processes = 8
        fnames_list = ['/home/nel/data/voltage_data/volpy_paper/memory/403106_3min_10000.hdf5',
                   '/home/nel/data/voltage_data/volpy_paper/memory/403106_3min_20000.hdf5',
                   '/home/nel/data/voltage_data/volpy_paper/memory/403106_3min_40000.hdf5']
    
    for fnames in fnames_list:
        time_list = test_computational_performance(fnames=fnames, path_ROIs=path_ROIs, n_processes=n_processes)
        time_all.append(time_list)
    np.savez('/home/nel/Code/NEL_LAB/volpy/figures/figure3_performance/time_frames.npz', time_all)
    """
    n_procs=[1] 
    for n_proc in n_procs:
        results['%dprocess' % n_proc] = [memory_usage(
            proc=lambda: test_computational_performance(
                    fnames=fnames, path_ROIs=path_ROIs, n_processes=n_proc), 
                    include_children=True, retval=True)]

    np.savez('/home/nel/Code/NEL_LAB/volpy/figures/figure3_performance/time_memory_proc_test{}.npz'.format(n_procs[0]), results)

#%% Visualization
# T vs frames

import numpy as np
import matplotlib
matplotlib.rcParams['pdf.fonttype'] = 42
matplotlib.rcParams['ps.fonttype'] = 42
import matplotlib.pyplot as plt

m = np.load('/home/nel/Code/NEL_LAB/volpy/figures/figure3_performance/time_frames.npz',
             allow_pickle=True)['arr_0']

plt.figure(figsize=(4, 4))
size = np.array([1, 2, 4])
plt.title('Processing time allocation')
plt.bar((size), (m[:,0]), width=0.5, bottom=0)
plt.bar((size), (m[:,1]), width=0.5, bottom=(m[:,0]))
plt.bar((size), (m[:,2]), width=0.5, bottom=(m[:,0] + m[:,1]))
plt.bar((size), (m[:,3]), width=0.5, bottom=(m[:,0] + m[:,1] + m[:,2]))
plt.legend(['motion corection', 'mem mapping', 'segmentation','spike extraction'],frameon=False)
plt.xlabel('frames (10^4)')
plt.ylabel('time (seconds)')

ax = plt.gca()
ax.locator_params(nbins=7)
ax.spines['right'].set_visible(False)
ax.spines['top'].set_visible(False)

plt.xticks(size, [str(int(i)) for i in size])
plt.tight_layout()
plt.savefig('/home/nel/data/voltage_data/volpy_paper/memory/time&size.pdf', bbox_inches='tight')    
    
#%% T vs cpu
memory = []
time = []
n_procs = [1, 2, 4, 8]
for n_proc in n_procs:
    mm = np.load('/home/nel/Code/NEL_LAB/volpy/figures/figure3_performance/time_memory_proc{}.npz'.format(n_proc),
                allow_pickle=True)['arr_0'].item()
    memory.append(max(mm['%dprocess'% n_proc][0][0]))
    time.append(mm['%dprocess'% n_proc][0][1])
    
time=np.array(time)
    
plt.figure(figsize=(4, 4))
plt.title('parallelization speed')
plt.bar((n_procs), (time[:,0]), width=0.5, bottom=0)
plt.bar((n_procs), (time[:,1]), width=0.5, bottom=(time[:,0]))
plt.bar((n_procs), (time[:,2]), width=0.5, bottom=(time[:,0] + time[:,1]))
plt.bar((n_procs), (time[:,3]), width=0.5, bottom=(time[:,0] + time[:,1] + time[:,2]))
plt.legend(['motion corection', 'memory mapping', 'segmentation','spike extraction'],frameon=False)
plt.xlabel('number of processors')
plt.ylabel('time (seconds)')

ax = plt.gca()
ax.locator_params(nbins=7)
ax.spines['right'].set_visible(False)
ax.spines['top'].set_visible(False)
plt.xticks(n_procs, [str(n_procs[i]) for i in range(len(n_procs))])
plt.savefig('/home/nel/data/voltage_data/volpy_paper/memory/time_cpu_{}.pdf'.format(max(n_procs)), bbox_inches='tight')


#%% memory vs cpu
mem = [round(m/1024, 2) for m in memory]
plt.figure(figsize=(4, 4))
plt.title('peak memory')
plt.scatter(n_procs, mem, color='orange',linewidth=5)
plt.xlabel('number of processors')
plt.ylabel('peak memory (GB)')

ax = plt.gca()
ax.spines['right'].set_visible(False)
ax.spines['top'].set_visible(False)
plt.xticks(n_procs, [str(n_procs[i]) for i in range(len(n_procs))])
plt.savefig('/home/nel/data/voltage_data/volpy_paper/memory/memory_cpu_{}.pdf'.format(max(n_procs)), bbox_inches='tight')
