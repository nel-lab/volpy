#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Mar 23 19:13:18 2020

@author: nel
"""

from memory_profiler import profile

@profile
def test(n_processes=1):
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
    
    
    def mrcnn():
        summary_image = np.load('/home/nel/data/voltage_data/volpy_paper/memory/summary.npz')['arr_0']
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
        return ROIs
    
    import multiprocessing
    p = multiprocessing.Process(target=mrcnn)
    p.start()
    p.join()
    return p
    
if __name__ == '__main__':
    from memory_profiler import memory_usage
    results={}
    for i in range(5):
        results['%d' % i] = [memory_usage(
            proc=lambda: test(n_processes=16), include_children=True, retval=False)]
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    