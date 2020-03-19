#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Mar 19 10:19:31 2020

@author: caichangjia
"""

from memory_profiler import profile
    
@profile
def my_func():
    a = 1 + 1
    """
    import caiman as cm
    import numpy as np
    fr=400
    fname = '/home/nel/caiman_data/example_movies/volpy/demo_voltage_imaging._rig__d1_100_d2_100_d3_1_order_F_frames_20000_.mmap'
    m = cm.load(fname, subindices=slice(0, 20000))[1000:]
    m.fr = fr
    img = m.mean(axis=0)
    img = (img-np.mean(img))/np.std(img)
    m1 = m.computeDFF(secsWindow=1, in_place=True)[0]
    #m = m.removeBL(400, returnBL=False, in_place=True)
    m = m - m1
    Cn = m.local_correlations(swap_dim=False, eight_neighbours=True)
    img_corr = (Cn-np.mean(Cn))/np.std(Cn)
    summary_image = np.stack([img, img, img_corr], axis=2).astype(np.float32)   
    """
if __name__ == '__main__':
    my_func()
    b = 3 + 4
    print(b)
    my_func()
    b = b + b
    print(b)
    my_func()
    
#main(fname, n_processors)
