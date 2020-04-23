#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Apr 13 18:11:23 2020
Supplementary figure 1
@author: caichangjia
"""
m = cm.load('/home/nel/Code/VolPy_many_things/Mask_RCNN/videos & imgs/neurons/FOV4_50um._rig__d1_128_d2_512_d3_1_order_F_frames_20000_.mmap')
corr = cm.load('/home/nel/Code/VolPy_many_things/Mask_RCNN/videos & imgs/summary imgs/corr_video/FOV4_50um_lcm.tif')
m = m.transpose([0,2,1])
corr = corr.transpose([0,2,1])
mm = m[100:8100]
mm = (mm-mm.min())/(mm.max()-mm.min())
corr = corr[:8000]
corr = (corr-corr.min())/(corr.max()-corr.min())

mmm = cm.concatenate((mm,corr), axis=2)
mmm[800:1600].save('/home/nel/data/voltage_data/volpy_paper/figS1/orig_corr_movie.avi')

#ffmpeg -i orig_corr_movie.avi -strict -2 compress.mp4
