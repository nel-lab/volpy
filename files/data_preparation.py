#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Aug 20 15:21:05 2019

@author: Changjia
"""
# Folder information of preparing files for UNet
import matplotlib.pyplot as plt
import caiman as cm
import numpy as np
import cv2
import scipy.io as io
import os
import cv2 as cv


#%% Save image from .npz to .png file
# If height/width of the image is less than 128, pad it to 128 
import os
import numpy as np
import cv2
img_dir = '/home/nel/data/voltage_data/volpy_paper/npz'
save_dir = '/home/nel/data/voltage_data/volpy_paper/img'
fls = os.listdir(img_dir)

a = []
for fi in fls:
    path = os.path.join(img_dir, fi)
    if 'mask' not in path:
        m = np.load(path)['arr_0']
        for i in range(3):
            m[:,:,i] = (m[:,:,i]-np.min(m[:,:,i]))/(np.max(m[:,:,i]-np.min(m[:,:,i])))*255
        m = m.astype(np.uint8)
        height, width = m.shape[:2]
        if height < 128:
            m = cv2.copyMakeBorder(m, 0, 
                 np.int(128-height), 
                 0, 
                 0, 
                 cv2.BORDER_CONSTANT, 
                 value=(0,0,0))
        if width < 128:
            m = cv2.copyMakeBorder(m, 0, 
                 0, 
                 0, 
                 np.int(128-width), 
                 cv2.BORDER_CONSTANT, 
                 value=(0,0,0))
        cv2.imwrite(os.path.join(save_dir,fi)[:-4]+'.png', m)
        print(np.mean(m, axis=(0,1)))
        a.append(np.mean(m, axis=(0,1)))   
        
#%% Save mask from .zip to json
from caiman.base.rois import nf_read_roi_zip
from caiman.base.rois import nf_masks_to_json
img_dir = '/home/nel/data/voltage_data/volpy_paper/img'
zip_dir = '/home/nel/data/voltage_data/volpy_paper/zip'
mask_dir = '/home/nel/data/voltage_data/volpy_paper/mask'
fls = os.listdir(img_dir)

for fi in fls:
    img_path = os.path.join(img_dir, fi)
    dims = cv2.imread(img_path).shape[:2]
    zip_path = os.path.join(zip_dir, fi[:-4]+'.zip')  
    masks = nf_read_roi_zip(zip_path, dims=dims)
    mask_path = os.path.join(mask_dir, fi[:-4]+'.json')
    nf_masks_to_json(masks, os.path.join(mask_path))
    
#%% Zip to json


dr = '/home/nel/data/voltage_data/volpy_paper/img'
dr_mask = '/home/nel/data/voltage_data/volpy_paper/mask'
files = sorted(os.listdir(dr))
files = [file for file in files if '.npz' in file]

for file in files:
    m = np.load(os.path.join(dr, file), allow_pickle=True)['arr_0']
    dims = m.shape[:2]
    mask = nf_read_roi_zip(os.path.join(dr_mask, file[:-4]+'.zip'), dims=dims)
    nf_masks_to_json(mask, os.path.join(dr_mask, file[:-4]+'.json'))















#%%
file = '/home/nel/Code/VolPy/Mask_RCNN/videos & imgs/neurons_mc/IVQ29_S5_FOV4_d1_164_d2_96_d3_1_order_C_frames_20000_.mmap'
file='/home/nel/Code/VolPy/Mask_RCNN/videos & imgs/neurons_mc/IVQ29_S5_FOV6_d1_228_d2_96_d3_1_order_C_frames_20000_.mmap'
file = '/home/nel/Code/VolPy/Mask_RCNN/videos & imgs/neurons_mc/IVQ48_S7_FOV5_d1_212_d2_96_d3_1_order_C_frames_20000_.mmap'
file =  '/home/nel/Code/VolPy/Mask_RCNN/videos & imgs/test/memmap__d1_360_d2_256_d3_1_order_C_frames_10000_.mmap'
m = cm.load(file, in_memory=True, fr=400)
m = m[100:,:,:]
img = np.mean(m,axis=0)
plt.plot(m.mean(axis=(1,2)))
#%%
ma = m.computeDFF(secsWindow=1)[0]
plt.plot(ma.mean(axis=(1,2)))
#%%
Cn = ma.copy().gaussian_blur_2D().local_correlations(swap_dim=False, eight_neighbours=True, frames_per_chunk=1000000000)
#%% 
plt.figure()
plt.subplot(1,2,1)
plt.imshow(Cn, cmap='gray', vmax=0.2)
plt.subplot(1,2,2)
plt.imshow(img,cmap='gray')


#%%
import os
#file = '/home/nel/Code/VolPy/Mask_RCNN/videos & imgs/neurons_mc/06152017Fish1-2_d1_364_d2_320_d3_1_order_C_frames_10000_.mmap'
dr = '/home/nel/Code/VolPy/Mask_RCNN/videos & imgs/neurons_mc/'
ds_list = sorted(os.listdir(dr))
l = ['_'.join(dd.split('_')[:-11]) for dd in ds_list]
#name = os.path.basename(file)
#dr1 = '/home/nel/Code/VolPy/Mask_RCNN/videos & imgs/neurons'
#d1 = sorted(os.listdir(dr1))
#d2 = [i for i in d1 if 'hdf5' in i]

r = np.arange(0,24,1)
for i in r:
    file = dr + ds_list[i]
    m = cm.load(file, in_memory=True, fr=400)
    #plt.plot(m.mean(axis=(1,2)))
    #%%
    m = m[100:,:,:]
    #plt.plot(m.mean(axis=(1,2)))
    img = m.mean(axis=0)
    #%%
    ma = m.computeDFF(secsWindow=1)[0]
    #plt.plot(ma.mean(axis=(1,2)))
    
    #%%
    Cn = ma.local_correlations(swap_dim=False, eight_neighbours=True, frames_per_chunk=1000000000)
    #%%
    """
    plt.subplot(1,2,1)
    plt.imshow(Cn, cmap='gray')
    plt.subplot(1,2,2)
    plt.imshow(img,cmap='gray')
    """
    Cn = Cn[np.newaxis,:,:]
    cm.movie(Cn).save('/home/nel/Code/VolPy/Mask_RCNN/videos & imgs/test/'+'aaa'+'_lci.tif')
    #%%
    from caiman.summary_images import local_correlations_movie
    lcm = local_correlations_movie(ma)
    lcm.save('/home/nel/Code/VolPy/Mask_RCNN/videos & imgs/'+l[i]+'_lcm.tif')
#%%
lcm = local_correlations_movie(ma[:2000])
#%%
# Johaness data
# Original data
    dr = '/home/nel/Code/Voltage_imaging/exampledata/Johannes/data/'
    ds_list = ['06152017Fish1-2', '10192017Fish2-1','10192017Fish3-1']
    ds = ds_list[2]
    fnames = dr + ds + '/registered.tif'
    m = cm.load(fnames,subindices=slice(2000,12000))
# Read
    dr = '/home/nel/Code/Voltage_imaging/exampledata/Johannes/data/'
    ds_list = ['06152017Fish1-2', '10192017Fish2-1','10192017Fish3-1']
    ds = ds_list[]
    #fnames = dr + ds + '/registered.tif'
    #m = cm.load(fnames,subindices=slice(0, 10000))
    #m.save(dr + ds + '/' + ds + '.hdf5')    
    fnames = dr + ds + '/' + ds + '.hdf5'
    m = cm.load(fnames)
    images = m
# Write
    npz_dir = '/home/nel/Code/VolPy/UNet/npz/Johannes/'
    sname = npz_dir+ds+'.npz'
    np.savez(sname, img_2c)
    
# Save to tif file for ImageJ
    sname = npz_dir+ds+'.npz'
    X = np.load(sname)['arr_0']
    iname = npz_dir+ds+'.tif'
    cm.movie(X).transpose([2,0,1]).save(iname)
    
# Plot two figures
    plt.figure();plt.imshow(X[:,:,0])
    plt.figure();plt.imshow(X[:,:,1])

# Johannes's ROI
    roi_dir = '/home/nel/Code/Voltage_imaging/exampledata/Johannes/data/'
    rname = roi_dir+ds+'/ROI_info.npy'
    y = np.load(rname, allow_pickle=True)
    dims = X.shape
    img = np.zeros((dims[0], dims[1]))
    for i in range(len(y)):
        img[y[i]['pixel_yx_list'][:,0],y[i]['pixel_yx_list'][:,1]] = 1
    plt.figure();plt.imshow(img)
    
# Read my ROI
    dims = (508, 288)#(360, 256)#(364,320) 
    roi_dir = '/home/nel/Code/VolPy/UNet/ROIs/Johannes'
    rname = roi_dir+ds+'_RoiSet.zip'
#    from caiman.base.rois import nf_read_roi_zip
    img = nf_read_roi_zip(rname,dims)
    plt.figure();plt.imshow(img.sum(axis=0))
    
# Form training data
    npz_dir = '/home/nel/Code/VolPy/UNet/npz/Johannes/'
    X_j = []
    dimension = []
    for index, file in enumerate(sorted(os.listdir(npz_dir))):
        if file[-3:] == 'npz':
            temp = np.load(npz_dir+ file)['arr_0']
            dimension.append([temp.shape[0], temp.shape[1]])
            temp = Mirror(temp)
            X_j.append(temp)
    X_j = np.array(X_j)

    roi_dir = '/home/nel/Code/VolPy/UNet/ROIs/Johannes/'    
    Y_j = []
    for index, file in enumerate(sorted(os.listdir(roi_dir))):
        print(file)
        from caiman.base.rois import nf_read_roi_zip
        name = roi_dir + file
        img = nf_read_roi_zip(name,dims=dimension[index]).sum(axis=0)
        print(img.shape)
        img = Mirror(img)
        Y_j.append(img)
        
    Y_j = np.array(Y_j)
    Y_j[Y_j>0] = 1
    Y_j = Y_j[:,:,:,np.newaxis]            
    
    


    
#%%
###############################################################################
# Adam's data
# Read hdf5/mmap file after motion correction
    import os
    dr = '/home/nel/Code/Voltage_imaging/exampledata/toshare_CA1/Data'
    #ds_list = [i[:-5] for i in os.listdir(dr) if '.hdf5' in i]
    #ds_list = ['IVQ32_S2_FOV1', 'IVQ38_S2_FOV3', 'IVQ48_S7_FOV7','IVQ38_S1_FOV5',
    #           'IVQ48_S7_FOV5', 'IVQ48_S7_FOV8', 'IVQ29_S5_FOV4','IVQ29_S5_FOV6']
    ds_list = [i[:-5] for i in os.listdir(dr) if 'order_F' in i]
    ds = ds_list[8]
    fnames = dr +  '/' + ds + '.mmap'
    m = cm.load(fnames)[2000:]
    
# Write
    npz_dir = '/home/nel/Code/VolPy/UNet/npz/Adam/'
    sname = npz_dir+ds+'.npz'
    np.savez(sname, img_2c)
    
# Save to tif file for ImageJ
    npz_dir = '/home/nel/Code/Voltage_imaging/exampledata/toshare_CA1_2/Mean_Corr_img/'
    ds_list = [i[:-4] for i in os.listdir(npz_dir)]
for ds in ds_list:
    sname = npz_dir+ds+'.npz'
    X = np.load(sname)['arr_0']
    iname = npz_dir+ds+'.tif'
    cm.movie(X).transpose([2,0,1]).save(iname)

# Read my ROIs
    dims = (280,96)#(164,96)#(128,88)#(284,96)#(212,96)#(176,92)#(256, 96) 
    roi_dir = '/home/nel/Code/VolPy/UNet/ROIs/Adam/'
    rname = roi_dir+ds[:13]+'_RoiSet.zip'
    from caiman.base.rois import nf_read_roi_zip
    img = nf_read_roi_zip(rname,dims)
    plt.figure();plt.imshow(img.sum(axis=0))
    
    
#%% change to one channel
    import os
    import numpy as np
    dr = '/home/nel/Code/VolPy/Mask_RCNN//backup/All _more_chosen/'
    dr = '/home/nel/Code/VolPy/Mask_RCNN/consensus/'
    
    ds_list = sorted(os.listdir(dr))
    for i in ds_list:
        if 'npz' in i:
            X = np.load(dr+i)['arr_0']
            np.savez('/home/nel/Code/VolPy/Mask_RCNN//backup/onechannel/'+i,X[:,:,0])
    
    ds_list=ds_list[6:23]
    for i in ds_list:
        if 'zip' in i:
            import zipfile
            from caiman.base.rois import nf_read_roi
            with zipfile.ZipFile('/home/nel/Code/VolPy/Mask_RCNN/datasets/IVQ29_S5_FOV4.zip') as zf:
                names = zf.namelist()
                coords = [nf_read_roi(zf.open(n))
                          for n in names]
                polygons = [{'name': 'polygon','all_points_x':i[:,1],'all_points_y':i[:,0]} for i in coords]
            np.savez('/home/nel/Code/VolPy/Mask_RCNN/datasets/IVQ29_S5_FOV4'+'_mask'+'.npz', polygons)
            

#%%
    dr = '/home/nel/Code/VolPy/Mask_RCNN//backup/All/'
    ds_list = sorted(os.listdir(dr))
    for i in ds_list:
        if 'npz' in i:
            X = np.load(dr+i)['arr_0']
            plt.imshow(X)



    
#%%
    import skimage
    dr = '/home/nel/Code/VolPy/Mask_RCNN/backup/onechannel_squeeze/'
    ds_list = sorted(os.listdir(dr))
    ds_set = [i for i in ds_list if 'mask' not in i]
    
    for i in ds_list:
        if 'mask' not in i:
            print(i)
            X = np.load(dr+i)['arr_0']
            plt.figure();plt.imshow(X)
            polygons = np.load(dr+i[:-4]+'_mask.npz', allow_pickle=True)['arr_0']
            dims = X.shape
            mask = np.zeros((dims[0],dims[1]))
            for j in range(len(polygons)):
            # Get indexes of pixels inside the polygon and set them to 1
                p = polygons[j]
                rr, cc = skimage.draw.polygon(p['all_points_y'], p['all_points_x'])
                mask[rr, cc] = 1
            plt.figure();plt.imshow(mask)
        
        polygons = np.load(dr+i+'_mask.npz', allow_pickle=True)['arr_0']
 
    dr = '/home/nel/Code/VolPy/Mask_RCNN/backup/onechannel/'
    ds_list = sorted(os.listdir(dr))
    ds_set = [i for i in ds_list if 'mask' not in i]
    #ds_set = set([i[:-4] for i in ds_list])
    for i in ds_set:
        if 'IVQ' in i:
            X = np.load(dr+i)['arr_0']
            X_new = squeeze4(squeeze4(X))
            np.savez('/home/nel/Code/VolPy/Mask_RCNN/backup/onechannel_squeeze/'+i[:-4], X_new)
            dims = X.shape
            polygons = np.load(dr+i[:-4]+'_mask.npz', allow_pickle=True)['arr_0']
            polygons_new=squeeze4mask(squeeze4mask(polygons,dims), dims)
            np.savez('/home/nel/Code/VolPy/Mask_RCNN/backup/onechannel_squeeze/'+i[:-4]+'_mask', polygons_new)
            



    import skimage
    mask = np.zeros((dims[0], dims[1]))
    for i, p in enumerate(polygons_new):
        # Get indexes of pixels inside the polygon and set them to 1
        rr, cc = skimage.draw.polygon(p['all_points_y'], p['all_points_x'])
        mask[rr, cc] = 1
    def squeeze4(X):
        X1 = np.flip(X, axis=1)
        X2 = np.flip(X, axis=0)
        X3 = np.flip(X2, axis=1)
        X_new = np.concatenate((np.concatenate((X,X1),axis=1),np.concatenate((X2,X3),axis=1)),axis=0)
        X_new = cv.resize(X_new,(X.shape[1], X.shape[0]), interpolation = cv.INTER_CUBIC)
        return X_new
    
    def squeeze4mask(polygons,dims):
        polygons0 = []
        polygons1 = []
        polygons2 = []
        polygons3 = []
        for p in polygons:
            p0 ={}
            p0['name'] = 'polygon'
            p0['all_points_x'] = np.floor(p['all_points_x']/2).astype(np.int16)
            p0['all_points_y'] = np.floor(p['all_points_y']/2).astype(np.int16)
            polygons0.append(p0)
            p1 ={}
            p1['name'] = 'polygon'
            p1['all_points_x'] = np.floor((2*dims[1] - p['all_points_x'])/2).astype(np.int16)
            p1['all_points_y'] = np.floor(p['all_points_y']/2).astype(np.int16)
            polygons1.append(p1)
            p2 ={}
            p2['name'] = 'polygon'
            p2['all_points_x'] = np.floor(p['all_points_x']/2).astype(np.int16)
            p2['all_points_y'] = np.floor((2*dims[0] - p['all_points_y'])/2).astype(np.int16)
            polygons2.append(p2)
            p3 ={}
            p3['name'] = 'polygon'
            p3['all_points_x'] = np.floor((2*dims[1] - p['all_points_x'])/2).astype(np.int16)
            p3['all_points_y'] = np.floor((2*dims[0] - p['all_points_y'])/2).astype(np.int16)
            polygons3.append(p3)
        polygons_new = polygons0 + polygons1 + polygons2 + polygons3
        return polygons_new
    
    
  
    
#%%
###############################################################################
# Kaspar's Data
    import numpy as np
    import matplotlib.pyplot as plt
    npz_dir = '/home/nel/Code/VolPy/UNet/npz/Kaspar/'
    X = []
    import os
    for index, file in enumerate(sorted(os.listdir(npz_dir))):
        temp = np.load(npz_dir + file)['arr_0']
        #X.append(Mirror(temp))
        X.append(temp)

#    for index, file in enumerate(sorted(os.listdir(npz_dir))):
#        cm.movie(X[index]).save(npz_dir+'/'+file[:-4]+'.tif')
    X = np.array(X)
    
    fig,ax = plt.subplots(1,X.shape[0])
    for i in range(X.shape[0]):
        ax[i].imshow(X[i,:,:,0])
    
# ROIs
    import scipy.io as io
    roi_dir = '/home/nel/Code/VolPy/UNet/ROIs/Kaspar/'
    Y = []
    for index, file in enumerate(sorted(os.listdir(roi_dir))):
        print(file)
        from caiman.base.rois import nf_read_roi_zip
        name = roi_dir + file
        img = nf_read_roi_zip(name,dims=(512,128)).sum(axis=0)
        #Y.append(Mirror(img))
        Y.append(img)
        
    Y = np.array(Y)
    Y[Y>0] = 1
    Y = Y[:,:,:,np.newaxis]

    fig,ax = plt.subplots(1,len(Y))
    for i in range(Y.shape[0]):
        ax[i].imshow(Y[i,:,:,0])
        
    X = X[np.array([0,1,2,3,5,6,4,7]),:,:,:]    
    Y = Y[np.array([0,2,1,4,6,5,3,7]),:,:,:]  
        
    Xtrain = np.concatenate([X[np.array([0,1,2,3,4,5]),:,:,:],X_j[np.array([0,1])]])
    Ytrain = np.concatenate([Y[np.array([0,1,2,3,4,5]),:,:,:], Y_j[np.array([0,1])]])
    Xtest = np.concatenate([X[[6,7],:,:,:], X_j[np.array([2])]])
    Ytest = np.concatenate([Y[[6,7],:,:,:], Y_j[np.array([2])]])
    
    Xtrain = X[np.array([0,1,2,3,4,5]),:,:,:]
    Ytrain = Y[np.array([0,1,2,3,4,5]),:,:,:]
    Xtest = X[[6,7],:,:,:]
    Ytest = Y[[6,7],:,:,:]
    
    
    
    
#%%
    def Mirror(temp,size=512):
        shape = temp.shape
        temp = cv2.copyMakeBorder(temp, size-shape[0], 0, 0, size-shape[1], cv2.BORDER_REFLECT)
        return temp
    
    
    
#%%
    import os
    import caiman as cm
    import numpy as np
    dr1 = '/home/nel/Code/VolPy/Mask_RCNN/videos & imgs/summary imgs/mean_npz/'
    dr2 = '/home/nel/Code/VolPy/Mask_RCNN/videos & imgs/summary imgs/corr/'

    ds1 = sorted(os.listdir(dr1))
    ds2 = sorted(os.listdir(dr2))
    
    for i in ds1:
        name1 = dr1+i
        m1 = np.load(name1)['arr_0']
        m1 = (m1-np.mean(m1))/np.std(m1)
        
        name2 = dr2+i[:-4]+'_lci.tif'
        m2 = cm.load(name2)
        m2 = (m2-np.mean(m2))/np.std(m2)
        
        name = '/home/nel/Code/VolPy/Mask_RCNN/videos & imgs/npz_2channel/' + i[:-4] + '.npz'
        if m1.shape[0] == m2.shape[0]:
            m = np.stack([m1,m1,m2],axis=2)
            np.savez(name, m)
        else:
            m = np.stack([m1,m1,m2.transpose([1,0])],axis=2)
            np.savez(name, m)
    
    
#%% Calcium data
import caiman as cm
import json
import os
root = '/home/nel/Code/VolPy/Mask_RCNN/Calcium_data/WEBSITE'
dirs = [di for di in os.listdir(root) if os.path.isdir(os.path.join(root,di))]

for di in dirs:
    """
    path0 = os.path.join(root,di,'projections/median_image.tif')
    m0 = cm.load(path0)    
    m0[m0<np.percentile(m0,5)] = np.percentile(m0,5) 
    m0[m0>np.percentile(m0,99)] = np.percentile(m0,99) 
    m0 = (m0-np.mean(m0))/np.std(m0)    
    path1 = os.path.join(root,di,'projections/correlation_image.tif')
    #print(path)
    m1 = np.array(cm.load(path1))
    m1 = (m1-np.mean(m1))/np.std(m1)
    dims = m1.shape
    #print(dims)
    m_stack = np.stack([m0,m0,m1],axis=2)
    np.savez(os.path.join('/home/nel/Code/VolPy/Mask_RCNN/Calcium_data/calcium', di), m_stack)
    #regions_filename = os.path.join(root, di, 'regions/consensus_regions.json' )
    #regions_filename = os.path.join(root, di, 'regions/L1__regions.json' )
    #regions_filename = os.path.join(root, di, 'regions/L2__regions.json' )
    #regions_filename = os.path.join(root, di, 'regions/L3__regions.json' )
    """
    regions_filename = os.path.join(root, di, 'regions/L4__regions.json' )
    
    with open(regions_filename) as f:
        regions = json.load(f)
    
    """
    def tomask(coords):
        mask = np.zeros(dims)
        for coor in coords:
            mask[coor[0], coor[1]] = 1
        return mask

    masks = np.array([tomask(s['coordinates']) for s in regions]).astype(np.uint8)
    print(di)
    print(masks.shape)
    """
    #plt.figure()
    #plt.imshow(masks.sum(axis=0))
    rr = [{'all_points_x':np.array(regions[i]['coordinates'])[:,1],'all_points_y':np.array(regions[i]['coordinates'])[:,0]} for i in range(len(regions))]
    np.savez(os.path.join('/home/nel/Code/VolPy/detectron2/datasets', di)+'_mask', rr)

#%%
import os
import numpy as np
import matplotlib.pyplot as plt
import skimage
import matplotlib.pyplot as plt
fls = os.listdir('/home/nel/Code/VolPy/Mask_RCNN/datasets/neurons_calcium/other')

for fi in fls:
    path = os.path.join('/home/nel/Code/VolPy/Mask_RCNN/datasets/neurons_calcium/other',fi)
    if '_mask' not in path:
        print(path)
        m = np.load(path)['arr_0']
        plt.figure()
        plt.imshow(m[:,:,2])
        np.savez(os.path.join('/home/nel/Code/VolPy/Mask_RCNN/datasets/neurons_calcium',fi),m[:,:,2])
        dims = (m.shape[0],m.shape[1])
        
        path_roi = path[:-4]+'_mask.npz'
        
        polygons = np.load(path_roi,allow_pickle=True)['arr_0']
        mask = np.zeros([dims[0],dims[1], len(polygons)],
                        dtype=np.uint8)
        for i, p in enumerate(polygons):
            # Get indexes of pixels inside the polygon and set them to 1
            rr, cc = skimage.draw.polygon(p['all_points_y'], p['all_points_x'])
            mask[rr, cc, i] = 1
        
        mask = mask.transpose([2,0,1])
        np.savez('/home/nel/Code/VolPy/Mask_RCNN/datasets/neurons_calcium/'+fi[:-4]+'_mask.npz',mask)

#%%
import os
import numpy as np
import cv2
img_dir = '/home/nel/Code/VolPy/detectron2/datasets/calcium/train'
save_dir = '/home/nel/Code/VolPy/detectron2/datasets/neurons_calcium/train'
fls = os.listdir(img_dir)

a = []
for fi in fls:
    path = os.path.join(img_dir, fi)
    if 'mask' not in path:
        m = np.load(path)['arr_0']
        for i in range(3):
            m[:,:,i] = (m[:,:,i]-np.min(m[:,:,i]))/(np.max(m[:,:,i]-np.min(m[:,:,i])))*255
        m = m.astype(np.uint8)
        height, width = m.shape[:2]
        if height < 128:
            m = cv2.copyMakeBorder(m, 0, 
                 np.int(128-height), 
                 0, 
                 0, 
                 cv2.BORDER_CONSTANT, 
                 value=(0,0,0))
        if width < 128:
            m = cv2.copyMakeBorder(m, 0, 
                 0, 
                 0, 
                 np.int(128-width), 
                 cv2.BORDER_CONSTANT, 
                 value=(0,0,0))
        cv2.imwrite(os.path.join(save_dir,fi)[:-4]+'.png', m)
        print(np.mean(m, axis=(0,1)))
        a.append(np.mean(m, axis=(0,1)))   
        
#%%
import os
import numpy as np
import cv2
img_dir = '/home/nel/Code/VolPy/detectron2/datasets/neurons_calcium/val'
save_dir = '/home/nel/Code/VolPy/detectron2/datasets/neurons_calcium_corr/val'
fls = os.listdir(img_dir)

a = []
for fi in fls:
    path = os.path.join(img_dir, fi)
    if 'mask' not in path:
        m = cv2.imread(path)
        m = np.dstack([m[:, :, 2]] * 3)
        cv2.imwrite(os.path.join(save_dir,fi)[:-4]+'.png', m)
        print(np.mean(m, axis=(0,1)))
        a.append(np.mean(m, axis=(0,1)))   
        print(np.array(a).mean(axis=0))
    
#%% moving json file
#import caiman as cm
import json
import os
root_dir = '/home/nel/data/calcium_data/caiman_paper'
dirs = [di for di in os.listdir(root_dir) if os.path.isdir(os.path.join(root_dir,di))]
save_dir = '/home/nel/data/calcium_data/detectron2/neurons_calcium_corr'

for di in dirs:
    file_name = os.path.join(root_dir, di, 'regions/L4__regions.json' )
    json_filename = os.path.join(save_dir, di) + '.json'

    with open(file_name) as f:
            regions = json.load(f)    
    
    with open(json_filename, 'w') as f:
        f.write(json.dumps(regions))
    
    



