#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Created on Mon Aug  3 10:21:47 2020
Comparison among different manual annotators
@author: @caichangjia
"""
import os
import numpy as np
import matplotlib
matplotlib.rcParams['pdf.fonttype'] = '42'
matplotlib.rcParams['ps.fonttype'] = '42'
import matplotlib.pyplot as plt
import caiman as cm
from caiman.base.rois import nf_read_roi_zip
from caiman.base.rois import nf_match_neurons_in_binary_masks

#%% Compare between labelers
comparison = {}

#%%
img_folder = '/home/nel/NEL-LAB Dropbox/NEL/Papers/VolPy/manual_annotations/summary_images/images/'
root_folder = '/home/nel/NEL-LAB Dropbox/NEL/Papers/VolPy/manual_annotations/summary_images/output/'
#combined_folder = '/home/nel/NEL-LAB Dropbox/NEL/Papers/VolPy/manual_annotations/summary_images/combination'
save_img_folder = '/home/nel/NEL-LAB Dropbox/NEL/Papers/VolPy/manual_annotations/summary_images/comparison_v1.2/'
annotators = sorted(os.listdir(root_folder))
#    select = [0, 1]
for select in [[0,1], [0,2], [1,2]]:
    filenames = sorted(os.listdir(root_folder+annotators[select[0]]))
    result = {}
    #filenames1 = sorted(os.listdir(root_folder+annotators[select[1]]))
    
    for file in filenames:
        img = cm.load(os.path.join(img_folder, file[:-4]+'_summary.tif'))[0]
        dims = img.shape
        mask0 = nf_read_roi_zip(os.path.join(root_folder, annotators[select[0]], file), dims=dims) * 1.
        mask1 = nf_read_roi_zip(os.path.join(root_folder, annotators[select[1]], file), dims=dims) * 1.
        
        tp_gt, tp_comp, fn_gt, fp_comp, performance_cons_off = nf_match_neurons_in_binary_masks(
                mask0, mask1, thresh_cost=0.7, min_dist=10, print_assignment=True,
                plot_results=True, Cn=img, labels=[annotators[select[0]], annotators[select[1]]])
        plt.savefig(os.path.join(save_img_folder, annotators[select[0]][0]+'&'+annotators[select[1]][0]+file[:-4]+'.pdf'))
        plt.close()
        result[file] = performance_cons_off
    
    #%%
    processed = {}
    for i in ['f1_score','recall','precision']:
        #result = eval(i)
        processed[i] = {}    
        for j in ['L1','TEG','HPC']:
            if j == 'L1':
                temp = [result[k][i] for k in result.keys() if 'Fish' not in k and 'IVQ' not in k]
                if i == 'number':
                    processed[i]['L1'] = sum(temp)
                else:
                    processed[i]['L1'] = sum(temp)/len(temp)
            if j == 'TEG':
                temp = [result[k][i] for k in result.keys() if 'Fish' in k]
                if i == 'number':
                    processed[i]['TEG'] = sum(temp)
                else:
                    processed[i]['TEG'] = sum(temp)/len(temp)
            if j == 'HPC':
                temp = [result[k][i] for k in result.keys() if 'IVQ' in k]
                if i == 'number':
                    processed[i]['HPC'] = sum(temp)
                else:
                    processed[i]['HPC'] = sum(temp)/len(temp)
    
    #%%
    comparison[annotators[select[0]][0] + '&' + annotators[select[1]][0]] = processed            
    print(comparison)
                    
#%% Show all matches and mismatches 
from caiman.base.rois import norm_nrg
import matplotlib.patches as mpatches
img_folder = '/home/nel/NEL-LAB Dropbox/NEL/Papers/VolPy/manual_annotations/summary_images/images/'
root_folder = '/home/nel/NEL-LAB Dropbox/NEL/Papers/VolPy/manual_annotations/summary_images/output/'
save_img_folder = '/home/nel/NEL-LAB Dropbox/NEL/Papers/VolPy/manual_annotations/summary_images/comparison_tri_v1.2/'
annotators = sorted(os.listdir(root_folder))
filenames = sorted(os.listdir(root_folder+annotators[select[0]]))

for file in filenames:
    c1 = []
    c2 = []
    j1 = []
    j2 = []
    m1 = []
    m2 = []    
    
    for select in [[0,1], [0,2], [1,2]]:
        #result = {}
        #filenames1 = sorted(os.listdir(root_folder+annotators[select[1]]))
    
        img = cm.load(os.path.join(img_folder, file[:-4]+'_summary.tif'))[0]
        dims = img.shape
        mask0 = nf_read_roi_zip(os.path.join(root_folder, annotators[select[0]], file), dims=dims) * 1.
        mask1 = nf_read_roi_zip(os.path.join(root_folder, annotators[select[1]], file), dims=dims) * 1.
        
        tp_gt, tp_comp, fn_gt, fp_comp, performance_cons_off = nf_match_neurons_in_binary_masks(
                mask0, mask1, thresh_cost=0.7, min_dist=10, print_assignment=True,
                plot_results=False, Cn=img, labels=[annotators[select[0]], annotators[select[1]]])
        #plt.savefig(os.path.join(save_img_folder, annotators[select[0]]+'&'+annotators[select[1]]+file[:-4]+'.pdf'))
        #plt.close()

        if select == [0, 1]:
            c_all = mask0.shape[0]
            j_all = mask1.shape[0]
            c1.append(tp_gt)
            j1.append(tp_comp)
        elif select == [0, 2]:
            c1.append(tp_gt)
            m1.append(tp_comp)
            m_all = mask1.shape[0]
        elif select == [1, 2]:
            j1.append(tp_gt)
            m1.append(tp_comp)
            
    c1 = set(c1[0]).union(c1[1])
    c2 = set(range(c_all)).difference(c1)
    j1 = set(j1[0]).union(j1[1])
    j2 = set(range(j_all)).difference(j1)
    m1 = set(m1[0]).union(m1[1])
    m2 = set(range(m_all)).difference(m1)
    
    #plt.imshow(img)
    mask0 = nf_read_roi_zip(os.path.join(root_folder, annotators[0], file), dims=dims) * 1.
    mask1 = nf_read_roi_zip(os.path.join(root_folder, annotators[1], file), dims=dims) * 1.
    mask2 = nf_read_roi_zip(os.path.join(root_folder, annotators[2], file), dims=dims) * 1.
    
    plt.figure()
    plt.subplot(1, 2, 1)
    plt.imshow(img, cmap='gray', vmin=np.percentile(img, 5), vmax=np.percentile(img, 95), alpha=0.5)
    level = .98
    [plt.contour(norm_nrg(mm), levels=[level], colors='r', linewidths=1) for mm in mask0[np.array(list(c1))]]
    [plt.contour(norm_nrg(mm), levels=[level], colors='g', linewidths=1) for mm in mask1[np.array(list(j1))]]
    [plt.contour(norm_nrg(mm), levels=[level], colors='b', linewidths=1) for mm in mask2[np.array(list(m1))]]
    ses_1 = mpatches.Patch(color='red', label='CC&AG')
    ses_2 = mpatches.Patch(color='green', label='JT')
    ses_3 = mpatches.Patch(color='blue', label='MK')            
    plt.legend(handles=[ses_1, ses_2, ses_3])
    plt.show()
    plt.subplot(1, 2, 2)
    plt.imshow(img, cmap='gray', vmin=np.percentile(img, 5), vmax=np.percentile(img, 95), alpha=0.5)
    level = .98
    if len(c2) > 0:
        [plt.contour(norm_nrg(mm), levels=[level], colors='r', linewidths=1) for mm in mask0[np.array(list(c2))]]
    if len(j2) > 0:
        [plt.contour(norm_nrg(mm), levels=[level], colors='g', linewidths=1) for mm in mask1[np.array(list(j2))]]
    if len(m2) > 0:
        [plt.contour(norm_nrg(mm), levels=[level], colors='b', linewidths=1) for mm in mask2[np.array(list(m2))]]
    ses_1 = mpatches.Patch(color='red', label='CC&AG')
    ses_2 = mpatches.Patch(color='green', label='JT')
    ses_3 = mpatches.Patch(color='blue', label='MK')            
    plt.legend(handles=[ses_1, ses_2, ses_3])
    plt.title('Comparison among three labelers')
    plt.show()
    plt.savefig(os.path.join(save_img_folder, file[:-4]+'.pdf'))
    plt.close()
    

#%% Compare with the combined datasets
comparison = {}

#%%
img_folder = '/home/nel/NEL-LAB Dropbox/NEL/Papers/VolPy/manual_annotations/summary_images/images/'
root_folder = '/home/nel/NEL-LAB Dropbox/NEL/Papers/VolPy/manual_annotations/summary_images/output/'
combined_folder = '/home/nel/NEL-LAB Dropbox/NEL/Papers/VolPy/manual_annotations/summary_images/combination_v1.2/'
save_img_folder = '/home/nel/NEL-LAB Dropbox/NEL/Papers/VolPy/manual_annotations/summary_images/comparison_combined_v1.2/'
annotators = sorted(os.listdir(root_folder))
#    select = [0, 1]
for select in [0, 1, 2]:
    filenames = sorted(os.listdir(root_folder+annotators[select]))
    result = {}
    #filenames1 = sorted(os.listdir(root_folder+annotators[select[1]]))
    
    for file in filenames:
        img = cm.load(os.path.join(img_folder, file[:-4]+'_summary.tif'))[0]
        dims = img.shape
        mask0 = nf_read_roi_zip(os.path.join(combined_folder, file), dims=dims) * 1.
        mask1 = nf_read_roi_zip(os.path.join(root_folder, annotators[select], file), dims=dims) * 1.
        plt.figure();plt.imshow(mask0.sum(0));plt.colorbar();plt.show()
        
        tp_gt, tp_comp, fn_gt, fp_comp, performance_cons_off = nf_match_neurons_in_binary_masks(
                mask0, mask1, thresh_cost=0.7, min_dist=10, print_assignment=True,
                plot_results=True, Cn=img, labels=['combined', annotators[select]])
        #plt.savefig(os.path.join(save_img_folder, 'combined'+'&'+annotators[select][0]+file[:-4]+'.pdf'))
        plt.close()
        result[file] = performance_cons_off
    
    #%%
    processed = {}
    for i in ['f1_score','recall','precision']:
        #result = eval(i)
        processed[i] = {}    
        for j in ['L1','TEG','HPC']:
            if j == 'L1':
                temp = [result[k][i] for k in result.keys() if 'Fish' not in k and 'IVQ' not in k]
                if i == 'number':
                    processed[i]['L1'] = sum(temp)
                else:
                    processed[i]['L1'] = sum(temp)/len(temp)
            if j == 'TEG':
                temp = [result[k][i] for k in result.keys() if 'Fish' in k]
                if i == 'number':
                    processed[i]['TEG'] = sum(temp)
                else:
                    processed[i]['TEG'] = sum(temp)/len(temp)
            if j == 'HPC':
                temp = [result[k][i] for k in result.keys() if 'IVQ' in k]
                if i == 'number':
                    processed[i]['HPC'] = sum(temp)
                else:
                    processed[i]['HPC'] = sum(temp)/len(temp)
    
    #%%
    comparison['combined' + '&' + annotators[select][0]] = processed            
    print(comparison)
    np.save('/home/nel/Code/NEL_LAB/Mask_RCNN/result_f1/voltage_v1.2_manual.npy', comparison)

#%% Show combined result
folder = '/home/nel/NEL-LAB Dropbox/NEL/Papers/VolPy/manual_annotations/summary_images/combination_v1.2'

for file in sorted(os.listdir(folder))[:8]:
    fname = os.path.join(folder, file)
    print(fname)
    mask = nf_read_roi_zip(fname, dims=(600,600))
    plt.figure()
    plt.imshow(mask.sum(0))
    plt.show()


#%%
import zipfile
from caiman.base.rois import nf_read_roi

root_dir = '/home/nel/Code/VolPy_many_things/Mask_RCNN/consensus_three_labelers/combination'
save_dir = '/home/nel/Code/VolPy_many_things/Mask_RCNN/consensus_three_labelers/combination_npz'

filenames = sorted(os.listdir(root_dir))

for file in filenames:
    with zipfile.ZipFile(os.path.join(root_dir, file)) as zf:
        names = zf.namelist()
        coords = [nf_read_roi(zf.open(n))
                  for n in names]
        polygons = [{'name': 'polygon','all_points_x':i[:,1],'all_points_y':i[:,0]} for i in coords]
        np.savez(save_dir+ '/' + file[:-4]+'_mask.npz', polygons)

    




ds_list=ds_list[6:23]
    for i in ds_list:
        if 'zip' in i:
            


#%%
from caiman.base.rois import norm_nrg
import matplotlib.patches as mpatches
img_folder = '/home/nel/NEL-LAB Dropbox/NEL/Papers/VolPy/manual_annotations/summary_images/corr/'
root_folder = '/home/nel/NEL-LAB Dropbox/NEL/Papers/VolPy/manual_annotations/summary_images/output/'
save_img_folder = '/home/nel/NEL-LAB Dropbox/NEL/Papers/VolPy/manual_annotations/summary_images/comparison/'
annotators = sorted(os.listdir(root_folder))

for file in filenames:
    cj0 = []
    cm0 = []
    jm0 = []
    c1 = []
    m1 = []
    j1 = []
    tri = []

    try:
        for select in [[0,1], [0,2], [1,2]]:
            filenames = sorted(os.listdir(root_folder+annotators[select[0]]))
            img = cm.load(os.path.join(img_folder, file[:-4]+'_lci.tif'))
            dims = img.shape
            try:
                mask0 = nf_read_roi_zip(os.path.join(root_folder, annotators[select[0]], file), dims=dims) * 1.
                mask1 = nf_read_roi_zip(os.path.join(root_folder, annotators[select[1]], file), dims=dims) * 1.
            except:
                mask0 = nf_read_roi_zip(os.path.join(root_folder, annotators[select[0]], file), dims=(dims[1], dims[0])) * 1.
                mask1 = nf_read_roi_zip(os.path.join(root_folder, annotators[select[1]], file), dims=(dims[1], dims[0])) * 1.
                img = img.T
            
            tp_gt, tp_comp, fn_gt, fp_comp, performance_cons_off = nf_match_neurons_in_binary_masks(
                    mask0, mask1, thresh_cost=0.7, min_dist=10, print_assignment=True,
                    plot_results=False, Cn=img, labels=[annotators[select[0]], annotators[select[1]]])
        
            if select == [0, 1]:
                cj0 = [tp_gt, tp_comp]
            elif select == [0, 2]:
                cm0 = [tp_gt, tp_comp]
            elif select == [1, 2]:
                jm0 = [tp_gt, tp_comp]
                
        for i in cj0[0]:
            if i in cm0[0]:
                tri.append([i, cj0[1][np.where(cj0[0] == i)[0][0]], cm0[1][np.where(cm0[0] == i)[0][0]]])
                     
        for i in tri:
            print(i)     
            temp = np.random.randint(0, 3) 
            if temp == 0:
                c1.append(i[0])
            elif temp == 1:
                m1.append(i[1])
            elif temp == 2:
                j1.append(i[2])
    
        tri = np.array(tri)
        for i in cj0[0]:
            if i not in tri[:,0]:
                if np.random.randint(0, 2) == 0:
                    c1.append(i)
                else:
                    j1.append(cj0[1][np.where(cj0[0] == i)][0])
        
        for i in cm0[0]:
            if i not in tri[:,0]:
                if np.random.randint(0, 2) == 0:
                    c1.append(i)
                else:
                    m1.append(cm0[1][np.where(cm0[0] == i)][0])   
                    
        for i in jm0[0]:
            if i not in tri[:,1]:
                if np.random.randint(0, 2) == 0:
                    j1.append(i)
                else:
                    m1.append(jm0[1][np.where(jm0[0] == i)][0])    
        
    #%%
    
        import zipfile
        from caiman.base.rois import nf_merge_roi_zip
        nf_merge_roi_zip(fnames = [os.path.join(root_folder, annotators[i], file) for i in range(3)], 
                                   idx_to_keep = [list(c1), list(j1), list(m1)], 
                                   new_fold = os.path.join('/home/nel/NEL-LAB Dropbox/NEL/Papers/VolPy/manual_annotations/summary_images/combination', file[:-4]))
    except:
        print('1')
        
    
    
    fname = os.path.join(root_folder, annotators[0], file)
    with zipfile.ZipFile(fname) as zf:
        names = zf.namelist()
        coords = [nf_read_roi(zf.open(n)) for n in names]
        print(zf.open(names[0]))
        for names
        
        zf.write('/home/nel/NEL-LAB Dropbox/NEL/Papers/VolPy/manual_annotations/summary_images', basename(filePath))


#%%


              
                

