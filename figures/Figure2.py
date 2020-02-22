#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Feb 13 14:38:44 2020

@author: nel
"""

import caiman as cm
import numpy as np
from scipy.ndimage.measurements import center_of_mass
from caiman.base.rois import com
import matplotlib
matplotlib.rcParams['pdf.fonttype'] = 42
matplotlib.rcParams['ps.fonttype'] = 42
import matplotlib.pyplot as plt
from skimage import measure
colorsets = plt.cm.tab10(np.linspace(0,1,10))
colorsets = colorsets[[0,1,2,3,4,5,6,8,9],:]

#%%
del1 = np.std(np.array(vpy.estimates['num_spikes'])[:,-4:], axis=1)
del2 = np.array(output['passedLocalityTest'])
del3 = np.array(vpy.estimates['low_spk'])
select = np.multiply(np.multiply((del1<200), (del2>-1)), (del3<2))

#%%
delarray = np.array([11,16,25,26])
select = np.ones(ROIs_mrcnn.shape[0]).astype(np.int)
select[delarray] = 0
select = select>0


#%%
A = ROIs_mrcnn.copy()[select]
C = np.stack(vpy.estimates['trace'], axis=0).copy()[select]
spike = [vpy.estimates['spikeTimes'][i] for i in np.where(select>0)[0]]

#%% Seperate components
def neuron_number(A, N, n, n_group):    
    l = [center_of_mass(a)[1] for a in A]
    li = np.argsort(l)
    if N != n*n_group:
        li = np.append(li, np.ones((1,n*n_group-N),dtype=np.int8)*(-1))
    mat = li.reshape((n_group, n), order='F')
    return mat

N = A.shape[0]
n = N
n_group = np.int(np.ceil(N/n))
mat = neuron_number(A, N, n, n_group)

#%%
Cn = summary_image[:,:,2]  
A = A.astype(np.float64)
save_path= '/home/nel/Code/VolPy/Paper/Figure2/'

def plot_neuron_contour(A, N, n, n_group, Cn, save_path):
    number=0
    number1=0
    for i in range(n_group):
        plt.figure()    
        vmax = np.percentile(Cn, 97)
        vmin = np.percentile(Cn, 5)
        plt.imshow(Cn, interpolation='None', vmax=vmax, vmin=vmin, cmap=plt.cm.gray)
        plt.title('Neurons location')
        d1, d2 = np.shape(Cn)
        cm1 = com(A.copy().reshape((N,-1), order='F').transpose(), d1, d2)
        colors='yellow'
        for j in range(n):
            index = mat[i,j]
            print(index) 
            img = A[index]
            img1 = img.copy()
            #img1[img1<np.percentile(img1[img1>0],15)] = 0
            #img1 = connected_components(img1)
            img2 = np.multiply(img, img1)
            contours = measure.find_contours(img2, 0.5)[0]
            #img2=img.copy()
            img2[img2 == 0] = np.nan
            if index != -1:
                plt.plot(contours[:, 1], contours[:, 0], linewidth=1, color=colorsets[np.mod(number,9)])
                plt.text(cm1[index, 1]+0, cm1[index, 0]-0, str(number), color=colors)
                number=number+1 
        plt.savefig(save_path+'neuron_contour{}-{}'.format(number1,number-1)+'.pdf')
        number1=number
        
plot_neuron_contour(A, N, n, n_group, Cn, save_path)

#%%
CZ = C[:,:20000]
CZ = (CZ-CZ.mean(axis=1)[:,np.newaxis])/CZ.std(axis=1)[:,np.newaxis]


def plot_neuron_signal(A, N, n, n_group, Cn, save_path):     
    number=0
    number1=0
    for i in range(n_group):
        fig, ax = plt.subplots((mat[i,:]>-1).sum(),1)
        length = (mat[i,:]>-1).sum()
        for j in range(n):
            if j==0:
                ax[j].set_title('Signals')
            #Y_r = cnm2.estimates.YrA + cnm2.estimates.C
            if mat[i,j]>-1:
                #Y_r = cnm2.estimates.F_dff[select,:]
                index = mat[i,j]
                T = C.shape[1]
                ax[j].plot(np.arange(20000), -CZ[index], 'c', linewidth=1, color=colorsets[np.mod(number,9)])
                ax[j].autoscale(tight=True)
                ax[j].scatter(spike[index],
                 np.max(-CZ[index]+0.5) * np.ones(spike[index].shape),
                 color=colorsets[np.mod(number,9)], marker='.',linewidth=0.01)
                #ax[j].plot(np.arange(T), cnm2.estimates.S[index, :][:], 'r', linewidth=2)
                ax[j].text(-30, 0, f'{number}', horizontalalignment='center',
                     verticalalignment='center')
                ax[j].set_ylim([(-CZ).min(),(-CZ).max()])
                if j==0:
                    #ax[j].legend(labels=['Filtered raw data', 'Inferred trace'], frameon=False)
                    ax[j].text(-30, 3000, 'neuron', horizontalalignment='center', 
                      verticalalignment='center')
                if j<length-1:
                    ax[j].axis('off')
                if j==length-1:
                    ax[j].spines['right'].set_visible(False)
                    ax[j].spines['top'].set_visible(False)  
                    ax[j].spines['left'].set_visible(True) 
                    ax[j].get_yaxis().set_visible(True)
                    ax[j].set_xlabel('Frames')
                number = number + 1
        plt.tight_layout()
        plt.savefig(save_path+'neuron_signal{}-{}'.format(number1,number-1)+'.pdf')
        number1=number    
        
plot_neuron_signal(A, N, n, n_group, Cn, save_path)




