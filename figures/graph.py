#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Sep 20 10:33:04 2019

@author: Changjia
This file produce figures for voltage imaging paper
"""
#%%
import caiman as cm
import os
from caiman.base.rois import nf_read_roi_zip
import numpy as np
from caiman.source_extraction.cnmf.estimates import Estimates, compare_components
from caiman.base.rois import nf_match_neurons_in_binary_masks
import skimage
import matplotlib
matplotlib.rcParams['pdf.fonttype'] = 42
matplotlib.rcParams['ps.fonttype'] = 42
import matplotlib.pyplot as plt
try:
    if __IPYTHON__:
        # this is used for debugging purposes only. allows to reload classes
        # when changed
        get_ipython().magic('load_ext autoreload')
        get_ipython().magic('autoreload 2')
except NameError:
    pass
#%% Mask and comparison to the ground truth
use_maskrcnn = True
if use_maskrcnn:
    import sys
    from caiman.paths import caiman_datadir
    model_dir = caiman_datadir()+'/model'
    sys.path.append(model_dir) 
    import mrcnn.model as modellib
    from mrcnn import visualize
    from mrcnn import neurons
    import tensorflow as tf
    
    
#%%
dr = '/home/nel/Code/VolPy/Mask_RCNN/datasets/neurons/val/'
ds_list = sorted(os.listdir(dr))
test_dr = '/home/nel/Code/VolPy/Mask_RCNN/datasets/neurons/val/'
test_ds_list = sorted(os.listdir(test_dr))
test_set = [i[:-4] for i in test_ds_list if 'mask' not in i]    
    
config = neurons.NeuronsConfig()
class InferenceConfig(config.__class__):
    # Run detection on one image at a time
    GPU_COUNT = 1
    IMAGES_PER_GPU = 1
    DETECTION_MIN_CONFIDENCE = 0.33
    IMAGE_RESIZE_MODE = "pad64"
    IMAGE_MAX_DIM=512
    CLASS_THRESHOLD = 0.3
    RPN_NMS_THRESHOLD = 0.7
    DETECTION_NMS_THRESHOLD = 0.3
config = InferenceConfig()
config.display()

DEVICE = "/cpu:0"  # /cpu:0 or /gpu:0
with tf.device(DEVICE):
    model = modellib.MaskRCNN(mode="inference", model_dir=model_dir,
                              config=config)
#weights_path = model_dir + '/mrcnn/mask_rcnn_neurons_0200.h5'
#weights_path = '/home/nel/Code/VolPy/Mask_RCNN/logs/neurons20190918T2052/mask_rcnn_neurons_0200.h5'
#weights_path = '/home/nel/Code/VolPy/Mask_RCNN/logs/neurons20191002T1113/mask_rcnn_neurons_0090.h5'
#weights_path = '/home/nel/Code/VolPy/Mask_RCNN/logs/neurons20191002T1337/mask_rcnn_neurons_0050.h5'
#weights_path = '/home/nel/Code/VolPy/Mask_RCNN/logs/neurons20191002T1337/mask_rcnn_neurons_0050.h5'
#weights_path = '/home/nel/Code/VolPy/Mask_RCNN/logs/neurons20191003T1509/mask_rcnn_neurons_0050.h5'
#weights_path = '/home/nel/Code/VolPy/Mask_RCNN/logs/neurons20191006T2333/mask_rcnn_neurons_0050.h5'
weights_path = '/home/nel/Code/VolPy/Mask_RCNN/logs/neurons20191014T1050/mask_rcnn_neurons_0080.h5'


model.load_weights(weights_path, by_name=True)

NEURONS_DIR = '/home/nel/Code/VolPy/Mask_RCNN/datasets/neurons/'
dataset = neurons.NeuronsDataset()
dataset.load_neurons(NEURONS_DIR, "train")
# Must call before using the dataset
dataset.prepare()

performance=[]
image_id = 2
for image_id in dataset.image_ids:
    image, image_meta, gt_class_id, gt_bbox, gt_mask =\
        modellib.load_image_gt(dataset, config, image_id, use_mini_mask=False)
    info = dataset.image_info[image_id]
    print("image ID: {}.{} ({}) {}".format(info["source"], info["id"], image_id, 
                                           dataset.image_reference(image_id)))
    
    # Run object detection
    
    results = model.detect([image], verbose=1)
    
    # Display results
    _, ax = plt.subplots(1,1, figsize=(16,16))
    r = results[0]
    display_result = True
    if display_result:  
        visualize.display_instances(image, r['rois'], r['masks'], r['class_ids'], 
                                    dataset.class_names, r['scores'], ax=ax,
                                    title="Predictions")
    #plt.savefig('/home/nel/Code/VolPy/Paper/' +dataset.image_info[image_id]['id'][:-4]+'_mrcnn_result.pdf')
    plt.close()
    
    mask_pr = r['masks'].copy().transpose([2,0,1])*(1.)
    mask_gt = gt_mask.copy().transpose([2,0,1])*(1.)
    
    tp_gt, tp_comp, fn_gt, fp_comp, performance_cons_off = nf_match_neurons_in_binary_masks(
            mask_gt, mask_pr, thresh_cost=0.7, min_dist=10, print_assignment=True,
            plot_results=True, Cn=image[:,:,0], labels=['GT', 'MRCNN'])
    #plt.savefig('/home/nel/Code/VolPy/Paper/' +dataset.image_info[image_id]['id'][:-4]+'_compare.pdf')
    plt.close()
    performance.append(performance_cons_off)    

#%% F1 score
train = np.array([[0.8514, 0.8641, 0.8724],
                  [0.8038, 0.7263, 0.8253],
                  [0.7089, 0.5509, 0.6815]])
val = np.array([[0.8853, 0.8223, 0.8419],
                [0.7647, 0.7536, 0.7912],
                [0.4667, 0.4742, 0.3533]])

train = np.array([[0.8180, 0.8409, 0.8436],
              [0.7689, 0.6973, 0.7437],
              [0.7089, 0.5342, 0.6815]])
val = np.array([[0.8432, 0.8094, 0.8140],
                [0.7647, 0.6377, 0.7253],
                [0.4667, 0.4742, 0.3533]])

    
l = [performance[i]['f1_score'] for i in range(len(performance))]

nn = [performance[i]['precision'] for i in range(len(performance))]

mm = [performance[i]['recall'] for i in range(len(performance))]


#%% F1 score
fig = plt.figure()
plt.title('F1 score')
ax = fig.add_subplot(111)
#bar_data = [performance[i]['f1_score'] for i in range(len(performance))]
#bar_data = [(bar_data[1]+bar_data[2])/2,bar_data[0],(bar_data[3]+bar_data[4]+bar_data[5])/3]
ax.bar(['Kaspar','Johannes','Adam'],bar_data,width=0.4,color=[[1,0,0],[0,0,1],[.5,.5,.5]])
#ax.spines['bottom'].set_visible(False)
plt.savefig('/home/nel/Code/VolPy/Paper/F1_train.pdf')

test: [0.8295668549905838, 0.6071428571428571, 0.5213182692184444]
train:[0.83962, 0.8375, 0.6161]
m = np.array([[0.83962   , 0.8375    , 0.6161    ],
       [0.82956685, 0.60714286, 0.52131827]])

labels = ['train', 'test']
x = np.arange(len(labels))/2  # the label locations
width = 0.2  # the width of the bars

fig, ax = plt.subplots()
rects1 = ax.bar(x - width/2, m[:,0], width/2, label='Kaspar')
rects2 = ax.bar(x, m[:,1], width/2, label='Johannes')
rects3 = ax.bar(x + width/2, m[:,2], width/2, label='Adam')

# Add some text for labels, title and custom x-axis tick labels, etc.
ax.set_ylabel('Scores')
ax.set_ylabel('F1 score')
ax.set_ylim([0.40,0.95])
ax.spines['right'].set_visible(False)
ax.spines['top'].set_visible(False)
ax.set_xticks(x)
ax.set_xticklabels(labels)
ax.legend(loc=1)
plt.savefig('/home/nel/Code/VolPy/Paper/F1_train&test.pdf')

#%% F1 score for all datasets
v1 = {'TEG_Voltron.01.02':0.7647058823529411, 'L1_Voltron.03.35':0.7850467289719626, 
      'L1_Voltron.04.50':0.9014084507042254, 'HPC_QuasAr.48.05':0.5, 
      'HPC_QuasAr.48.07':0.4, 'HPC_QuasAr.48.08':0.5}

t1 = {'TEG_Voltron.02.01':0.782608695652174, 'TEG_Voltron.03.01': 0.7551020408163265,
      'L1_Voltron.00.00': 0.7682119205298014,'L1_Voltron.01.00': 0.8468468468468469,
      'L1_Voltron.01.35': 0.7703703703703704,'L1_Voltron.02.00': 0.8455284552845529,
      'L1_Voltron.02.80': 0.875,'L1_Voltron.03.00': 0.8571428571428571,
      'L1_Voltron.04.00': 0.7628865979381443,'HPC_QuasAr.29.06': 0.6666666666666666,
      'HPC_QuasAr.32.01': 0.6,'HPC_QuasAr.38.05': 0.5714285714285714,
      'HPC_QuasAr.38.03': 1.0,'HPC_QuasAr.39.07': 0.8333333333333334,
      'HPC_QuasAr.39.03': 0.8333333333333334,'HPC_QuasAr.39.04': 0.6666666666666666,
      'HPC_QuasAr.48.01': 0.5}

dict1_train={'10192017Fish2-1': 0.782608695652174,
 '10192017Fish3-1': 0.7551020408163265,
 '403106_3min': 0.7682119205298014,
 'FOV1': 0.8468468468468469,
 'FOV1_35um': 0.7703703703703704,
 'FOV2': 0.8455284552845529,
 'FOV2_80um': 0.875,
 'FOV3': 0.8571428571428571,
 'FOV4': 0.7628865979381443,
 'IVQ29_S5_FOV6': 0.6666666666666666,
 'IVQ32_S2_FOV1': 0.6,
 'IVQ38_S1_FOV5': 0.5714285714285714,
 'IVQ38_S2_FOV3': 1.0,
 'IVQ39_S1_FOV7': 0.8333333333333334,
 'IVQ39_S2_FOV3': 0.8333333333333334,
 'IVQ39_S2_FOV4': 0.6666666666666666,
 'IVQ48_S7_FOV1': 0.5}
dict1_val={'06152017Fish1-2': 0.7647058823529411,
 'FOV3_35um': 0.7850467289719626,
 'FOV4_50um': 0.9014084507042254,
 'IVQ48_S7_FOV5': 0.5,
 'IVQ48_S7_FOV7': 0.4,
 'IVQ48_S7_FOV8': 0.5}
dict1 = dict1_train.copy()
dict1.update(dict1_val)

dict2_train={'06152017Fish1-2': 0.6956521739130435,
 '10192017Fish3-1': 0.6990291262135923,
 'FOV1': 0.8952380952380953,
 'FOV2': 0.8429752066115702,
 'FOV2_80um': 0.8421052631578947,
 'FOV3': 0.8027210884353742,
 'FOV3_35um': 0.7766990291262136,
 'FOV4': 0.8444444444444444,
 'FOV4_50um': 0.8823529411764706,
 'IVQ38_S2_FOV3': 1.0,
 'IVQ39_S1_FOV7': 0.5,
 'IVQ39_S2_FOV3': 0.7272727272727273,
 'IVQ39_S2_FOV4': 0.5454545454545454,
 'IVQ48_S7_FOV1': 0.36363636363636365,
 'IVQ48_S7_FOV5': 0.47058823529411764,
 'IVQ48_S7_FOV7': 0.4,
 'IVQ48_S7_FOV8': 0.26666666666666666}
dict2_val = {'10192017Fish2-1': 0.6376811594202898,
 '403106_3min': 0.8129032258064516,
 'FOV1_35um': 0.8059701492537313,
 'IVQ29_S5_FOV6': 0.6666666666666666,
 'IVQ32_S2_FOV1': 0.21052631578947367,
 'IVQ38_S1_FOV5': 0.5454545454545454}
dict2 = dict2_train.copy()
dict2.update(dict2_val)

dict3_train={'06152017Fish1-2': 0.7076923076923077,
 '10192017Fish2-1': 0.7796610169491526,
 '403106_3min': 0.8734177215189873,
 'FOV1': 0.8867924528301887,
 'FOV1_35um': 0.8244274809160306,
 'FOV3': 0.7671232876712328,
 'FOV3_35um': 0.822429906542056,
 'FOV4': 0.8351648351648352,
 'FOV4_50um': 0.8955223880597015,
 'IVQ29_S5_FOV6': 0.6666666666666666,
 'IVQ32_S2_FOV1': 0.5333333333333333,
 'IVQ38_S1_FOV5': 0.7272727272727273,
 'IVQ38_S2_FOV3': 1.0,
 'IVQ39_S1_FOV7': 0.9090909090909091,
 'IVQ48_S7_FOV5': 0.6153846153846154,
 'IVQ48_S7_FOV7': 0.5,
 'IVQ48_S7_FOV8': 0.5}
dict3_val = {'10192017Fish3-1': 0.7252747252747253,
 'FOV2': 0.8099173553719008,
 'FOV2_80um': 0.8181818181818182,
 'IVQ39_S2_FOV3': 0.6153846153846154,
 'IVQ39_S2_FOV4': 0.4444444444444444,
 'IVQ48_S7_FOV1': 0.0}
dict3 = dict3_train.copy()
dict3.update(dict3_val)

dict_mean = {}
for key in sorted(dict1):
    print(key)
    dict_mean[key] = (dict1[key] + dict2[key] + dict3[key])/3
    
dict_mean = {
 'L1_Voltron.00.00':0.8129032258064516,
 'L1_Voltron.01.00':0.876,
 'L1_Voltron.01.35':0.8059701492537313,
 'L1_Voltron.02.00':0.8099173553719008,
 'L1_Voltron.02.80':0.8181818181818182,
 'L1_Voltron.03.00':0.801,
 'L1_Voltron.03.35':0.7850467289719626,
 'L1_Voltron.04.00':0.814,
 'L1_Voltron.04.50':0.9014084507042254,      
 'TEG_Voltron.01.02': 0.7647058823529411,
 'TEG_Voltron.02.01': 0.6376811594202898,
 'TEG_Voltron.03.01': 0.7252747252747253,
 'HPC_QuasAr.29.06': 0.6666666666666666,
 'HPC_QuasAr.32.01': 0.21052631578947367,
 'HPC_QuasAr.38.05': 0.5454545454545454,
 'HPC_QuasAr.38.03': 1.0,
 'HPC_QuasAr.39.07': 0.747,
 'HPC_QuasAr.39.03': 0.6153846153846154,
 'HPC_QuasAr.39.04': 0.4444444444444444,
 'HPC_QuasAr.48.01': 0.0,
 'HPC_QuasAr.48.05': 0.5,
 'HPC_QuasAr.48.07': 0.4,
 'HPC_QuasAr.48.08': 0.5}

#result = np.array([[key,dict_mean[key]] for key in dict_mean])
plt.figure(figsize=(15,15))
keys = [key for key in dict_mean]
values = [dict_mean[key] for key in dict_mean]
i = 0
a = np.array([ 0,  1,  2,  3,  4,  5,  6,  7,  8])
plt.bar(a,values[a[0]:a[-1]+1], width=0.5, label='L1_Voltron', color='red')
b= np.array([ 9, 10, 11])
plt.bar(b,values[b[0]:b[-1]+1], width=0.5, label='TEG_Voltron', color='blue')
c= np.array([ 12, 13, 14,15,16,17,18,19,20,21,22])
plt.bar(c,values[c[0]:], width=0.5, label='HPC_QuasAr', color='gray')

#plt.bar(1,10)
#plt.bar(np.arange(len(keys)),values, width=0.5, label='mrcnn')
plt.ylabel('F1 scores')
plt.xticks(np.arange(len(keys)), keys, rotation='vertical',fontsize=8)
plt.legend()
ax = plt.gca()
ax.spines['right'].set_visible(False)
ax.spines['top'].set_visible(False)
plt.tight_layout()
plt.savefig('/home/nel/Code/VolPy/Paper/pic_paper/F1_alldata_new.pdf')
#%%%%%%%%%%%
# T vs frames
"""
10000 frames
motion correction time:
54.37724542617798
memory map time:
11.589507341384888
init time:
6.994962453842163
spike pursuit time:
70.4176857471466

20000 frames
motion correction time:
110.28093028068542
memory map time:
23.277902603149414
init time:
7.705028057098389
spike pursuit time:
150.00337171554565

36000 frames
motion correction time:
171.50250887870789
memory map time:
69.83776998519897
init time:
11.04919719696045
spike pursuit time:
300.5299062728882
"""


import matplotlib.pyplot as plt

m = np.array([[54.3,31.7,42.4+10,75.3],
              [114.5,9.4,86.2+10,171.3],
              [118.0,30.3,83.0+10,320.5]])

plt.figure(figsize=(4, 4))


size = np.array([2.6, 5.2, 9.4])
plt.title('Processing time allocation')
plt.bar((size), (m[:,0]), width=0.5, bottom=0)
plt.bar((size), (m[:,1]), width=0.5, bottom=(m[:,0]))
plt.bar((size), (m[:,2]), width=0.5, bottom=(m[:,0] + m[:,1]))
plt.bar((size), (m[:,3]), width=0.5, bottom=(m[:,0] + m[:,1] + m[:,2]))
plt.legend(['motion corection', 'mem mapping', 'segmentation','SpikePursuit'],frameon=False)
plt.xlabel('size (GB)')
plt.ylabel('time (seconds)')

#pl.plot((np.sort(size)), (10 ** np.sort(size)) / 31.45, '--k')
#plt.xlim([3.6, 5.2])
ax = plt.gca()
ax.locator_params(nbins=7)
ax.spines['right'].set_visible(False)
ax.spines['top'].set_visible(False)
#axx.set_yticklabels([str(int((ss) / 60))[:5] for ss in axx.get_yticks()])
#axx.set_xticklabels([int(i) for i in size])
plt.xticks(size, [str(int(i)) for i in size])
plt.savefig('/home/nel/Code/VolPy/Paper/t&size.pdf')


#%% T vs cpu
"""
36000 frames
motion correction time:
171.50250887870789
memory map time:
69.83776998519897
init time:
11.04919719696045
spike pursuit time:
300.5299062728882

36000 frames 4 cpu
motion correction time:
175.04763960838318
memory map time:
56.60492563247681
init time:
20.04719305038452
spike pursuit time:
395.6903967857361

36000 frames 2cpu
motion correction time:
250.72345852851868
memory map time:
56.17555284500122
init time:
12.629401445388794
spike pursuit time:
555.668318271637

36000 frames 1cpu
motion correction time:
257.0926282405853
memory map time:
70.89436888694763
init time:
12.887078523635864
spike pursuit time:
994.7748708724976
"""

m = np.array([[278.6,52.3,91.4+10,952.8],
              [204,36.7,84.5+10,581.1],
              [136.5,26.1,87.5+10,426.6],
              [123.4,36.7,83+10,335.1],
              [118.0,30.3,83.0+10,320.5]])


    
plt.figure(figsize=(4, 4))
size = np.array([1, 2, 4, 6, 8])
plt.title('parallelization speed')
plt.bar((size), (m[:,0]), width=0.5, bottom=0)
plt.bar((size), (m[:,1]), width=0.5, bottom=(m[:,0]))
plt.bar((size), (m[:,2]), width=0.5, bottom=(m[:,0] + m[:,1]))
plt.bar((size), (m[:,3]), width=0.5, bottom=(m[:,0] + m[:,1] + m[:,2]))
plt.legend(['motion corection', 'mem mapping', 'segmentation','SpikePursuit'],frameon=False)
plt.xlabel('number of processors')
plt.ylabel('time (seconds)')

#pl.plot((np.sort(size)), (10 ** np.sort(size)) / 31.45, '--k')
#plt.xlim([3.6, 5.2])
ax = plt.gca()
ax.locator_params(nbins=7)
ax.spines['right'].set_visible(False)
ax.spines['top'].set_visible(False)
#ax.set_xticklabels(size)
plt.xticks(size, ('1', '2', '4', '6', '8'))
plt.savefig('/home/nel/Code/VolPy/Paper/t&cpu.pdf')

#%% memory vs cpu
"""
36000 frames 8 cpu with gpu
memory 60228.816406

36000 frames 4 cpu with gpu
memory 50313.636719

36000 frames 2 cpu with gpu
memory 39683.988281

36000 frames 1 cpu with gpu
36935.65625
"""

m = np.array([15500,15500,21500,26950,37000])/1024
plt.figure(figsize=(4, 4))
size = np.array([1, 2, 4, 6, 8])
plt.title('peak memory')
plt.scatter(size, m, color='orange',linewidth=5)

#plt.legend([''],frameon=False)
plt.xlabel('number of processors')
plt.ylabel('peak memory (GB)')

ax = plt.gca()
ax.spines['right'].set_visible(False)
ax.spines['top'].set_visible(False)
plt.xticks(size, ('1', '2', '4', '6', '8'))
plt.savefig('/home/nel/Code/VolPy/Paper/memory&cpu.pdf')


#%%


"""
summary_image = np.load(dr + test_set[i] + '.npz')['arr_0']
summary_image = skimage.color.gray2rgb(np.array(summary_image))
dims= (summary_image.shape[0], summary_image.shape[1])
fname = dr + test_set[i] + '_mask.npz'
mask_gt = np.load(fname, allow_pickle=True)['arr_0']

results = model.detect([summary_image], verbose=1)
r = results[0]
mask_pr = r['masks'].transpose([2,0,1])

_, ax = plt.subplots(1,1, figsize=(16,16))
visualize.display_instances(summary_image, r['rois'], r['masks'], r['class_ids'], 
                        ['BG', 'neurons'], r['scores'], ax=ax,
                        title="Predictions")
"""


#%% Pipeline
n = 1
plt.figure();plt.plot(-vpy.estimates['trace'][n])
plt.scatter(vpy.estimates['spikeTimes'][n],((-vpy.estimates['trace'][n]).max()+10)*np.ones(len(vpy.estimates['spikeTimes'][n])))
ax = plt.gca()
ax.spines['right'].set_visible(False)
ax.spines['top'].set_visible(False)
plt.savefig('/home/nel/Code/VolPy/Paper/pic/neuron15.pdf')

m = cm.load(fname_new)
mm = m[:,r['masks'][:,:,n]]
plt.plot(-mm.mean(axis=1)[200:450]-np.min(-mm.mean(axis=1)[200:450]))
ax = plt.gca()
ax.spines['right'].set_visible(False)
ax.spines['top'].set_visible(False)
plt.savefig('/home/nel/Code/VolPy/Paper/neuron27_original_sm.pdf')

plt.figure();plt.plot(-vpy.estimates['trace'][n][200:450])
p = vpy.estimates['spikeTimes'][n][np.multiply(vpy.estimates['spikeTimes'][n]>200,vpy.estimates['spikeTimes'][n]<450)]
plt.scatter(p-200,((-vpy.estimates['trace'][n]).max()+10)*np.ones(len(p)))
plt.hlines(vpy.estimates['thresh'][n], 0,250,color='gray', linestyles='dashed')
ax = plt.gca()
ax.spines['right'].set_visible(False)
ax.spines['top'].set_visible(False)
plt.savefig('/home/nel/Code/VolPy/Paper/neuron27_processed.pdf')

plt.imshow(-vpy.estimates['spatialFilter'][9]);plt.colorbar()
plt.savefig('/home/nel/Code/VolPy/Paper/pic/neuron9_spatial_filter.pdf')

#%% Count number of neurons for training
import os
import numpy as np
dr = '/home/nel/Code/VolPy/Mask_RCNN/datasets/neurons/train/'
ds_list = sorted(os.listdir(dr))
l = []
for i in ds_list:
    if 'mask' in i:
        m = np.load((dr + i), allow_pickle=True)['arr_0']
        print(i)
        """
        if 'IVQ' in i:
            print(len(m)/16)
            l.append(len(m)/16)
        else:
        """
        print(len(m))
        l.append(len(m))

#%% Video
m2 = m2 * 1.2
mm = cm.concatenate([m1, m2], axis=2)
mm = cm.concatenate([mm, lcm1], axis=2)

m = cm.load('/home/nel/Code/VolPy/Mask_RCNN/videos & imgs/neurons_mc/403106_3min_d1_128_d2_512_d3_1_order_C_frames_20000_.mmap')
lcm = cm.load('/home/nel/Code/VolPy/Paper/pic_paper/video/corr_video.tif')

m = m[:19871,:,:]
m.shape
lcm.shape
m = m.transpose([0,2,1])
lcm = lcm.transpose([0,2,1])
m.shape
lcm.shape
m.play()
lcm.play()
lcm.play(fr=30)
lcm.play(fr=300)
(m-m.mean(axis=0)).play()
moviehandle = cm.concatenate([m, lcm], axis=2)
moviehandle.shape
moviehandle.play(fr=300)
m.shape
m
lcm
m/1000
m/10000
m.max
m
np.max(m)
np.min(m)
np.max(lcm)
np.min(lcm)
m = m-np.min(m)
np.min(m)
m = m/(np.max(m))
np.max(m)
m*1.4-0.4
moviehandle = cm.concatenate([m, lcm], axis=2)
moviehandle.play()
moviehandle.play(fr=400)
moviehandle = cm.concatenate([m, m-m.mean(axis=0), lcm], axis=2)
moviehandle.save('/home/nel/Code/VolPy/Paper/pic_paper/video/raw_corr_video.tif')
moviehandle.shape
moviehandle[:6000].save('/home/nel/Code/VolPy/Paper/pic_paper/video/raw_corr_video_15s.tif')
m1 = m[:6000]
lcm1 = lcm[:6000]
mm = cm.concatenate([m1, m1-m1.mean(axis=0)], axis=2)
mm.shape
mm = cm.concatenate([mm, lcm1], axis=2)
mm.shape
mm.play()
m1-m1.mean(axis=0)
m1.max()
(m1-m1.mean(axis=0)).max()
(m1-m1.mean(axis=0)).min()
m2 = m1-m1.mean(axis=0)
m2 = (m2-m2.min())
m2.min()
m2 = m2/m2.max()
m2.max()
m2 = m2*1.4
mm = cm.concatenate([m1, m2], axis=2)
mm = cm.concatenate([mm, lcm1], axis=2)
mm.shape
mm.play()
m2
m2 = m2 * 1.2
mm = cm.concatenate([m1, m2], axis=2)
mm = cm.concatenate([mm, lcm1], axis=2)
mm.play()
'/home/nel/Code/VolPy/Mask_RCNN/videos & imgs/summary imgs/mean'


#%% F1 score new
dict1_train={'10192017Fish2-1': 0.8076923076923077,
 '10192017Fish3-1': 0.7555555555555555,
 'FOV1': 0.8952380952380953,
 'FOV1_35um': 0.8769230769230769,
 'FOV2': 0.9090909090909091,
 'FOV2_80um': 0.963855421686747,
 'FOV3': 0.8947368421052632,
 'FOV4': 0.8936170212765957,
 'IVQ29_S5_FOV6': 0.5714285714285714,
 'IVQ32_S2_FOV1': 0.5,
 'IVQ38_S1_FOV5': 0.8,
 'IVQ38_S2_FOV3': 1.0,
 'IVQ39_S1_FOV7': 0.7692307692307693,
 'IVQ39_S2_FOV3': 0.9090909090909091,
 'IVQ39_S2_FOV4': 0.5,
 'IVQ48_S7_FOV1': 0.4}
dict1_val={'06152017Fish1-2': 0.6984126984126984,
 '403106_3min': 0.8695652173913043,
 'FOV3_35um': 0.9009009009009009,
 'FOV4_50um': 0.9696969696969697,
 'IVQ29_S5_FOV4': 0.2857142857142857,
 'IVQ48_S7_FOV5': 0.5,
 'IVQ48_S7_FOV7': 0.4,
 'IVQ48_S7_FOV8': 0.3333333333333333}
dict1 = dict1_train.copy()
dict1.update(dict1_val.copy())

dict1_recall = {'06152017Fish1-2': 0.6666666666666666,
 '403106_3min': 0.8333333333333334,
 'FOV3_35um': 0.8771929824561403,
 'FOV4_50um': 0.9411764705882353,
 'IVQ29_S5_FOV4': 0.5,
 'IVQ48_S7_FOV5': 0.75,
 'IVQ48_S7_FOV7': 1.0,
 'IVQ48_S7_FOV8': 0.6666666666666666}

dict1_precision = {'06152017Fish1-2': 0.7333333333333333,
 '403106_3min': 0.9090909090909091,
 'FOV3_35um': 0.9259259259259259,
 'FOV4_50um': 1.0,
 'IVQ29_S5_FOV4': 0.2,
 'IVQ48_S7_FOV5': 0.375,
 'IVQ48_S7_FOV7': 0.25,
 'IVQ48_S7_FOV8': 0.2222222222222222}


dict2_train={'06152017Fish1-2': 0.6774193548387096,
 '10192017Fish3-1': 0.735632183908046,
 '403106_3min': 0.8846153846153846,
 'FOV2_80um': 0.9761904761904762,
 'FOV3': 0.8707482993197279,
 'FOV3_35um': 0.8888888888888888,
 'FOV4': 0.896551724137931,
 'FOV4_50um': 0.9538461538461539,
 'IVQ29_S5_FOV4': 0.5714285714285714,
 'IVQ32_S2_FOV1': 0.5454545454545454,
 'IVQ38_S1_FOV5': 0.8,
 'IVQ39_S2_FOV4': 0.75,
 'IVQ48_S7_FOV1': 0.25,
 'IVQ48_S7_FOV5': 0.7272727272727273,
 'IVQ48_S7_FOV7': 0.5,
 'IVQ48_S7_FOV8': 0.2222222222222222}
dict2_val={'10192017Fish2-1': 0.7407407407407407,
 'FOV1': 0.8932038834951457,
 'FOV1_35um': 0.873015873015873,
 'FOV2': 0.8907563025210085,
 'IVQ29_S5_FOV6': 0.18181818181818182,
 'IVQ38_S2_FOV3': 0.6666666666666666,
 'IVQ39_S1_FOV7': 0.8333333333333334,
 'IVQ39_S2_FOV3': 0.5333333333333333}
dict2 = dict2_train.copy()
dict2.update(dict2_val.copy())

dict3_train={'06152017Fish1-2': 0.7368421052631579,
 '10192017Fish2-1': 0.8461538461538461,
 '403106_3min': 0.8789808917197452,
 'FOV1': 0.8823529411764706,
 'FOV1_35um': 0.8387096774193549,
 'FOV2': 0.8888888888888888,
 'FOV3_35um': 0.8514851485148515,
 'FOV4_50um': 0.9206349206349206,
 'IVQ29_S5_FOV4': 0.6666666666666666,
 'IVQ29_S5_FOV6': 0.8,
 'IVQ38_S2_FOV3': 1.0,
 'IVQ39_S1_FOV7': 0.8,
 'IVQ39_S2_FOV3': 0.8,
 'IVQ48_S7_FOV5': 0.7272727272727273,
 'IVQ48_S7_FOV7': 0.4444444444444444,
 'IVQ48_S7_FOV8': 0.5454545454545454}
dict3_val={'10192017Fish3-1': 0.6987951807228916,
 'FOV2_80um': 0.8860759493670886,
 'FOV3': 0.8671328671328671,
 'FOV4': 0.8888888888888888,
 'IVQ32_S2_FOV1': 0.26666666666666666,
 'IVQ38_S1_FOV5': 0.6666666666666666,
 'IVQ39_S2_FOV4': 0.8888888888888888,
 'IVQ48_S7_FOV1': 0.0}
dict3 = dict3_train.copy()
dict3.update(dict3_val.copy())





#%%
dict_all = dict1_val.copy()
dict_all.update(dict2_val.copy())
dict_all.update(dict3_val.copy())

plt.figure(figsize=(15,15))
keys = [key for key in sorted(dict_all)]
keys = ['403106_3min', 'FOV1', 'FOV1_35um', 'FOV2', 'FOV2_80um', 'FOV3',
 'FOV3_35um', 'FOV4', 'FOV4_50um', '06152017Fish1-2', '10192017Fish2-1',
 '10192017Fish3-1', 'IVQ29_S5_FOV4', 'IVQ29_S5_FOV6', 'IVQ32_S2_FOV1',
 'IVQ38_S1_FOV5', 'IVQ38_S2_FOV3', 'IVQ39_S1_FOV7', 'IVQ39_S2_FOV3',
 'IVQ39_S2_FOV4', 'IVQ48_S7_FOV1', 'IVQ48_S7_FOV5', 'IVQ48_S7_FOV7', 'IVQ48_S7_FOV8']
values = [dict_all[key] for key in keys]
i = 0
a = np.arange(0,9)
plt.bar(a,values[a[0]:a[-1]+1], width=0.5, label='L1', color='red')
b= np.arange(9,12)
plt.bar(b,values[b[0]:b[-1]+1], width=0.5, label='TEG', color='blue')
c= np.arange(12,24)
plt.bar(c,values[c[0]:], width=0.5, label='HPC', color='gray')
plt.legend()
plt.ylabel('F1 scores')
plt.xticks(np.arange(len(keys)), keys, rotation='vertical',fontsize=8)
plt.legend()
ax = plt.gca()
ax.spines['right'].set_visible(False)
ax.spines['top'].set_visible(False)
plt.tight_layout()
plt.savefig('/home/nel/Code/VolPy/Paper/pic_paper_new/F1_all.pdf')

#%%
t = np.zeros((3,3))
# train, fish
t[1,0] = sum([i for i in dict1_train.values()][0:2])/2
t[1,1] = sum([i for i in dict2_train.values()][0:2])/2
t[1,2] = sum([i for i in dict3_train.values()][0:2])/2
# train, l1
t[0,0] = sum([i for i in dict1_train.values()][2:8])/6
t[0,1] = sum([i for i in dict2_train.values()][2:8])/6
t[0,2] =sum([i for i in dict3_train.values()][2:8])/6
# train hpc
t[2,0] =sum([i for i in dict1_train.values()][8:])/8
t[2,1] =sum([i for i in dict2_train.values()][8:])/8
t[2,2] =sum([i for i in dict3_train.values()][8:])/8

# val fish
v = np.zeros((3,3))
v[1,0] = sum([i for i in dict1_val.values()][0:1])/1
v[1,1] = sum([i for i in dict2_val.values()][0:1])/1
v[1,2] = sum([i for i in dict3_val.values()][0:1])/1
# train, l1
v[0,0] = sum([i for i in dict1_val.values()][1:4])/3
v[0,1] = sum([i for i in dict2_val.values()][1:4])/3
v[0,2] = sum([i for i in dict3_val.values()][1:4])/3
# train hpc
v[2,0] = sum([i for i in dict1_val.values()][4:])/4
v[2,1] = sum([i for i in dict2_val.values()][4:])/4
v[2,2] = sum([i for i in dict3_val.values()][4:])/4

print(np.mean(t, axis=1))
print(np.std(t, axis=1))
print(np.mean(v, axis=1))
print(np.std(v, axis=1))

#%%
labels=['L1','TEG','HPC']
F1_mean = np.stack([np.mean(t, axis=1),np.mean(v, axis=1)], axis=1)
F1_std = np.stack([np.std(t, axis=1),np.std(v, axis=1)], axis=1)
x = np.arange(len(labels))
width = 0.35       # the width of the bars: can also be len(x) sequence

plt.figure()
ax = plt.gca()
rects1 = ax.bar(x - width/2, F1_mean[:,0], width, yerr=F1_std[:,0], label='train', error_kw=dict(ecolor='gray', lw=1.5, capsize=2, capthick=1))
rects2 = ax.bar(x + width/2, F1_mean[:,1], width, yerr=F1_std[:,1], label='val', error_kw=dict(ecolor='gray', lw=1.5, capsize=2, capthick=1))

# Add some text for labels, title and custom x-axis tick labels, etc.
ax.set_ylabel('F1 Score')
ax.set_xticks(x)
ax.set_xticklabels(labels)
ax.spines['right'].set_visible(False)
ax.spines['top'].set_visible(False)
ax.legend()
plt.savefig('/home/nel/Code/VolPy/Paper/pic_paper_new/F1_cross_validation.pdf')

#%%
dr = '/home/nel/Code/VolPy/Mask_RCNN/videos & imgs/summary imgs/mean'
for i in sorted(os.listdir(dr)):
    if 'IVQ' in i:
       a='/home/nel/Code/VolPy/Mask_RCNN/videos & imgs/summary imgs/mean/'+i 
       cm.load(a).transpose([2,1,0]).save('/home/nel/Code/VolPy/Mask_RCNN/videos & imgs/summary imgs/'+i)


#%%
dr = '/home/nel/Code/VolPy/Mask_RCNN/backup/Adam_backup'
ds_list = sorted(os.listdir(dr))
for i in ds_list:
    if 'zip' in i:
        import zipfile
        from caiman.base.rois import nf_read_roi
        with zipfile.ZipFile('/home/nel/Code/VolPy/Mask_RCNN/backup/Adam_backup/'+i) as zf:
            names = zf.namelist()
            coords = [nf_read_roi(zf.open(n))
                      for n in names]
            polygons = [{'name': 'polygon','all_points_x':i[:,1],'all_points_y':i[:,0]} for i in coords]
        np.savez('/home/nel/Code/VolPy/Mask_RCNN/backup/Adam_backup/'+i[:-4]+'_mask'+'.npz', polygons)
       
# %%
dr = '/home/nel/Code/VolPy/Mask_RCNN/videos & imgs/neurons_mc'
dz = '/home/nel/Code/VolPy/Paper/ZENODO' 
ds_list = sorted(os.listdir(dr)) 
for i in ds_list[3:]:
    if 'Fish' in i:
        m = cm.load(dr+'/'+i)
        m.save(dz+'/'+i[:-4]+'tif')
    if 'IVQ' in i:
        m = cm.load(dr+'/'+i)
        m.save(dz+'/'+i[:-4]+'tif')
    else:
        m = cm.load(dr+'/'+i)
        m = m.transpose([0,2,1])
        m.save(dz+'/'+i[:-4]+'tif')
        
        
#%% Reconstructed video
#%%
vpy.estimates['passedLocalityTest']
index = [0,1,2,3,4,7,8,9,10,11,12,13]
recons_mv = np.empty(mv.shape)
bwexp = vpy.estimates['bwexp'][index[0]]
A = np.zeros((bwexp.shape[0],bwexp.shape[1],len(index)))
for i in range(len(index)):
    print(i)
    bwexp = vpy.estimates['bwexp'][index[i]]
    Xinds = np.where(np.any(bwexp > 0, axis=1) > 0)[0]
    Yinds = np.where(np.any(bwexp > 0, axis=0) > 0)[0]
    weight = -vpy.estimates['spatialFilter'][index[i]]
    weight = weight/weight.max()
    weight[weight<0.4] = 0        
    A[Xinds[0]:Xinds[-1]+1,Yinds[0]:Yinds[-1]+1,i] = weight 
    
plt.imshow(A.sum(2))
C = np.vstack([vpy.estimates['recons_signal'][index[i]] for i in range(len(index))])
recons_mv = np.dot(A, C)    
recons_mv = recons_mv.transpose([2,0,1])
    
    #%%
    #recons_mv = recons_mv+np.random.rand(recons_mv.shape[0],recons_mv.shape[1],recons_mv.shape[2])*0.000001
    #recons_mv[-1,:,:]=np.ones((recons_mv.shape[1],recons_mv.shape[2]))
    recons_mv[:,0,0]=np.ones(recons_mv.shape[0])*0.0001
#%%
    cm.movie(recons_mv).play()
    
#%%
m_rig = cm.load(mc.mmap_file)
m_rig.fr=400
m_rig_bl = m_rig.computeDFF(secsWindow=1)[0]
m_rig_bl.shape
m_rig_bl.play()
m_rig_bl.play(fr=30)
m_rig.max()
m_rig.min()
m_rig_bl.max()
m_rig_bl.min()
recons_mv.max()
recons_mv.min()
m_rig = m_rig/m_rig.max()
m_rig.max()
m_rig_bl = m_rig_bl/(-m_rig_bl.min())
m_rig_bl.max()
m_rig_bl = -m_rig_bl

recons_mv = cm.movie(np.array(recons_mv),fr=400)
recons_mv = recons_mv/recons_mv.max()
m_all = cm.concatenate((m_rig,m_rig_bl,recons_mv),axis=2)

#%%  
plt.figure()
plt.imshow(ROIs_mrcnn[n])
plt.imshow(mv[0],alpha=0.5)

plt.figure()
plt.imshow(vpy.estimates['bwexp'][n])
    
       
plt.figure()
plt.plot(vpy.estimates['templates'][n])

plt.figure()
plt.plot(vpy.estimates['recons_signal'][n][:2000])     
        
       
       