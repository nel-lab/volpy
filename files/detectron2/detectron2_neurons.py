#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jan 27 16:16:03 2020

@author: nel
"""

#%%
# You may need to restart your runtime prior to this, to let your installation take effect
# Some basic setup:
# Setup detectron2 logger
import detectron2
from detectron2.utils.logger import setup_logger
setup_logger()

# import some common libraries
import numpy as np
import cv2
import random
#from google.colab.patches import cv2_imshow

# import some common detectron2 utilities
from detectron2 import model_zoo
from detectron2.engine import DefaultPredictor
from detectron2.config import get_cfg
from detectron2.utils.visualizer import Visualizer
from detectron2.data import MetadataCatalog

#%%
import os
import numpy as np
import json
import cv2
from detectron2.structures import BoxMode
from glob import glob

dt = ['neurons','neurons_calcium', 'neurons_calcium_corr'][0]
root_dir = os.path.join('/home/nel/data/calcium_data/detectron2', dt)
output_dir = os.path.join('/home/nel/output/detectron2', dt)

"""
def get_neurons_dicts(root_dir, dataset = 'train'):
    dataset_dicts = []
    img_list = glob(os.path.join(root_dir, dataset, '*.png'))
    for idx, file_name in enumerate(img_list):
        record = {}
        annotations = [] 
        height, width = cv2.imread(file_name).shape[:2]
        record["file_name"] = file_name
        record["image_id"] = idx
        record["height"] = height
        record["width"] = width
        
        json_name = file_name[:-4] + '.json'
        with open(json_name) as f:
            regions = json.load(f)
        for i in range(len(regions)):
            py, px = zip(*regions[i]['coordinates'])
            poly = [(x + 0.5, y + 0.5) for x, y in zip(px, py)]
            poly = [p for x in poly for p in x]          
            obj = {
                "bbox": [np.min(px), np.min(py), np.max(px), np.max(py)],
                "bbox_mode": BoxMode.XYXY_ABS,
                "segmentation": [poly],
                "category_id": 0,
                "iscrowd": 0
            }
            annotations.append(obj)
        record["annotations"] = annotations
        dataset_dicts.append(record)
    return dataset_dicts
"""
def get_neurons_dicts(root_dir, dataset = 'train'):
    dataset_dicts = []
    img_list = glob(os.path.join(root_dir, dataset, '*.png'))
    for idx, file_name in enumerate(img_list):
        record = {}
        annotations = [] 
        height, width = cv2.imread(file_name).shape[:2]
        record["file_name"] = file_name
        record["image_id"] = idx
        record["height"] = height
        record["width"] = width
        
        mask_name = file_name[:-4] + '_mask.npz'
        regions = np.load(mask_name, allow_pickle=True)['arr_0']
        for i in range(len(regions)):
            py = regions[i]['all_points_y']
            px = regions[i]['all_points_x']
            poly = [(x + 0.5, y + 0.5) for x, y in zip(px, py)]
            poly = [p for x in poly for p in x]          
            obj = {
                "bbox": [np.min(px), np.min(py), np.max(px), np.max(py)],
                "bbox_mode": BoxMode.XYXY_ABS,
                "segmentation": [poly],
                "category_id": 0,
                "iscrowd": 0
            }
            annotations.append(obj)
        record["annotations"] = annotations
        dataset_dicts.append(record)
    return dataset_dicts


from detectron2.data import DatasetCatalog, MetadataCatalog
for d in ["train", "val"]:
    DatasetCatalog.register("neurons_" + d, lambda d=d: get_neurons_dicts(root_dir, d))
    MetadataCatalog.get("neurons_" + d).set(thing_classes=["neurons"])

neurons_metadata = MetadataCatalog.get("neurons_train")

#%%
dataset_dicts = get_neurons_dicts(root_dir, 'val')
for d in random.sample(dataset_dicts, 2):
    img = cv2.imread(d["file_name"])
    visualizer = Visualizer(img[:, :, ::-1], metadata=neurons_metadata, scale=2)
    vis = visualizer.draw_dataset_dict(d)
    cv2.imshow('p',vis.get_image()[:, :, ::-1])
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    
#%%
from detectron2.engine import DefaultTrainer
from detectron2.config import get_cfg

cfg = get_cfg()
cfg.merge_from_file(model_zoo.get_config_file("COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml"))
cfg.MODEL.PIXEL_MEAN = [45,45,45]     #calcium [78,78,45]     # voltage [47, 47, 34]
cfg.MODEL.WEIGHTS = model_zoo.get_checkpoint_url("COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml")  # Let training initialize from model zoo

# INPUT
cfg.INPUT.MIN_SIZE_TRAIN = (128,128)
cfg.INPUT.MIN_SIZE_TEST = 0
cfg.INPUT.CROP['ENABLED'] = True
cfg.INPUT.CROP.TYPE = "absolute"
cfg.INPUT.CROP.SIZE = [128, 128]

# DATASETS
cfg.DATASETS.TRAIN = ("neurons_train",)
cfg.DATASETS.TEST = ()
cfg.DATALOADER.NUM_WORKERS = 15

cfg.MODEL.FPN.IN_FEATURES = ["res2", "res3", "res4", "res5"]
cfg.MODEL.ANCHOR_GENERATOR.SIZES = [[16, 32, 64, 128, 256]]
cfg.MODEL.RPN.IN_FEATURES = ['p2', 'p3', 'p4', 'p5', 'p6']
cfg.MODEL.RPN.PRE_NMS_TOPK_TRAIN = 3000
cfg.MODEL.RPN.POST_NMS_TOPK_TRAIN = 2000
cfg.MODEL.RPN.PRE_NMS_TOPK_TEST = 3000
cfg.MODEL.RPN.POST_NMS_TOPK_TEST = 2000
#cfg.MODEL.BACKBONE.FREEZE_AT
 
cfg.MODEL.ROI_HEADS.BATCH_SIZE_PER_IMAGE = 400
cfg.MODEL.RPN.BATCH_SIZE_PER_IMAGE = 1000
cfg.SOLVER.IMS_PER_BATCH = 6
cfg.SOLVER.STEPS = (300,3000)
cfg.SOLVER.BASE_LR = 0.01  # pick a good LR
cfg.SOLVER.MAX_ITER = 1000    # 300 iterations seems good enough for this toy dataset; you may need to train longer for a practical dataset
#cfg.MODEL.ROI_HEADS.BATCH_SIZE_PER_IMAGE = 128   # faster, and good enough for this toy dataset (default: 512)
cfg.MODEL.ROI_HEADS.NUM_CLASSES = 1  # only has one class (neuron)
#cfg.SOLVER.WARMUP_METHOD

#%%
import time
t = time.localtime()
last = False
if last:
    cfg.OUTPUT_DIR = os.path.join(output_dir, sorted(os.listdir(os.path.join(output_dir)))[-1])
else:    
    time_string = time.strftime("%Y%m%dt%H%M", t)
    cfg.OUTPUT_DIR = os.path.join(output_dir, time_string)  
    os.makedirs(cfg.OUTPUT_DIR, exist_ok=True)

#%%
from detectron2.data import build_detection_train_loader
from detectron2.data import detection_utils as utils
import matplotlib.pyplot as plt
import copy
import torch
class Trainer(DefaultTrainer):
    @classmethod
    def build_train_loader(cls, cfg):
        return build_detection_train_loader(cfg, mapper=mapper)
    
#%%
# dataset_dict = dataset_dicts[1]
def mapper(dataset_dict):
    # Implement a mapper, similar to the default DatasetMapper, but with your own customizations
    dataset_dict = copy.deepcopy(dataset_dict)  # it will be modified by code below
    image = utils.read_image(dataset_dict["file_name"], format="BGR")
    masks = anno_to_mask(image, dataset_dict['annotations'])
    #plt.figure(1);plt.subplot(211);plt.imshow(image);plt.subplot(212);plt.imshow(masks.sum(axis=2))
    # Crop and augmentation
    image, masks = random_crop(image, masks)
    masks = masks[:, :, np.where(masks.sum(axis=(0,1))>50)[0]]
    augmentation = create_augmentation()
    image, masks = imgaug_augmentation(image, masks, augmentation)
    masks = masks[:, :, np.where(masks.sum(axis=(0,1))>50)[0]]
    masks = masks.transpose([2,0,1])
    annotations = masks_to_anno(masks)
    #plt.figure(2);plt.subplot(211);plt.imshow(image);plt.subplot(212);plt.imshow(masks.sum(axis=0))
    # Save into dataset_dict
    dataset_dict["image"] = torch.as_tensor(image.transpose(2, 0, 1).astype("float32"))
    instances = utils.annotations_to_instances(annotations, image.shape[:2])
    dataset_dict["instances"] = utils.filter_empty_instances(instances)
    return dataset_dict  

def create_augmentation():
    import imgaug.augmenters as iaa
    sometimes = lambda aug: iaa.Sometimes(0.5, aug)
    augmentation = iaa.Sequential([iaa.Fliplr(0.5),
                                      iaa.Flipud(0.5),
                                      iaa.Affine(rotate=(-180, 180)),          
                                      #sometimes(iaa.GaussianBlur(sigma=(0, 0.25))),
                                      sometimes(iaa.Multiply((0.5,2))),
                                      sometimes(iaa.Affine(shear=(-2,2))),
                                      sometimes(iaa.Affine(scale=(0.8, 1.3)))],random_order=True)
    return augmentation

def anno_to_mask(image, regions):
    masks = np.zeros((image.shape[0], image.shape[1], len(regions)))
    for i in range(len(regions)):
        coords = regions[i]['segmentation'][0]
        coords = [[coords[2 * j + 1], coords[2 * j]] for j in range(int(len(coords)/2))]
        #mask = np.zeros((image.shape[0], image.shape[1]))
        #mask[tuple(zip(*coords))] = 1
        masks[:,:,i][tuple(zip(*coords))] = 1
    return masks

def random_crop(image, masks, min_dim=128):
    # crop image
    from numpy import random
    h, w = image.shape[:2]
    y = random.randint(0, (h - min_dim))
    x = random.randint(0, (w - min_dim))
    image = image[y:y + min_dim, x:x + min_dim]
    # crop masks    
    masks = masks[y:y + min_dim, x:x + min_dim]
    return image, masks

def imgaug_augmentation(image, masks, augmentation):
    import imgaug
    MASK_AUGMENTERS = ["Sequential", "SomeOf", "OneOf", "Sometimes",
                           "Fliplr", "Flipud", "CropAndPad",
                           "Affine", "PiecewiseAffine"]

    def hook(images, augmenter, parents, default):
        """Determines which augmenters to apply to masks."""
        return augmenter.__class__.__name__ in MASK_AUGMENTERS
    # Make augmenters deterministic to apply similarly to images and masks
    det = augmentation.to_deterministic()
    image = det.augment_image(image)
    # Change mask to np.uint8 because imgaug doesn't support np.bool
    masks = det.augment_image(masks.astype(np.uint8),
                                 hooks=imgaug.HooksImages(activator=hook))
    return image, masks

def masks_to_anno(masks):
    from detectron2.structures import BoxMode
    annotations = []
    for m in masks:
        py, px = np.where(m)
        poly = [(x, y) for x, y in zip(px, py)]
        poly = [p for x in poly for p in x]          
        obj = {"bbox": [np.min(px), np.min(py), np.max(px), np.max(py)],
                        "bbox_mode": BoxMode.XYXY_ABS,
                        "segmentation": [poly],
                        "category_id": 0,
                        "iscrowd": 0}
        annotations.append(obj) 
    return annotations

#%%
trainer = DefaultTrainer(cfg) 
trainer.resume_or_load(resume=False)
trainer.train()

#%%
"""
cfg.TEST.AUG.ENABLED = True
cfg.TEST.AUG.MIN_SIZES = 128
cfg.TEST.AUG.MAX_SIZE = 4000
cfg.TEST.AUG.FLIP = True
"""

cfg.MODEL.RPN.PRE_NMS_TOPK_TEST = 6000
cfg.MODEL.ROI_HEADS.BATCH_SIZE_PER_IMAGE = 3000
cfg.DATASETS.PRECOMPUTED_PROPOSAL_TOPK_TEST = 3000

cfg.MODEL.RPN.PRE_NMS_TOPK_TEST = 1000
cfg.MODEL.RPN.POST_NMS_TOPK_TEST = 500
cfg.MODEL.ROI_HEADS.BATCH_SIZE_PER_IMAGE = 200
cfg.TEST.DETECTIONS_PER_IMAGE = 100

cfg.MODEL.WEIGHTS = os.path.join(cfg.OUTPUT_DIR, "model_final.pth")
cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.5   # set the testing threshold for this model
cfg.DATASETS.TEST = ("neurons_val", )
cfg.DATALOADER.NUM_WORKERS = 10
predictor = DefaultPredictor(cfg)






#%%
op = ["train", "val"][1]
from detectron2.utils.visualizer import ColorMode
#from caiman.base.rois import nf_match_neurons_in_binary_masks
dataset_dicts = get_neurons_dicts(root_dir, op)
time = os.path.split(cfg.OUTPUT_DIR)[-1].split('_')[-1]
folder = os.path.join(root_dir, "picture", time, op)
try:
    os.makedirs(folder)
    print('make folder')
except:
    print('already exist')

d = dataset_dicts[2]
for d in random.sample(dataset_dicts, 1):
    im = cv2.imread(d["file_name"])
    outputs = predictor(im)
    v = Visualizer(im[:, :, ::-1],
                   metadata=neurons_metadata, 
                   scale=2, 
                   instance_mode=ColorMode.IMAGE_BW   # remove the colors of unsegmented pixels
    )
    v = v.draw_instance_predictions(outputs["instances"].to("cpu"))
    cv2.imshow('Detectron',v.get_image()[:, :, ::-1])
    cv2.waitKey(0)
    cv2.imwrite(folder + "/" + os.path.basename(d['file_name'])[:-4]+'_detect.png', v.get_image()[:, :, ::-1])
    cv2.destroyAllWindows()
    
    """    
    visualizer = Visualizer(im[:, :, ::-1], metadata=neurons_metadata, scale=2)
    vis = visualizer.draw_dataset_dict(d)
    cv2.imshow('Groundtruth',vis.get_image()[:, :, ::-1])
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    """
    
    print(d["file_name"])
    print("Total number of neurons from Groundtruth:", len(d["annotations"]))
    print("Total number of neurons from Detectron:", outputs["instances"].to("cpu").pred_masks.shape[0])
    
    mask_pr = np.asarray(outputs["instances"].to("cpu").pred_masks)*1.
    mask_gt = anno_to_mask(im, d["annotations"]).transpose([2, 0, 1])
    mask_gt = mask_gt*1.
    
    tp_gt, tp_comp, fn_gt, fp_comp, performance_cons_off = nf_match_neurons_in_binary_masks(
                    mask_gt, mask_pr, thresh_cost=0.7, min_dist=10, print_assignment=True,
                    plot_results=True, Cn=im[:,:,0], labels=['GT', 'MRCNN'])
    plt.savefig(folder + "/" + os.path.basename(d['file_name'])[:-4]+'_compare.pdf')
    plt.close()
    
    performance_cons_off["name"] = d["file_name"] 
    performance.append(performance_cons_off)  

#%%
#from skimage.draw import polygon

import matplotlib.pyplot as plt

op = ["train", "val"][1]
performance = []
folder = '/home/nel/Code/VolPy/Paper/detectron_pic/' + os.path.basename(cfg.OUTPUT_DIR)
dataset_dicts = get_neurons_dicts(dt + "/" + op)

try:
    os.makedirs(os.path.join(folder, op))
    print('make folder')
except:
    print('already exist')

for d in dataset_dicts:
    mask_gt = np.zeros((len(d["annotations"]), d["height"], d["width"]))
    for n in range(len(d["annotations"])):
        coords = (np.array(d["annotations"][n]["segmentation"]).reshape(-1, 2) - 0.5).astype(np.int16)
        rr, cc  = coords[:, 1], coords[:, 0]
        mask_gt[n, rr, cc] = 1
    
    im = cv2.imread(d["file_name"])
    outputs = predictor(im)
    mask_pr = np.asarray(outputs["instances"].to("cpu").pred_masks)*1.
    mask_gt = mask_gt*1.
    
    tp_gt, tp_comp, fn_gt, fp_comp, performance_cons_off = nf_match_neurons_in_binary_masks(
                    mask_gt, mask_pr, thresh_cost=0.7, min_dist=10, print_assignment=True,
                    plot_results=True, Cn=im[:,:,0], labels=['GT', 'MRCNN'])
    plt.savefig(folder + "/" + op + "/" +os.path.basename(d['file_name'])[:-4]+'_compare.pdf')
    plt.close()
    
    performance_cons_off["name"] = d["file_name"] 
    performance.append(performance_cons_off)  

#%%
performance
np.array([performance[i]["f1_score"] for i in range(len(performance))]).mean()

    
#%%
from detectron2.evaluation import COCOEvaluator, inference_on_dataset
from detectron2.data import build_detection_test_loader
evaluator = COCOEvaluator("balloon_val", cfg, False, output_dir="./output/")
val_loader = build_detection_test_loader(cfg, "balloon_val")
inference_on_dataset(trainer.model, val_loader, evaluator)
# another equivalent way is to use trainer.test

#%%
detectron2_neurons.__name


#%%
from skimage.draw import polygon
from caiman.base.rois import nf_match_neurons_in_binary_masks
import matplotlib.pyplot as plt

im = cv2.imread(d["file_name"])
outputs = predictor(im)
v = Visualizer(im[:, :, ::-1],
               metadata=neurons_metadata, 
               scale=2, 
               instance_mode=ColorMode.IMAGE_BW   # remove the colors of unsegmented pixels
)
v = v.draw_instance_predictions(outputs["instances"].to("cpu"))
cv2.imshow('p',v.get_image()[:, :, ::-1])
cv2.waitKey(0)
cv2.destroyAllWindows()












