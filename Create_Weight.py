import random
import os
import scipy.io as io
import nibabel as nib
import numpy as np


INPUT = '/shenlab/local/zhenghan/pj_wx/BtrflyNet/datasets/Label2D/train/'
OUTPUT = '/shenlab/local/zhenghan/pj_wx/BtrflyNet/datasets/Label2D/weight/'
INPUT_SEG = '/shenlab/local/zhenghan/original/seg/'

threshold = 0.6

def get_train_list(path):
    file_list_train = []

    for root, dirs, files in os.walk(path):
        for file in files:
            if file.startswith('.'):
                continue
            else:
                if (file[0:-4] == 'verse113') | (file[0:-4] == 'verse104') | (file[0:-4] == 'verse201'):
                    continue
                else:
                    file_list_train.append(os.path.join(file))

    wholesize = len(file_list_train)

    return file_list_train, wholesize

def get_gt_heatmap(file_list_train):
    front = []
    side = []
    for file in file_list_train:
        path = INPUT + file
        front.append((io.loadmat(path))['front'])
        side.append((io.loadmat(path))['side'])
    return front, side

def get_seg_data(file_list_train):
    seg = []

    for file in file_list_train:
        path = INPUT_SEG + file[0:-4] + '_seg.nii'
        imgseg = nib.load(path)
        seg.append(imgseg.get_fdata())
    return seg

def frequency_statistics(gt_heatmap_list):
    label_pixel = np.zeros(25)
    #background_pixel = np.zeros(25)
    all_pixel = 0
    for gt_heatmap in gt_heatmap_list:
        all_pixel += gt_heatmap.shape[0] * gt_heatmap.shape[1] ##* gt_heatmap.shape[2]
        #channel_pixel = gt_heatmap.shape[0] * gt_heatmap.shape[1]
        gt_label = np.zeros((gt_heatmap.shape[0],gt_heatmap.shape[1]))
        for i in range(24):
            label = i + 1
            gt_channel = np.where(gt_heatmap[:,:,label] > threshold, 1, 0)
            gt_label = np.where(gt_heatmap[:,:,label] > threshold, 1, gt_label)
            label_pixel[label] += gt_channel.sum()
            #background_pixel[label] += (channel_pixel - gt_label.sum())
        background = np.where(gt_label == 0, 1, 0)
        label_pixel[0] += background.sum()

    label_freq = label_pixel / float(all_pixel)
    #background_freq = background_pixel / float(all_pixel)

    return label_freq

def get_save_weighted_para(file_list, front_heatmap_list, side_heatmap_list, front_freq, side_freq, save_path):
    front_medium = np.median(front_freq)
    side_medium = np.median(side_freq)
    idx = 0
    for file in file_list:
        front_heatmap = front_heatmap_list[idx]
        side_heatmap = side_heatmap_list[idx]
        outpath = save_path + file


        w, h, c = front_heatmap.shape[0], front_heatmap.shape[1], front_heatmap.shape[2]
        front_weight = np.zeros(c)
        #gt_label = np.zeros((front_heatmap.shape[0], front_heatmap.shape[1]))
        for label in range(25):
            #label = i + 1
            front_weight[label] = front_medium/front_freq[label]
            #gt_label = np.where(front_heatmap[:,:,label] > threshold, 1, gt_label)
        #front_weight[:,:,0]= np.where(gt_label == 0, front_medium/front_freq[0], 0)

        w, h, c = side_heatmap.shape[0], side_heatmap.shape[1], side_heatmap.shape[2]
        side_weight = np.zeros(c)
        #gt_label = np.zeros((side_heatmap.shape[0], side_heatmap.shape[1]))
        for label in range(25):
            #label = i + 1
            side_weight[label] = side_medium / side_freq[label]
            #gt_label = np.where(side_heatmap[:, :, label] > threshold, 1, gt_label)
        #side_weight[:, :, 0] = np.where(gt_label == 0, side_medium / side_freq[0], 0)


        io.savemat(outpath,{'front':front_weight, 'side':side_weight})
        idx += 1






file_list_train, train_len = get_train_list(INPUT)
front, side = get_gt_heatmap(file_list_train)
#seg = get_seg_data(file_list_train)
front_label_freq = frequency_statistics(front)
side_label_freq = frequency_statistics(side)
front_label_medium = np.median(front_label_freq)
side_label_medium = np.median(side_label_freq)
front_weight = front_label_medium / front_label_freq
side_weight = side_label_medium / side_label_freq
io.savemat(OUTPUT + 'train_weight.mat',{'front':front_weight, 'side':side_weight})

#get_save_weighted_para(file_list_train, front, side, front_label_freq, side_label_freq, OUTPUT)


"""
testfront, testside, testsegfront, testsegside = front[1], side[1], np.max(seg[1],axis=0), np.max(seg[1],axis=2)

#segfront_20 = np.where(testsegfront == 20, 1, 0)
#segside_20 = np.where(testsegside == 20, 1, 0)
for i in range(7):
    index = i + 18
    front_20 = testfront[:, :, index]
    side_20 = testside[:, :, index]
    front_20_r = np.where(testsegfront == index, front_20, 3)
    side_20_r = np.where(testsegside == index, side_20, 3)
    front_min = np.min(front_20_r)
    side_min = np.min(side_20_r)
    print(front_min,side_min)
"""
a = 1
