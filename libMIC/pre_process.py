"""Script for testing
Author: Guotai Wang
"""

import os
import sys
import csv
import math
import time
import random
import numpy as np
import pandas as pd
from PIL import Image
from scipy import ndimage
import matplotlib.pyplot as plt
from datetime import datetime
from random import shuffle


def remove_background(Im):
    """
     mask1: mask of white regon
     mask2: mask of black region (background)
     fill mask 2 with mean values of mask 1
    """
    Ir = Im[:, :, 0] 
    Ig = Im[:, :, 1]
    Ib = Im[:, :, 2]
    Ir = Ir * np.ones(Ir.shape)
    Ig = Ig * np.ones(Ig.shape)
    Ib = Ib * np.ones(Ib.shape)
    mask1 = Ig > Ig.mean()
    mask2 = Ib < Ib.mean()
    meanr = Ir[mask1].mean()* np.ones_like(Ir, np.float32)
    meang = Ig[mask1].mean()* np.ones_like(Ig, np.float32)
    meanb = Ib[mask1].mean()* np.ones_like(Ib, np.float32)
    Ir[mask2] = meanr[mask2]
    Ig[mask2] = meang[mask2]
    Ib[mask2] = meanb[mask2]
    I1 = np.asarray([Ir, Ig, Ib])
    I1 = np.transpose(I1, [1, 2, 0])
    I1 = np.asarray(I1, np.uint8)
    return I1 

def  remove_margin(Im):
    """
    remove margin for images from new system
    """
    shape = Im.shape
    H, W = shape[0], shape[1]
    w = 30
    m = 8
    Isub_list = []
    for i in range(6):
        Isub = Im[:, (w + m )*i : (w + m)*i + w, :]
        Isub_list.append(Isub)
    Isub = np.concatenate(Isub_list, axis = 1)
    return Isub

def preprocess_old_system():
    """
    preprocess of images for old system
    step 1 fill blck region with mean of white region
    step 2 resize the image in to 240 (W) x 200 (H)
    """
    folder = "D:/BaiduNetdiskDownload/phrosis_data/old_system"
    out_folder = "D:/BaiduNetdiskDownload/phrosis_data/old_process"
    filenames = os.listdir(folder)
    filenames = [item for item in filenames if ".jpg" in item]
    for i in range(len(filenames)):
        full_name = folder + '/' + filenames[i]
        print(filenames[i])
        I = Image.open(full_name)
        Im = np.asarray(I)
        Im = remove_background(Im)
        I1 = Image.fromarray(Im)
        I1 = I1.resize((240, 200),Image.BILINEAR)
        I1.save(out_folder + '/' + filenames[i])


def preprocess_new_system():
    """
    preprocess of images for new system
    step 1 crop region of interest
    step 2 fill blck region with mean of white region
    step 3 remove margin of each column
    step 4 resize the image in to 240 (W) x 200 (H)
    """
    folder = "D:/BaiduNetdiskDownload/phrosis_data/new_system"
    out_folder = "D:/BaiduNetdiskDownload/phrosis_data/new_process"
    filenames = os.listdir(folder)
    filenames = [item for item in filenames if ".bmp" in item]
    print("file number", len(filenames))
    for i in range(len(filenames)):
        full_name = folder + '/' + filenames[i]
        I = Image.open(full_name)
        Im = np.asarray(I)
        [W, H] = I.size
        assert(W == 240 and H == 210)
        Isub = Im[43:173, 10:230, :]   # output size 130 x 220
        Isub = remove_background(Isub)
        Isub = remove_margin(Isub)     # output size 130 x 180
        I1 = Image.fromarray(Isub)
        I1 = I1.resize((240, 200),Image.BILINEAR)

        save_name = out_folder + '/' + filenames[i][:-3] + 'png'
        I1.save(save_name)
        # plt.imshow(I1)
        # plt.show()

def write_list_as_file(input_list, filename):
    with open(filename, 'w') as f:
        for idx in range(len(input_list)):
            item = input_list[idx]
            if(idx < len(input_list) - 1):
                item = item + '\n'
            f.write(item)
    f.close()

def obtain_ground_truth_old_system():
    img_folder  =  "D:/BaiduNetdiskDownload/phrosis_data/old_system"
    gt_file     =  "D:/BaiduNetdiskDownload/phrosis_data/ground_truth_old_raw.txt"
    output_file =  "D:/BaiduNetdiskDownload/phrosis_data/ground_truth_old.txt"
    with open(gt_file, 'r') as f:
        dl_list_raw = f.readlines()
    dl_list_raw = [item.strip() for item in dl_list_raw]
    for item in dl_list_raw:
        print(item.split('\t'))
    print("Item number in the exel file {0:}".format(len(dl_list_raw)))

    dl_list = []
    file_names = os.listdir(img_folder)
    file_names = [item for item in file_names if ".jpg" in item]
    print('Image number', len(file_names))
    for file_name in file_names:
        file_name_short = file_name[:-12]
        gt_exist = False
        for idx in range(len(dl_list_raw)):
            temp_item = dl_list_raw[idx]
            temp_item_split = temp_item.split('\t')
            if(file_name_short == temp_item_split[0]):
                dl_list.append(dl_list_raw.pop(idx))
                gt_exist = True
                break
        if(gt_exist == False):
            print("{0:} gt not exist".format(file_name))
    print('Matched item number',len(dl_list))
    write_list_as_file(dl_list, output_file)

def split_dataset(old = True):
    random.seed(0)
    data_root = "D:/BaiduNetdiskDownload/phrosis_data"
    if(old):
        gt_file =  data_root + "/names/ground_truth_old.txt"
    else:
        gt_file =  data_root + "/names/ground_truth_new.txt"
    with open(gt_file, 'r') as f:
        dl_list= f.readlines()
    dl_list = [item.strip() for item in dl_list]
    shuffle(dl_list)
    N = len(dl_list)
    print('total number of data: ',N)
    ticks = [0.0, 0.2, 0.4, 0.6, 0.8, 1.0]
    for fold in range(5):
        N0 = int(N*ticks[fold])
        N1 = int(N*ticks[fold + 1])
        dl_list_sub = dl_list[N0:N1]
        gt_file_fold = gt_file[:-4] + "_fold_{0:}.txt".format(fold + 1)
        write_list_as_file(dl_list_sub, gt_file_fold)

def create_csv_files(old = True):
    data_root = "D:/BaiduNetdiskDownload/phrosis_data/"
    header = ['filename','IGGK', 'IGG^', 'IGMK', 'IGM^', \
              'IGAK', 'IGA^', 'lightK', 'light^', 'heavy']

    subgroup = 'old' if old else 'new'
    for fold in range(5):
        txt_name = data_root + 'names/ground_truth_{0:}_fold_{1:}.txt'.format(\
            subgroup, fold + 1)
        with open(txt_name, 'r') as f:
            dl_list= f.readlines()
        dl_list = [item.strip() for item in dl_list]
        dl_list_csv = []
        for item in dl_list:
            item_as_list = item.split('\t')
            assert(len(item_as_list) == 10)
            image_name = item_as_list[0]
            if(old):
                image_name = "old_process/{0:}DTouch64.jpg".format(image_name)
            else:
                image_name = "new_process/{0:}.png".format(image_name)
            item_as_list[0] = image_name
            # print(item_as_list)
            dl_list_csv.append(item_as_list)
    
        csv_name = data_root + 'names/{0:}_fold_{1:}.csv'.format(\
            subgroup, fold + 1)
        with open(csv_name, 'w') as csv_file:
            csv_writer = csv.writer(csv_file, delimiter = ',', lineterminator='\n',
                quotechar ='"', quoting=csv.QUOTE_MINIMAL)
            csv_writer.writerow(header)
            for csv_item in dl_list_csv:
                print(csv_item)
                csv_writer.writerow(csv_item)

def get_data_mean_and_std():
    img_dir = "D:/BaiduNetdiskDownload/phrosis_data/old_process"
    filenames = os.listdir(img_dir)
    filenames = [item for item in filenames if \
        ((".jpg" in item) or (".png" in item))]
    img_list = []
    for filename in filenames:
        full_name = "{0:}/{1:}".format(img_dir, filename)
        img = Image.open(full_name)
        img = np.asarray(img)
        img_list.append(img)
        # print(filename, img.shape)
    img_array = np.asarray(img_list)
    [N, H, W, C] = img_array.shape
    print('image array shape', img_array.shape)
    img_array = np.reshape(img_array, [N * H * W, C])
    mean = np.mean(img_array, axis = 0)
    std  = np.std(img_array, axis = 0)
    print('mean ', mean)
    print('std  ', std)
if __name__ == "__main__":
    # for old system
    # preprocess_old_system()
    # split_dataset(old = True)
    # create_csv_files(old = True)

    # for new system
    # preprocess_new_system()
    # obtain_ground_truth_old_system() 
    # split_dataset(old = False)
    create_csv_files(old = False)
    # get_data_mean_and_std()

    