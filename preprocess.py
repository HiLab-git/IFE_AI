"""Script for preprocessing
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

def remove_small_components(img): # 2D or 3D
    if(img.sum()==0):
        print('the largest component is null')
        return img
    if(len(img.shape) == 3):
        s = ndimage.generate_binary_structure(3,1) # iterate structure
    elif(len(img.shape) == 2):
        s = ndimage.generate_binary_structure(2,1) # iterate structure
    else:
        raise ValueError("the dimension number shoud be 2 or 3")

    labeled_array, numpatches = ndimage.label(img,s) # labeling
    label = np.zeros_like(img)
    for i in range(numpatches):
        temp_label = labeled_array == (i + 1)
        if(temp_label.sum() > 10):
            label = np.maximum(label, temp_label)
    return np.asarray(label, np.uint8)

def get_color_region(Im):
    """
    get the region of bands
    """
    Im = np.asarray(Im, np.float32)
    Ir = Im[:, :, 0] 
    Ig = Im[:, :, 1]
    Ib = Im[:, :, 2]
    diff = (Ir - Ib) * (Ir - Ib) + (Ig - Ib) * (Ig - Ib)
    mask = (diff > 200) * (Ib > 200)
    return mask

def remove_background(Im, b_color):
    """
    remove noise in the background of each lane,
    and set the intesnity values to b_color
    """
    Ir = Im[:, :, 0] 
    Ig = Im[:, :, 1]
    Ib = Im[:, :, 2]
    H, W = Ir.shape
    mask = get_color_region(Im)
    mask = ndimage.morphology.binary_closing(mask)
    mask = ndimage.morphology.binary_opening(mask)
    if(mask.sum() < 50):
        bg_mask = np.ones_like(mask)
    else:
        mask = remove_small_components(mask)
        indxes = np.nonzero(mask)
        h0 = max(indxes[0].min(), 0)
        h1 = min(indxes[0].max(), H)
        w0 = indxes[1].min()
        w1 = indxes[1].max()
        mask[np.ix_(range(h0, h1), range(w0, w1))] = np.ones((h1 - h0, w1 - w0))
        bg_mask = 1 - mask 

    meanr = b_color[0] * np.ones_like(Ig, np.float32)
    meang = b_color[1] * np.ones_like(Ig, np.float32)
    meanb = b_color[2] * np.ones_like(Ib, np.float32)
    Ir_out, Ig_out, Ib_out = Ir * 1, Ig * 1, Ib * 1 
    Ir_out[bg_mask > 0] = meanr[bg_mask > 0]
    Ig_out[bg_mask > 0] = meang[bg_mask > 0]
    Ib_out[bg_mask > 0] = meanb[bg_mask > 0]
    I1 = np.asarray([Ir_out, Ig_out, Ib_out])
    I1 = np.transpose(I1, [1, 2, 0])
    I1 = np.asarray(I1, np.uint8)
    return I1 

def  remove_margin_and_background(Im, w=30, m = 8):
    """
    remove margin and background for images from new system
    """
    shape = Im.shape
    Ir = Im[:, :, 0] 
    Ig = Im[:, :, 1]
    Ib = Im[:, :, 2]
    Ir_m = np.percentile(Ir, 50)
    Ig_m = np.percentile(Ig, 50)
    Ib_m = np.percentile(Ib, 50)

    H, W = shape[0], shape[1]
    Isub_list = []
    for i in range(6):
        Isub = Im[:, (w + m )*i : (w + m)*i + w, :]
        Isub = remove_background(Isub, [Ir_m, Ig_m, Ib_m])
        Isub_list.append(Isub)
    Isub = np.concatenate(Isub_list, axis = 1)
    return Isub
    
def preprocess_of_one_image(I, group = 'a'):
    """
    preprocess of images for new system
    step 1 crop the lanes with a margin
    step 2 for each lane, get the background region and replace 
           the color as their average to reduce noise
    step 3 resize the image in to 144 (W) x 144 (H)
    group: 'a' -- for images in the new system
           'b' -- for images in the old system
    """
    Im = np.asarray(I)
    [W, H] = I.size
    if(group == 'a'):
        assert(W == 240 and H == 210)
        Isub = Im[63:173, 10:230, :]   # output size 110 x 220
        Isub = remove_margin_and_background(Isub) # output size 110 x 180
    else:
        Isub = Im[35:, 7:-7, :] 
        Isub = remove_margin_and_background(Isub, w=50, m=0)
    I1 = Image.fromarray(Isub)
    I1 = I1.resize((144, 144),Image.BILINEAR)
    return I1

def preprocess_demo(group = 'a'):
    """
    demo of image preprocessing
    group: 'a' -- for images in the new system
           'b' -- for images in the old system
    """
    in_folder  = "data/data_{0:}".format(group)
    out_folder = in_folder + "_process"
    filenames = os.listdir(in_folder)
    filenames = [item for item in filenames if \
        ".bmp" in item or ".jpg" in item]
    for i in range(len(filenames)):
        print(filenames[i])
        full_name = in_folder + '/' + filenames[i]
        I  = Image.open(full_name)
        I1 = preprocess_of_one_image(I, group)
        plt.subplot(1,2,1); plt.imshow(I); plt.title("Original image")
        plt.subplot(1,2,2); plt.imshow(I1); plt.title("After preprocessing")
        plt.show()
        I1.save(out_folder + '/' + filenames[i])
    
if __name__ == "__main__":
    # group = 'a' for new system, 'b' for old system
    preprocess_demo(group = 'a') 
    preprocess_demo(group = 'b')
    

    