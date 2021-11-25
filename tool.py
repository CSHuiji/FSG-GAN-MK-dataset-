import sys
import os
import dlib
import numpy as np
import glob
import cv2
import pandas as pd



def read_list(list_path):
    img_list = []
    label_list = []
    with open(list_path, 'r') as f:
        for line in f.readlines()[0:]:
            img_path = line.strip().split()
            img_list.append(img_path[0])
            label_list.append(img_path[1])
    print('There are {} images..'.format(len(img_list)))
    return img_list, label_list


def pix_weight_add(pix_back, pix_over, alpha):
    rate_back = (255 - alpha) / 255
    rate_over = alpha / 255
    pix_out = pix_back
    for i in range(3):
        pix_out[i] = pix_back[i] * rate_back + pix_over[i] * rate_over
        pix_out[i] = max(0, pix_out[i])
        pix_out[i] = min(255, pix_out[i])
        pix_out[i] = int(pix_out[i])
    return pix_out


def add_overlap(img_back, img_over, x_index, y_index):
    img_out = np.zeros(img_back.shape)
    img_out[:,:,:] = img_back[:,:,:]
    x_back_start = 0
    x_back_end = img_back.shape[1]
    y_back_start = 0
    y_back_end = img_back.shape[0]

    x_over_start = 0 + x_index
    x_over_end = img_over.shape[1] + x_index
    y_over_start = 0 + y_index
    y_over_end = img_over.shape[0] + y_index

    for_x_start = max(x_back_start, x_over_start)
    for_x_end = min(x_back_end, x_over_end)
    for_y_start = max(y_back_start, y_over_start)
    for_y_end = min(y_back_end, y_over_end)

    for i_y in range(for_y_start, for_y_end):
        for i_x in range(for_x_start, for_x_end):
            if img_over[i_y - y_index][i_x - x_index][3] == 255:
                img_out[i_y][i_x][0:3] = img_over[i_y - y_index][i_x - x_index][0:3]
            elif img_over[i_y - y_index][i_x - x_index][3] == 0:
                continue
            else:
                img_alpha = img_over[i_y - y_index][i_x - x_index][3]
                img_out[i_y][i_x][0:3] = pix_weight_add(img_out[i_y][i_x][0:3], img_over[i_y - y_index][i_x - x_index][0:3], img_alpha)

    return img_out
