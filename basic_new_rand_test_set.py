import sys
import os
import random

import dlib
import numpy as np
import glob
import cv2
import pandas as pd
import tool as tl

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


# https://blog.csdn.net/wyx100/article/details/80541726
# 旋转angle角度，缺失背景白色（255, 255, 255）填充
def rotate_bound_white_bg(image, angle):
    # grab the dimensions of the image and then determine the
    # center
    (h, w) = image.shape[:2]
    (cX, cY) = (w // 2, h // 2)

    # grab the rotation matrix (applying the negative of the
    # angle to rotate clockwise), then grab the sine and cosine
    # (i.e., the rotation components of the matrix)
    # -angle位置参数为角度参数负值表示顺时针旋转; 1.0位置参数scale是调整尺寸比例（图像缩放参数），建议0.75
    M = cv2.getRotationMatrix2D((cX, cY), -angle, 1.0)
    cos = np.abs(M[0, 0])
    sin = np.abs(M[0, 1])

    # compute the new bounding dimensions of the image
    nW = int((h * sin) + (w * cos))
    nH = int((h * cos) + (w * sin))

    # adjust the rotation matrix to take into account translation
    M[0, 2] += (nW / 2) - cX
    M[1, 2] += (nH / 2) - cY

    # perform the actual rotation and return the image
    # borderValue 缺失背景填充色彩，此处为白色，可自定义
    return cv2.warpAffine(image, M, (nW, nH),borderValue=(255,255,255,0))
    # borderValue 缺省，默认是黑色（0, 0 , 0）
    # return cv2.warpAffine(image, M, (nW, nH))


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


def p2a(part, x_offset, y_offset):
    return [part.x + x_offset, part.y + y_offset]


def getDetArray(shape, lt_inde, lb_inde, rt_inde, rb_inde, x_offset, y_offset):
    return np.float32([p2a(shape.part(lt_inde), x_offset, y_offset), p2a(shape.part(lb_inde), x_offset, y_offset),
                       p2a(shape.part(rt_inde), x_offset, y_offset), p2a(shape.part(rb_inde), x_offset, y_offset)])


def modify_mask(img_mask, shape):
    img_occ_p1 = np.array([10, 15])
    img_occ_p2 = np.array([100, 20])
    img_occ_p3 = np.array([195, 25])
    img_occ_p4 = np.array([12, 65])

    img_occ_p6 = np.array([185, 87])
    img_occ_p7 = np.array([24, 105])
    img_occ_p8 = np.array([105, 163])
    img_occ_p9 = np.array([173, 120])

    total_cols = img_mask.shape[1]
    total_rows = img_mask.shape[0]
    crop_col_index = (img_occ_p2[0] + img_occ_p8[0]) // 2
    crop_row_index = (img_occ_p2[1] + img_occ_p8[1]) // 2

    img_occ_p5 = np.array([crop_col_index, crop_row_index])

    x_offset = img_occ_p5[0] - shape.part(62).x
    y_offset = img_occ_p5[1] - shape.part(62).y

    img_mask_1 = np.zeros(img_mask.shape, dtype="uint8")
    img_mask_2 = np.zeros(img_mask.shape, dtype="uint8")
    img_mask_3 = np.zeros(img_mask.shape, dtype="uint8")
    img_mask_4 = np.zeros(img_mask.shape, dtype="uint8")
    crop_off = 4
    img_mask_1[0:crop_row_index+crop_off, 0:crop_col_index+crop_off] = \
        img_mask[0:crop_row_index+crop_off, 0:crop_col_index+crop_off]
    img_mask_2[0:crop_row_index+crop_off, crop_col_index-crop_off:] = \
        img_mask[0:crop_row_index+crop_off, crop_col_index-crop_off:]
    img_mask_3[crop_row_index-crop_off:, 0:crop_col_index+crop_off] = \
        img_mask[crop_row_index-crop_off:, 0:crop_col_index+crop_off]
    img_mask_4[crop_row_index-crop_off:, crop_col_index-crop_off:] = \
        img_mask[crop_row_index-crop_off:, crop_col_index-crop_off:]

    src1 = np.float32([img_occ_p1, img_occ_p4, img_occ_p2, img_occ_p5])
    src2 = np.float32([img_occ_p2, img_occ_p5, img_occ_p3, img_occ_p6])
    src3 = np.float32([img_occ_p4, img_occ_p7, img_occ_p5, img_occ_p8])
    src4 = np.float32([img_occ_p5, img_occ_p8, img_occ_p6, img_occ_p9])

    dst1 = getDetArray(shape, 1, 3, 29, 62, x_offset, y_offset)
    dst2 = getDetArray(shape, 29, 62, 15, 13, x_offset, y_offset)
    dst3 = getDetArray(shape, 3, 5, 62, 8, x_offset, y_offset)
    dst4 = getDetArray(shape, 62, 8, 13, 11, x_offset, y_offset)

    M1 = cv2.getPerspectiveTransform(src1, dst1)
    M2 = cv2.getPerspectiveTransform(src2, dst2)
    M3 = cv2.getPerspectiveTransform(src3, dst3)
    M4 = cv2.getPerspectiveTransform(src4, dst4)

    new1 = cv2.warpPerspective(img_mask_1, M1, (img_mask_1.shape[1], img_mask_1.shape[0]),
                              borderMode=cv2.BORDER_CONSTANT)
    new2 = cv2.warpPerspective(img_mask_2, M2, (img_mask_2.shape[1], img_mask_2.shape[0]),
                              borderMode=cv2.BORDER_CONSTANT)
    new3 = cv2.warpPerspective(img_mask_3, M3, (img_mask_3.shape[1], img_mask_3.shape[0]),
                              borderMode=cv2.BORDER_CONSTANT)
    new4 = cv2.warpPerspective(img_mask_4, M4, (img_mask_4.shape[1], img_mask_4.shape[0]),
                              borderMode=cv2.BORDER_CONSTANT)

    # for i_y in range(img_mask_1.shape[0]):
    #     for i_x in range(img_mask_1.shape[1]):
    #         if new1[i_y][i_x][3] == 255:
    #             continue
    #         elif new1[i_y][i_x][3] == 0:
    #             if new2[i_y][i_x][3] != 0:
    #                 new1[i_y][i_x] = new2[i_y][i_x]
    #             elif new3[i_y][i_x][3] != 0:
    #                 new1[i_y][i_x] = new3[i_y][i_x]
    #             elif new4[i_y][i_x][3] != 0:
    #                 new1[i_y][i_x] = new4[i_y][i_x]
    #new1 = new1 + new2 + new3 + new4
    for i_y in range(img_mask_1.shape[0]):
        for i_x in range(img_mask_1.shape[1]):

            one_pix_div = sum([(new1[i_y][i_x][3] != 0), (new2[i_y][i_x][3] != 0),
                               (new3[i_y][i_x][3] != 0), (new4[i_y][i_x][3] != 0)])
            if one_pix_div > 1:
                # one_pix = [0., 0., 0., 0.] + new1[i_y][i_x] + \
                #           new2[i_y][i_x] + new3[i_y][i_x] + new4[i_y][i_x]
                one_pix = np.zeros([4,4])
                one_pix[0, :] = new1[i_y][i_x]
                one_pix[1, :] = new2[i_y][i_x]
                one_pix[2, :] = new3[i_y][i_x]
                one_pix[3, :] = new4[i_y][i_x]
                one_pix = np.max(one_pix, axis=0)

                # one_pix = new1[i_y][i_x] * 1. * new1[i_y][i_x][3] / 255. + \
                #           new2[i_y][i_x] * 1. * new2[i_y][i_x][3] / 255. + \
                #           new3[i_y][i_x] * 1. * new3[i_y][i_x][3] / 255. + \
                #           new4[i_y][i_x] * 1. * new4[i_y][i_x][3] / 255.
                #
                # one_pix = one_pix // one_pix_div
                one_pix[3] = 255
            else:
                one_pix = new1[i_y][i_x] + new2[i_y][i_x] + \
                          new3[i_y][i_x] + new4[i_y][i_x]
            new1[i_y][i_x] = one_pix

    #cv2.imshow("A", new1)
    #cv2.waitKey(0)
    #cv2.imwrite("E.png", new1)
    return new1,-x_offset,-y_offset


def check_name(f_name, f_label = ""):
    if f_name.__len__() < 5:
        return False
    if f_label != "" and f_name[0:f_label.__len__()] == f_label:
        return False
    f_type = f_name[-3:].lower()
    if f_type not in ["png", "jpg"]:
        return False
    return True


def main(f_start=0, f_end=0):
    random.seed(1)

    #素材读取 TODO 自动化。
    total_sc = 6
    img_sc = []
    for i in range(total_sc):
        img_sc.append(cv2.imread("SU/SU_T_" + str(i) + ".png", -1))

    empty_img = np.zeros([128, 128, 3], np.uint8)
    empty_img = empty_img + 255

#D:\SFTP\MK\lmk0824\data\prepared_image\img_align_celeba_crop
    img_dir_path = "D:\\SFTP\\MK\\lmk0824\\data\\prepared_image\\img_align_celeba_crop\\"
    dir_path_basic = "D:\\SFTP\\MK\\MK_210113_FGNH\\"
    dir_path = {
        "rand": dir_path_basic + "celeba_rand_tsst\\",
        "rand_gt": dir_path_basic + "celeba_rand_tsst_gt\\",
    }
    for i in dir_path:
        if not os.path.exists(dir_path[i]):
            os.makedirs(dir_path[i])

    print_per = 200
    counter = 0

    # file_list = os.listdir(img_dir_path)
    #file_list = read_list("identity_lfw.txt")


    # img_dir_path = "./"
    # file_list = ["000045.jpg", "000040.jpg", "000040.jpg", "000047.jpg", "000043.jpg"]
    file_counter = -1
    #total_number = file_list[0].__len__()

    file_list = os.listdir(img_dir_path)
    file_list = read_list("CelebA_test.txt")
    file_list = file_list[0]

    for f_name in file_list:
        if f_end != 0:
            file_counter = file_counter + 1
            print(file_counter)
            if file_counter < f_start:
                continue
            if file_counter > f_end:
                continue

        cur_img_path = os.path.join(img_dir_path, f_name)
        img = cv2.imread(cur_img_path, -1)


        #原图 128大小 宽度从 128-32
        rand_size = random.randint(60, 80)
        rand_angle = random.randint(0, 360 - 1)
        rang_sc_id = random.randint(0, total_sc - 1)
        rand_index_X = random.randint(30, 98 - 1)
        rand_index_y = random.randint(30, 98 - 1)

        picked_sc = np.zeros(img_sc[rang_sc_id].shape, dtype="uint8")
        picked_sc[:] = img_sc[rang_sc_id][:]
        picked_sc = rotate_bound_white_bg(picked_sc, rand_angle)
        (sc_h, sc_w) = picked_sc.shape[:2]
        if sc_h > sc_w:
            dsize_h = rand_size
            dsize_w = int(float(rand_size) * float(sc_w) / float(sc_h))
        else:
            dsize_w = rand_size
            dsize_h = int(float(rand_size) * float(sc_h) / float(sc_w))
        img_over = cv2.resize(picked_sc, dsize=(dsize_w, dsize_h))
        x_index = rand_index_X - int(dsize_w / 2)
        y_index = rand_index_y - int(dsize_h / 2)

        img_out = add_overlap(img, img_over, x_index, y_index)
        img_out_gt = add_overlap(empty_img, img_over, x_index, y_index)
        cv2.imwrite(os.path.join(dir_path['rand'], f_name), img_out)
        cv2.imwrite(os.path.join(dir_path['rand_gt'], f_name), img_out_gt)

        counter = counter + 1
        if counter % print_per == 0:
            print(counter)

main()