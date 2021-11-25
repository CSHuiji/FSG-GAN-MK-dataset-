import sys
import os
import dlib
import numpy as np
import glob
import cv2
import pandas as pd

def init():
    #提供图片，获取landmark，人脸朝向。
    shape_predictor_path = "shape_predictor_68_face_landmarks.dat" #dlib 提供的人脸landmark predictor。
    face_detector = dlib.get_frontal_face_detector()
    shape_predictor = dlib.shape_predictor(shape_predictor_path)
    return face_detector, shape_predictor

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


def get_landmark(img, face_detector, shape_predictor):
    # shape = shape_predictor(img, dlib.rectangle(0, 0, img.shape[0], img.shape[1]))
    # return shape


    dets = face_detector(img, 1)
    if dets.__len__() != 1:
        #人脸结果错误，
        shape = shape_predictor(img, dlib.rectangle(38, 38, 218, 218))
        return shape
    #shape = shape_predictor(img, dlib.rectangle(0, 0, img.shape[0], img.shape[1]))
    shape = shape_predictor(img, dets[0])
    # win = dlib.image_window()
    # win.clear_overlay()
    # win.set_image(img)
    # win.add_overlay(shape)
    return shape


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


def modify_sun_glass(img_sun_glass, shape):
    img_occ_p1 = np.array([45, 35])
    img_occ_p2 = np.array([95, 35])
    img_occ_p3 = np.array([51, 70])
    img_occ_p4 = np.array([85, 70])
    img_occ_p  = np.array([66, 35])

    x_offset = img_occ_p[0] - (shape.part(38).x + shape.part(45).x) // 2
    y_offset = img_occ_p[1] - (shape.part(38).y + shape.part(45).x) // 2

    src = np.float32([img_occ_p1, img_occ_p3, img_occ_p2, img_occ_p4])
    dst = getDetArray(shape, 38, 48, 45, 54, x_offset, y_offset)
    M = cv2.getPerspectiveTransform(src, dst)
    new = cv2.warpPerspective(img_sun_glass, M, (img_sun_glass.shape[1], img_sun_glass.shape[0]),
                              borderMode=cv2.BORDER_CONSTANT)

    # cv2.imshow("A", new)
    # cv2.waitKey(0)
    # cv2.imwrite("E.png", new)
    return new,-x_offset,-y_offset


def add_Sun_visor():
    img_sum_visor = cv2.imread("sum_visor.png", -1) #带透明色的png 素材
    img_occ_x = img_sum_visor.shape[1]
    img_occ_y = img_sum_visor.shape[0]
    img_o_x_index = -(img_occ_x // 2)
    img_o_y_index = -127

    face_detector, shape_predictor = init()


    img_path = "B.png"
    img = cv2.imread(img_path, -1)

    shape = get_landmark(img, face_detector, shape_predictor)
    if not shape:
        print(img_path + "失败")
    else:
        x_index = shape.part(29).x + img_o_x_index
        y_index = shape.part(29).y + img_o_y_index

        img_out = add_overlap(img, img_sum_visor, x_index, y_index)
        cv2.imwrite("D.png", img_out)
        print(img_out)


def check_name(f_name, f_label = ""):
    if f_name.__len__() < 5:
        return False
    if f_label != "" and f_name[0:f_label.__len__()] == f_label:
        return False
    f_type = f_name[-3:].lower()
    if f_type not in ["png", "jpg"]:
        return False
    return True


def add_Mask():
    img_mask = cv2.imread("mask.png", -1) #带透明色的png 素材
    face_detector, shape_predictor = init()


    img_dir_path = "E:\\lmk0824\\data\\prepared_image\\img_align_celeba_crop\\"
    img_dir_path_out = "E:\\lmk0824\\data\\prepared_image\\img_align_mask\\"
    print_per = 200
    counter = 0

    if img_dir_path_out == "":
        img_dir_path_out = img_dir_path

    file_list = os.listdir(img_dir_path)

    #img_dir_path = "./"
    #file_list = ["000045.jpg", "000040.jpg", "000040.jpg", "000047.jpg", "000043.jpg"]

    for f_name in file_list:
        if check_name(f_name, "MASK"):
            img = cv2.imread(img_dir_path + f_name, -1)
            shape = get_landmark(img, face_detector, shape_predictor)
            if not shape:
                print(f_name + "失败")
            else:
                img_mask_modified, x_index, y_index = modify_mask(img_mask, shape)
                img_out = add_overlap(img, img_mask_modified, x_index, y_index)
                cv2.imwrite(img_dir_path_out + "MASK_" + f_name, img_out)
            counter = counter + 1
            if counter % print_per == 0:
                print(counter)


def add_sun_glass():
    img_sun_glass = cv2.imread("sun_glass.png", -1) #带透明色的png 素材
    face_detector, shape_predictor = init()


    img_dir_path = "E:\\lmk0824\\data\\prepared_image\\img_align_celeba_crop\\"
    img_dir_path_out = "E:\\lmk0824\\data\\prepared_image\\img_align_mask\\"
    print_per = 200
    counter = 0

    if img_dir_path_out == "":
        img_dir_path_out = img_dir_path

    file_list = os.listdir(img_dir_path)

    img_dir_path = "./"
    img_dir_path_out = img_dir_path
    file_list = ["A.png", "B.png"]

    for f_name in file_list:
        if check_name(f_name):
            img = cv2.imread(img_dir_path + f_name, -1)
            shape = get_landmark(img, face_detector, shape_predictor)
            if not shape:
                print(f_name + "失败")
            else:
                img_sun_glass_modified, x_index, y_index = modify_sun_glass(img_sun_glass, shape)
                img_out = add_overlap(img, img_sun_glass_modified, x_index, y_index)
                cv2.imwrite(img_dir_path_out + "SG_" + f_name, img_out)
            counter = counter + 1
            if counter % print_per == 0:
                print(counter)


def main(f_start=0, f_end=0):
    img_mask = cv2.imread("mask.png", -1)  # 带透明色的png 素材
    img_sun_glass = cv2.imread("sun_glass.png", -1)  # 带透明色的png 素材

    face_detector, shape_predictor = init()
    empty_img = np.zeros([256, 256, 3], np.uint8)
    empty_img = empty_img + 255

    img_sum_visor = cv2.imread("sum_visor.png", -1)  # 带透明色的png 素材
    img_occ_x = img_sum_visor.shape[1]
    img_occ_y = img_sum_visor.shape[0]
    img_o_x_index = -(img_occ_x // 2)
    img_o_y_index = -127

    img_dir_path = "D:\\SFTP\\MK\\MK_201104\\img_align_celeba\\"
    dir_path = {
        "sun_visor": "D:\\SFTP\\MK\\MK_201104\\celeba_test_set_sun_visor\\",
        "sun_glass": "D:\\SFTP\\MK\\MK_201104\\celeba_test_set_sun_glass\\",
        "mask_sv": "D:\\SFTP\\MK\\MK_201104\\celeba_test_set_mask_sv\\",
        "sun_visor_gt": "D:\\SFTP\\MK\\MK_201104\\celeba_test_set_sun_visor_gt\\",
        "sun_glass_gt": "D:\\SFTP\\MK\\MK_201104\\celeba_test_set_sun_glass_gt\\",
        "mask_sv_gt": "D:\\SFTP\\MK\\MK_201104\\celeba_test_set_mask_sv_gt\\",
    }
    for i in dir_path:
        if not os.path.exists(dir_path[i]):
            os.makedirs(dir_path[i])

    print_per = 200
    counter = 0

    #file_list = os.listdir(img_dir_path)
    file_list = read_list("CelebA_test.txt")
    file_list = file_list[0]


    # img_dir_path = "./"
    # file_list = ["000045.jpg", "000040.jpg", "000040.jpg", "000047.jpg", "000043.jpg"]
    file_counter = -1
    for img_name in file_list:
        if f_end != 0:
            file_counter = file_counter + 1
            print(file_counter)
            if file_counter < f_start:
                continue
            if file_counter > f_end:
                continue

        cur_img_path = os.path.join(img_dir_path, img_name)
        img_ori = cv2.imread(cur_img_path, -1)
        img = cv2.resize(img_ori, dsize=(256, 256))
        # cv2.imwrite(os.path.join(dir_path['test'], img_name), img)
        shape = get_landmark(img, face_detector, shape_predictor)
        if not shape:
            print(img_name + "失败")
        else:
            img_mask_modified, x_index, y_index = modify_mask(img_mask, shape)
            img_out = add_overlap(img, img_mask_modified, x_index, y_index)
            img_out_gt = add_overlap(empty_img, img_mask_modified, x_index, y_index)
            # cv2.imwrite(os.path.join(dir_path['mask'], img_name), img_out)
            # cv2.imwrite(os.path.join(dir_path['mask_gt'], img_name), img_out_gt)


            x_index_sv = shape.part(21).x + img_o_x_index
            y_index_sv = shape.part(21).y + img_o_y_index
            img_out = add_overlap(img_out, img_sum_visor, x_index_sv, y_index_sv)
            img_out_gt = add_overlap(img_out_gt, img_sum_visor, x_index_sv, y_index_sv)
            cv2.imwrite(os.path.join(dir_path['mask_sv'], img_name), img_out)
            cv2.imwrite(os.path.join(dir_path['mask_sv_gt'], img_name), img_out_gt)

            img_sun_glass_modified, x_index, y_index = modify_sun_glass(img_sun_glass, shape)
            img_out = add_overlap(img, img_sun_glass_modified, x_index, y_index)
            img_out_gt = add_overlap(empty_img, img_sun_glass_modified, x_index, y_index)
            cv2.imwrite(os.path.join(dir_path['sun_glass'], img_name), img_out)
            cv2.imwrite(os.path.join(dir_path['sun_glass_gt'], img_name), img_out_gt)

            x_index_sv = shape.part(29).x + img_o_x_index
            y_index_sv = shape.part(29).y + img_o_y_index
            img_out = add_overlap(img, img_sum_visor, x_index_sv, y_index_sv)
            img_out_gt = add_overlap(empty_img, img_sum_visor, x_index_sv, y_index_sv)
            cv2.imwrite(os.path.join(dir_path['sun_visor'], img_name), img_out)
            cv2.imwrite(os.path.join(dir_path['sun_visor_gt'], img_name), img_out_gt)

        counter = counter + 1
        if counter % print_per == 0:
            print(counter)


#main()