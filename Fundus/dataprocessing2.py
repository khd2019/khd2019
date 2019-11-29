import os
import argparse
import sys
import time
import random
import keras
import cv2
import numpy as np



def image_preprocessing(im, rescale, resize_factor):
    ## 이미지 크기 조정 및 픽셀 범위 재설정
    h, w, c = 3072, 3900, 3
    nh, nw = int(h//resize_factor), int(w//resize_factor)
    # print(im.shape)

    res = cv2.resize(im, (nw, nh), interpolation=cv2.INTER_AREA)

    if rescale == True:
        res = res / 255.

    return res


def Label2Class(label):     # one hot encoding (0-3 --> [., ., ., .])

    resvec = [0, 0, 0, 0]
    if label == 'AMD':		cls = 1;    resvec[cls] = 1
    elif label == 'RVO':	cls = 2;    resvec[cls] = 1
    elif label == 'DMR':	cls = 3;    resvec[cls] = 1
    else:					cls = 0;    resvec[cls] = 1		# Normal

    return resvec


def dataset_loader(img_path, idx, rescale, resize_factor):

    t1 = time.time()
    print('Loading training data...\n')
    if not ((resize_factor == 1.) and (rescale == False)):
        print('Image preprocessing...')
    if not resize_factor == 1.:
        print('Image size is 3900*3072*3')
        print('Resizing the image into {}*{}*{}...'.format(int(3072//resize_factor), int(3900//resize_factor), 3))
    if not rescale == False:
        print('Rescaling range of 0-255 to 0-1...\n')

    ## 이미지 읽기
    nor_list = [os.path.join(dirpath, f) for dirpath, dirnames, files in os.walk(img_path, 'NOR') for f in files if all(s in f for s in ['NOR', '.jpg'])]
    amd_list = [os.path.join(dirpath, f) for dirpath, dirnames, files in os.walk(img_path, 'AMD') for f in files if all(s in f for s in ['AMD', '.jpg'])]
    dmr_list = [os.path.join(dirpath, f) for dirpath, dirnames, files in os.walk(img_path, 'DMR') for f in files if all(s in f for s in ['DMR', '.jpg'])]
    rvo_list = [os.path.join(dirpath, f) for dirpath, dirnames, files in os.walk(img_path, 'RVO') for f in files if all(s in f for s in ['RVO', '.jpg'])]

    nor_list.sort()
    amd_list.sort()
    dmr_list.sort()
    rvo_list.sort()

    idx_step = int(len(nor_list)/10)
    if idx == 9:
        nor_list = nor_list[idx*idx_step:]
    else:  
        nor_list = nor_list[idx*idx_step:(idx+1)*idx_step]

    print("@@@@ 1. len(nor_list):" , len(nor_list))
    print("@@@@ 1. idx_step:" , idx_step)

    idx_step = int(len(amd_list)/10)
    if idx == 9:
        amd_list = amd_list[idx*idx_step:]
    else: 
        amd_list = amd_list[idx*idx_step:(idx+1)*idx_step]

    print("@@@@ 2. len(nor_list):" , len(amd_list))
    print("@@@@ 2. idx_step:" , idx_step)

    idx_step = int(len(rvo_list)/10)
    if idx == 9:
        rvo_list = rvo_list[idx*idx_step:]
    else:  
        rvo_list = rvo_list[idx*idx_step:(idx+1)*idx_step]

    print("@@@@ 3. len(nor_list):" , len(rvo_list))
    print("@@@@ 3. idx_step:" , idx_step)

    idx_step = int(len(dmr_list)/10)
    if idx == 9:
        dmr_list = dmr_list[idx*idx_step:]
    else:  
        dmr_list = dmr_list[idx*idx_step:(idx+1)*idx_step]

    print("@@@@ 4. len(nor_list):" , len(dmr_list))
    print("@@@@ 4. idx_step:" , idx_step)


    p_list = []
    for i, p in enumerate (nor_list):
        p_list.append(p)
    for i, p in enumerate (amd_list):
        p_list.append(p)
    for i, p in enumerate (rvo_list):
        p_list.append(p)
    for i, p in enumerate (dmr_list):
        p_list.append(p)

    images = []
    labels = []
    num_data = len(p_list)
    for i, p in enumerate(p_list):
        im = cv2.imread(p, 3)
        if not (resize_factor == 1.):
            im = image_preprocessing(im, rescale=rescale, resize_factor=resize_factor)
        images.append(im)

        # label 데이터 생성
        l = Label2Class(p.split('/')[-2])
        labels.append(l)

        print(i+1, '/', num_data, ' image(s)')

    images = np.array(images)
    labels = np.array(labels)

    t2 = time.time()
    print('Dataset prepared for' ,t2 -t1 ,'sec')
    print('Images:' ,images.shape ,'np.array.shape(files, views, width, height)')
    print('Labels:', labels.shape, ' among 0-3 classes')

    return images, labels


