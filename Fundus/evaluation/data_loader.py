import os
import numpy as np
import sys
import time
import nsml
from nsml.constants import DATASET_PATH
import cv2

# 우리가 evaluation 하기 전에 결과를 불러오는 함수

def test_data_loader(img_path):

    print('Loading test data...')
    p_list = [os.path.join(dirpath, f) for dirpath, dirnames, files in os.walk(img_path) for f in files if all(s in f for s in ['.jpg'])]
    p_list.sort()

    images = []
    for i, p in enumerate(p_list):      # image 읽기
        im = cv2.imread(p, 3)
        images.append(im)

    return images, p_list


def Make_Pred_File(p_list, result, output_file):

    class_names = []
    IDs = []
    lines = []
    for i, p in enumerate(p_list):
        cn = p.split('/')[-2]
        class_names.append(cn)
        IDs.append(p.split('/')[-1])

        lines.append(class_names[i] + ',' + IDs[i] + ',' + result[i])

    print('Writing output...')
    with open(output_file, 'wt') as file_writer:
        file_writer.write('\n'.join(lines))

    if os.stat(output_file).st_size ==0:
        raise AssertionError('output result of inference is nothing')

    return 0


def feed_infer(output_file, infer_func):
    data, p_list = test_data_loader(os.path.join(DATASET_PATH,'test'))

    result = infer_func(data)
    result = [str(r) for r in result]

    Make_Pred_File(p_list, result, output_file)

