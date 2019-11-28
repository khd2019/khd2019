import os
import numpy as np
import sys
import time
import nsml
from nsml.constants import DATASET_PATH


def test_data_loader (root_path): ## main loader를 사용해도 무관하나/ 현재 main 코드에서는 data와 label을 동시에 읽기 때문에 따로 만들어줌
    data = [] # data path 저장을 위한 변수
    labels=[] # 테스트 id 순서 기록 
    for dir_name,_,_ in os.walk(root_path):
        try: 
            data_id = dir_name.split('/')[-1]
            int(data_id)    
        except: pass
        else: 
            data.append(np.load(dir_name+'/mammo.npz')['arr_0'])            
            labels.append(int(data_id[0]))
    data = np.array(data) ## list to numpy
    return data, labels


def feed_infer(output_file, infer_func):
    data, labels = test_data_loader(os.path.join(DATASET_PATH,'test'))
    result = infer_func(data)
    print('write output')
    with open(output_file, 'wt') as file_writer:
        file_writer.write('\n'.join('%s, %s' % x for x in zip(result, labels)))
    if os.stat(output_file).st_size ==0:
        raise AssertionError('output result of inference is nothing')
