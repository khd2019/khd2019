import numpy as np
import argparse
import os
import sys

def read_result_file(file):
    pred = []
    labels=[]
    with open (os.path.join(file),'rt')as f:
        for l in f:
            #sys.stdout.write('',l)
            label = l.split(',')
            pred.append(label[0])
            labels.append(label[1])
    return pred, labels

def evaluate(prediction, ground_truth): ## 이부분만 평가 기준에 맞게 변경할 것 
    zero_one =0
    zero_two =0
    zero_zero =0
    one_zero =0
    one_one =0
    one_two =0
    two_zero =0
    two_one =0
    two_two =0
    for y_hat, y_pred in zip(prediction,ground_truth):
        y_hat = str(y_hat).strip()
        y_pred = str(y_pred).strip()
        if y_pred == '0':
            if y_hat == '0':
                zero_zero +=1
            elif y_hat =='1':
                zero_one+=1
            else: zero_two +=1
        elif y_pred == '1':
            if y_hat == '0':
                one_zero +=1
            elif y_hat =='1':
                one_one+=1
            else: one_two +=1
        elif y_pred == '2':
            if y_hat == '0':
                two_zero +=1
            elif y_hat =='1':
                two_one+=1
            else: two_two +=1
    try: prec_zero = zero_zero / (zero_zero + zero_one+zero_two)
    except: prec_zero=0
    try: recall_zero = zero_zero / (zero_zero + one_zero+two_zero)    
    except: recall_zero=0
    try: prec_one = one_one / (one_zero + one_one+one_two)
    except: prec_one =0
    try: recall_one = one_one / (zero_one + one_one+two_one)    
    except: recall_one=0
    try: prec_two = two_two / (two_zero + two_one+two_two)
    except: prec_two=0 
    try: recall_two = two_two / (zero_two + one_two+two_two)    
    except: recall_two=0

    try:    f1_zero = 2*(prec_zero*recall_zero) / (prec_zero+recall_zero)
    except: f1_zero=0
    try: f1_one = 2*(prec_one*recall_one) / (prec_one+recall_one)
    except: f1_one=0
    try: f1_two = 2*(prec_two*recall_two) / (prec_two+recall_two)
    except: f1_two=0
    score = f1_zero*0.5 + f1_one*0.25 +f1_two *0.25

    return score

def evaluation_metrics(file):
    prediction, ground_truth = read_result_file(file) 
    return evaluate(prediction,ground_truth)

if __name__ == '__main__':
    args = argparse.ArgumentParser()
    args.add_argument('--prediction',type=str,default='pred.txt')
    config = args.parse_args()
    ### test_label 에 대한 접근은 반드시 evaluation.py에서만 할 것!! **가장 중요** 다른 py 파일에서 읽으면 오류남!!
    #test_label_path = 'data/khdmammo_e/test/test_label' ## full path를 써줄것!! data/[데이터 이름]/test/test_label
    print(evaluation_metrics(config.prediction))
