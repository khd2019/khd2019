import numpy as np
import argparse
import os

DATASET_PATH = 'data/KHD2019_FUNDUS'


def read_prediction(prediction_file):
    pred_array = np.loadtxt(prediction_file, dtype='str')
    return pred_array

def read_ground_truth(ground_truth_file):
    label_array = np.loadtxt(ground_truth_file, dtype='str')
    return label_array

def sensitivity(y_pred, y_gt, target):
    p = 0
    tp = 0
    for i, pred in enumerate(y_pred):
        if y_gt[i][-1] == target:
            p += 1
            if y_pred[i][-1] == target:
                tp += 1
    sens = tp / p
    return sens

def specificity(y_pred, y_gt, target):
    n = 0
    tn = 0
    for i, pred in enumerate(y_pred):
        if not y_gt[i][-1] == target:
            n += 1
            if not y_pred[i][-1] == target:
                tn += 1
    spec = tn / n
    return spec


def evaluate(num_classes, prediction, ground_truth): ## 이부분만 평가 기준에 맞게 변경할 것

    # performance
    class_acc = []
    class_sens = []
    class_spec = []

    for i in range(num_classes):
        class_sens.append(sensitivity(prediction, ground_truth, str(i)))
        class_spec.append(specificity(prediction, ground_truth, str(i)))
        class_acc.append((class_sens[i] + class_spec[i]) / 2)

    ttl_sens = sum(class_sens) / len(class_sens)
    ttl_spec = sum(class_spec) / len(class_spec)
    ttl_acc = sum(class_acc) / len(class_acc)

    return class_acc, class_sens, class_spec, ttl_acc, ttl_sens, ttl_spec


def evaluation_metrics(num_classes, prediction_file, ground_truth_file):
    prediction = read_prediction(prediction_file) 
    ground_truth = read_ground_truth(ground_truth_file)
    return evaluate(num_classes, prediction, ground_truth)

if __name__ == '__main__':
    args = argparse.ArgumentParser()
    args.add_argument('--prediction',type=str,default='pred.txt')       # 파일 이름 변경
    config = args.parse_args()
    num_classes = 4
    classes = ['Normal', 'AMD', 'RVO', 'DMR']

    test_path = os.path.join(DATASET_PATH, 'test', 'test_label')
    class_acc, class_sens, class_spec, ttl_acc, ttl_sens, ttl_spec = evaluation_metrics(num_classes, config.prediction, test_path)

    #for i in range(num_classes):
    #    print(' < Class {} >\nAccuracy= {}\n Sensitivity = {}\n Specificity = {}\n'.format(classes[i], class_acc[i], class_sens[i], class_spec[i]))
    #print(' << Total Performance >>\nAccuracy = {}\nSensitivity = {}\nSpecificity = {}'.format(ttl_acc, ttl_sens, ttl_spec))

    res_acc = round(ttl_acc, 4) * 100
    res_sens = round(ttl_sens, 4) / 100
    res_spec = round(ttl_spec, 4) / 1000000

    res = round(res_acc + res_sens + res_spec, 10)

    print(res)
