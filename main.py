import os
import argparse
import sys
import time
import random
import keras
import cv2
import numpy as np


from keras.utils import np_utils
from sklearn.model_selection import train_test_split
from keras.models import Sequential
from keras.layers import Dense, Dropout
from keras.layers import Conv2D, MaxPooling2D, Flatten
from keras.layers import BatchNormalization, ReLU
from keras.preprocessing.image import ImageDataGenerator
from keras import optimizers
from keras.callbacks import ModelCheckpoint, ReduceLROnPlateau
from keras.utils.training_utils import multi_gpu_model
import keras.backend.tensorflow_backend as K
import nsml
from nsml.constants import DATASET_PATH, GPU_NUM

from model import cnn_sample
# from dataprocessing import image_preprocessing, dataset_loader


## setting values of preprocessing parameters
RESIZE = 10.
RESCALE = True


def bind_model(model):
    def save(dir_name):
        os.makedirs(dir_name, exist_ok=True)
        model.save_weights(os.path.join(dir_name, 'model'))
        # model.save_weights(file_path,'model')
        print('model saved!')

    def load(dir_name):
        model.load_weights(os.path.join(dir_name, 'model'))
        print('model loaded!')

    def infer(data, rescale=RESCALE, resize_factor=RESIZE):  ## test mode
        ##### DO NOT CHANGE ORDER OF TEST DATA #####
        X = []
        for i, d in enumerate(data):
            # test 데이터를 training 데이터와 같이 전처리 하기
            X.append(image_preprocessing(d, rescale, resize_factor))
        X = np.array(X)

        pred = model.predict_classes(X)     # 모델 예측 결과: 0-3
        print('Prediction done!\n Saving the result...')
        return pred

    nsml.bind(save=save, load=load, infer=infer)




def image_preprocessing(im, rescale=RESCALE, resize_factor=RESIZE):
    ## 이미지 크기 조정 및 픽셀 범위 재설정
    h, w, c = 3900, 3072, 3
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


def dataset_loader(img_path, rescale=RESCALE, resize_factor=RESIZE):

    t1 = time.time()
    print('Loading training data...\n')
    if not ((resize_factor == 1.) and (rescale == False)):
        print('Image preprocessing...')
    if not resize_factor == 1.:
        print('Image size is 3900*3072*3')
        print('Resizing the image into {}*{}*{}...'.format(int(3900//resize_factor), int(3072//resize_factor), 3))
    if not rescale == False:
        print('Rescaling range of 0-255 to 0-1...\n')

    ## 이미지 읽기
    p_list = [os.path.join(dirpath, f) for dirpath, dirnames, files in os.walk(img_path) for f in files if all(s in f for s in ['.jpg'])]
    p_list.sort()

    images = []
    labels = []
    for i, p in enumerate(p_list):
        im = cv2.imread(p, 3)
        if not (resize_factor == 1.):
            im = image_preprocessing(im, rescale=rescale, resize_factor=resize_factor)
        images.append(im)

        # label 데이터 생성
        l = Label2Class(p.split('/')[-2])
        labels.append(l)

    images = np.array(images)
    labels = np.array(labels)

    t2 = time.time()
    print('Dataset prepared for' ,t2 -t1 ,'sec')
    print('Images:' ,images.shape ,'np.array.shape(files, views, width, height)')
    print('Labels:', labels.shape, ' among 0-3 classes')

    return images, labels



if __name__ == '__main__':
    args = argparse.ArgumentParser()

    # hyperparameters
    args.add_argument('--epoch', type=int, default=10)                          # epoch 수 설정
    args.add_argument('--batch_size', type=int, default=8)                      # batch size 설정
    args.add_argument('--num_classes', type=int, default=4)                     # DO NOT CHANGE num_classes, class 수는 항상 4

    # DONOTCHANGE: They are reserved for nsml
    args.add_argument('--mode', type=str, default='train', help='submit일때 해당값이 test로 설정됩니다.')
    args.add_argument('--iteration', type=str, default='0',
                      help='fork 명령어를 입력할때의 체크포인트로 설정됩니다. 체크포인트 옵션을 안주면 마지막 wall time 의 model 을 가져옵니다.')
    args.add_argument('--pause', type=int, default=0, help='model 을 load 할때 1로 설정됩니다.')

    config = args.parse_args()

    seed = 1234
    np.random.seed(seed)

    # training parameters
    nb_epoch = config.epoch
    batch_size = config.batch_size
    num_classes = config.num_classes

    """ Model """
    
    learning_rate = 1e-4

    h, w = int(3900//RESIZE), int(3072//RESIZE)
    model = cnn_sample(in_shape=(h, w, 3), num_classes=num_classes)
    adam = optimizers.Adam(lr=learning_rate, decay=1e-5)                    # optional optimization
    sgd = optimizers.SGD(lr=learning_rate, momentum=0.9, nesterov=True)
    model.compile(loss='categorical_crossentropy', optimizer=sgd, metrics=['categorical_accuracy'])

    bind_model(model)
    if config.pause:  ## test mode일 때
        print('Inferring Start...')
        nsml.paused(scope=locals())

    if config.mode == 'train':  ### training mode일 때
        print('Training Start...')

        img_path = DATASET_PATH + '/train/'
        images, labels = dataset_loader(img_path, resize_factor=RESIZE, rescale=RESCALE)
        # containing optimal parameters

        ## data 섞기
        dataset = [[X, Y] for X, Y in zip(images, labels)]
        random.shuffle(dataset)
        X = np.array([n[0] for n in dataset])
        Y = np.array([n[1] for n in dataset])

        '''
        ## Augmentation 예시
        kwargs = dict(
            rotation_range=180,
            zoom_range=0.0,
            width_shift_range=0.0,
            height_shift_range=0.0,
            horizontal_flip=True,
            vertical_flip=True
        )
        train_datagen = ImageDataGenerator(**kwargs)
        train_generator = train_datagen.flow(x=X, y=Y, shuffle= False, batch_size=batch_size, seed=seed)
        # then flow and fit_generator....
        '''

        """ Callback """
        monitor = 'categorical_accuracy'
        reduce_lr = ReduceLROnPlateau(monitor=monitor, patience=3)

        """ Training loop """
        STEP_SIZE_TRAIN = len(X) // batch_size
        print('\n\nSTEP_SIZE_TRAIN = {}\n\n'.format(STEP_SIZE_TRAIN))
        t0 = time.time()

        ## data를 trainin과 validation dataset으로 나누기
        train_val_ratio = 0.8
        tmp = int(len(Y)*train_val_ratio)
        X_train = X[:tmp]
        Y_train = Y[:tmp]
        X_val = X[tmp:]
        Y_val = Y[tmp:]

        for epoch in range(nb_epoch):
            t1 = time.time()
            print("### Model Fitting.. ###")
            print('epoch = {} / {}'.format(epoch+1, nb_epoch))
            print('chaeck point = {}'.format(epoch))

            # for no augmentation case
            hist = model.fit(X_train, Y_train,
                             validation_data=(X_val, Y_val),
                             batch_size=batch_size,
                             #initial_epoch=epoch,
                             callbacks=[reduce_lr],
                             shuffle=True
                             )
            t2 = time.time()
            print(hist.history)
            print('Training time for one epoch : %.1f' % ((t2 - t1)))
            train_acc = hist.history['categorical_accuracy'][0]
            train_loss = hist.history['loss'][0]
            val_acc = hist.history['val_categorical_accuracy'][0]
            val_loss = hist.history['val_loss'][0]

            nsml.report(summary=True, step=epoch, epoch_total=nb_epoch, loss=train_loss, acc=train_acc, val_loss=val_loss, val_acc=val_acc)
            nsml.save(epoch)
        print('Total training time : %.1f' % (time.time() - t0))
        # print(model.predict_classes(X))



