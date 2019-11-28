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
from dataprocessing import image_preprocessing, dataset_loader


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

    h, w = int(3072//RESIZE), int(3900//RESIZE)
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
            print('check point = {}'.format(epoch))

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



