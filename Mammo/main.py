import os
import argparse
import sys
import time
import cv2
import numpy as np
import keras
from keras.utils import np_utils
from sklearn.model_selection import train_test_split
from keras.models import Sequential
from keras.layers import Dense, Dropout
from keras.layers import Conv2D, MaxPooling2D, Flatten
from keras.callbacks import  ModelCheckpoint, ReduceLROnPlateau
from keras.utils.training_utils import multi_gpu_model
import keras.backend.tensorflow_backend as K
import nsml
from keras.layers import BatchNormalization
from nsml.constants import DATASET_PATH, GPU_NUM


def bind_model(model):
    def save(dir_name):
        os.makedirs(dir_name, exist_ok=True)
        model.save_weights(os.path.join(dir_name, 'model'))
       # model.save_weights(file_path,'model')
        print('model saved!')

    def load(dir_name):
        model.load_weights(os.path.join(dir_name, 'model'))
        print('model loaded!')

    def infer(data): ## 해당 부분은 data loader의 infer_func을 의미
        X = preprocessing(data)
        pred = model.predict_classes(X)
        print('predicted')
        return pred

    nsml.bind(save=save, load=load, infer=infer)

def data_loader (root_path):
    t = time.time()
    print('Data loading...')
    data = [] # data path 저장을 위한 변수
    labels=[] # 테스트 id 순서 기록
    ## 하위 데이터 path 읽기
    for dir_name,_,_ in os.walk(root_path):
        try: 
            data_id = dir_name.split('/')[-1]
            int(data_id)    
        except: pass
        else: 
            data.append(np.load(dir_name+'/mammo.npz')['arr_0'])            
            labels.append(int(data_id[0]))
    data = np.array(data) ## list to numpy 
    labels = np.array(labels) ## list to numpy 
    print('Dataset Reading Success \n Reading time',time.time()-t,'sec')
    print('Dataset:',data.shape,'np.array.shape(files, views, width, height)')
    print('Labels:', labels.shape, 'each of which 0~2')
    return data, labels

def preprocessing(data): 
    print('Preprocessing start')
    # 자유롭게 작성해주시면 됩니다.
    X = np.concatenate([np.concatenate([data[:,0],data[:,1]],axis=1)
                    ,np.concatenate([data[:,2],data[:,3]],axis=1)],axis=2)
    X =  np.expand_dims(X, axis=3)
    X = X-X.min()/(X.max()-X.min())    
    print('Preprocessing complete...')
    print('The shape of X changed',X.shape)
    return X


def cnn_basic():
    model = Sequential()
    model.add(Conv2D(filters=4, kernel_size=(5, 5), activation='relu',strides=2, padding='same', input_shape=(820, 666,1))) ## shape size 정해주기
    model.add(MaxPooling2D(pool_size=(2, 2), padding='same',strides=2,))
    model.add(BatchNormalization())
    model.add(Conv2D(filters=8, kernel_size=(5, 5), activation='relu', padding='same',strides=2,))
    model.add(MaxPooling2D(pool_size=(2, 2), padding='same',strides=2,))
    model.add(BatchNormalization())
    model.add(Conv2D(filters=16, kernel_size=(5, 5), activation='relu', padding='same',strides=2,))
    model.add(MaxPooling2D(pool_size=(2, 2), padding='same',strides=2,))
    model.add(BatchNormalization())
    model.add(Flatten())
    model.add(Dense(256, activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(num_classes, activation='softmax'))
    
    return model

if __name__ == '__main__':
    args = argparse.ArgumentParser()

    # hyperparameters
    args.add_argument('--epoch', type=int, default=5)
    args.add_argument('--batch_size', type=int, default=32)
    args.add_argument('--num_classes', type=int, default=3)
    #args.add_argument('--width', type=int, default=100)
    #args.add_argument('--height', type=int, default=100)

    # DONOTCHANGE: They are reserved for nsml
    args.add_argument('--mode', type=str, default='train', help='submit일때 해당값이 test로 설정됩니다.')
    args.add_argument('--iteration', type=str, default='0',
                      help='fork 명령어를 입력할때의 체크포인트로 설정됩니다. 체크포인트 옵션을 안주면 마지막 wall time 의 model 을 가져옵니다.')
    args.add_argument('--pause', type=int, default=0, help='model 을 load 할때 1로 설정됩니다.')

    config = args.parse_args()

   # training parameters
    nb_epoch = config.epoch
    batch_size = config.batch_size
    num_classes = config.num_classes
    #width = config.width
    #height = config.height
    
    model = cnn_basic()
    model.summary()
    adam = keras.optimizers.Adam(lr=1e-7, decay=1e-5)
    model.compile(loss='categorical_crossentropy', optimizer=adam, metrics=['acc'])
    
    bind_model(model)
    if config.pause: ## test mode 일때는 여기만 접근
        print('Inferring Start...')
        nsml.paused(scope=locals())

    if config.mode == 'train': ### training mode 일때는 여기만 접근
        print('Training Start...')
        # train mode 일때, path 설정
        img_path = DATASET_PATH + '/train/'
        data, labels = data_loader(img_path)
        X = preprocessing(data)
        y = np_utils.to_categorical(labels, 3)
        
        """ Callback """
        monitor = 'acc'
        reduce_lr = ReduceLROnPlateau(monitor=monitor, patience=3)

        """ Training loop """
        t0 = time.time()
        for epoch in range(nb_epoch):
            print("### Model Fitting.. ###")
            hist = model.fit(X, y, 
                            batch_size=batch_size, 
                             callbacks=[reduce_lr]) 
            train_acc = hist.history['acc'][0]
            train_loss = hist.history['loss'][0]

            nsml.report(summary=True, step=epoch, epoch_total=nb_epoch, loss=train_loss, acc=train_acc)
            nsml.save(epoch)
        print('Total training time : %.1f' % (time.time() - t0))

