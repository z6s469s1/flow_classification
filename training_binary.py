import os
import pickle

import cv2
import numpy as np

import keras
from keras.models import Sequential
from keras.layers import Dense,Dropout,Flatten,Conv2D,MaxPooling2D
from keras.utils import np_utils
from keras.callbacks import EarlyStopping
from keras.callbacks import ModelCheckpoint
from keras import optimizers

from sklearn.metrics import accuracy_score
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from sklearn.metrics import f1_score



CLASS_DIC={0:"nonStreaming",1:"isStreaming"}
IMG_SHAPE=(200,200,1)
NUM_CLASS=1
TRAINING_DATA_PATH="dataset/training/"
TESTING_DATA_PATH="dataset/testing/"
EVALUATION_MODEL_PATH="record/weights-improvement-30-0.98.hdf5"


def get_dataset(path,class_dic):
    print("Loading Dataset....")
    filenames=os.listdir(path)
    imgs=[]
    labels=[]
    for i,filename in enumerate(filenames):
        if ("jpg" in filename) or("png" in filename):
            imgs.append(cv2.imread(path+filename,cv2.IMREAD_UNCHANGED))
        
            label=""
            for key in class_dic.keys():
                if class_dic[key] in filename:
                    labels.append(key)
                    break
        
        print(str(i+1)+"/"+str(len(filenames)))
        
    imgs=np.array(imgs)
    labels=np.array(labels)
    print("Loading Completed!")
        
    
    return imgs,labels

def VGG16(input_shape,num_class):
    model=Sequential()

    model.add(Conv2D(64, (3, 3), input_shape=input_shape, padding='same',activation='relu'))
    model.add(Conv2D(64, (3, 3), activation='relu', padding='same'))
    model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))
    model.add(Conv2D(128, (3, 3), activation='relu', padding='same'))
    model.add(Conv2D(128, (3, 3), activation='relu', padding='same'))
    model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))
    model.add(Conv2D(256, (3, 3), activation='relu', padding='same'))
    model.add(Conv2D(256, (3, 3), activation='relu', padding='same'))
    model.add(Conv2D(256, (3, 3), activation='relu', padding='same'))
    model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))
    model.add(Conv2D(512, (3, 3), activation='relu', padding='same'))
    model.add(Conv2D(512, (3, 3), activation='relu', padding='same'))
    model.add(Conv2D(512, (3, 3), activation='relu', padding='same'))
    model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))
    model.add(Conv2D(512, (3, 3), activation='relu', padding='same'))
    model.add(Conv2D(512, (3, 3), activation='relu', padding='same'))
    model.add(Conv2D(512, (3, 3), activation='relu', padding='same'))
    model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2), padding='same'))
    model.add(Flatten())
    model.add(Dense(4096, activation='relu'))
    model.add(Dense(4096, activation='relu'))
    model.add(Dense(num_class, activation='softmax'))
    
    return model


def CNN(input_shape,num_class):
    
    model = Sequential()
    model.add(Conv2D(32, (3, 3), activation='relu',input_shape=input_shape))
    model.add(MaxPooling2D((2, 2)))
    model.add(Conv2D(64, (3, 3), activation='relu'))
    model.add(MaxPooling2D((2, 2)))
    model.add(Conv2D(128, (3, 3), activation='relu'))
    model.add(MaxPooling2D((2, 2)))
    model.add(Conv2D(128, (3, 3), activation='relu'))
    model.add(MaxPooling2D((2, 2)))
    model.add(Flatten())
    model.add(Dropout(0.5))
    model.add(Dense(512, activation='relu'))
    model.add(Dense(num_class, activation='sigmoid'))
    
    return model
    
def training(class_dic,img_shape,num_class,training_data_path,testing_data_path):
   
    x_train,y_train=get_dataset(training_data_path,class_dic)
    x_test,y_test=get_dataset(testing_data_path,class_dic) 
    
    x_train4D=x_train.reshape(x_train.shape[0],img_shape[0],img_shape[1],img_shape[2]).astype('float32')
    x_test4D=x_test.reshape(x_test.shape[0],img_shape[0],img_shape[1],img_shape[2]).astype('float32')
    
    x_train4D_normalize=x_train4D/255
    x_test4D_normalize=x_test4D/255

    y_trainOneHot=y_train
    y_testOneHot=y_test
    
    
    model=CNN(img_shape,num_class)
    print(model.summary())

    model.compile(loss='binary_crossentropy',optimizer=optimizers.RMSprop(lr=1e-4),metrics=['accuracy'])
    
    
    
    # checkpoint
    filepath="weights-improvement-{epoch:02d}-{val_acc:.2f}.hdf5"
    checkpoint = ModelCheckpoint(filepath, monitor='val_acc', verbose=1, save_best_only=True,mode='max')
    callbacks_list = [checkpoint]
    training_history=model.fit(x=x_train4D_normalize,y=y_trainOneHot,validation_split=0.15,epochs=30,batch_size=128,verbose=1,callbacks=callbacks_list)

    
    
    
    # early_stopping
    """
    early_stopping = EarlyStopping(monitor='val_loss',mode='min', patience=50, verbose=1)
    model.compile(loss='categorical_crossentropy',optimizer='adam',metrics=['accuracy'])
    training_history=model.fit(x=x_train4D_normalize,y=y_trainOneHot,validation_split=0.15,epochs=20,batch_size=300,verbose=1,callbacks=[early_stopping])
    """
    
    
    model.save_weights('training_weights_CNN_b128.h5')
    model.load_weights('training_weights_CNN_b128.h5')
    
    
    with open('training_history_CNN_b128.pickle', 'wb') as file_pi:
        pickle.dump(training_history, file_pi)
        
    score=model.evaluate(x_test4D_normalize,y_testOneHot)
    print(score)
        
    
    
    

def evaluation(class_dic,img_shape,num_class,testing_data_path,evaluation_model_path):
      
    class_dic={0:"nonStreaming",1:"isStreaming"}
    img_shape=(200,200,1)
    num_class=1
    
    #load datasets
    testing_data_path="dataset/testing/"
    x_test,y_test=get_dataset(testing_data_path,class_dic)

    
    #data preprocessing
    x_test4D=x_test.reshape(x_test.shape[0],img_shape[0],img_shape[1],img_shape[2]).astype('float32')
    x_test4D_normalize=x_test4D/255
    y_testOneHot=y_test
    
    
    #setting model 
    model=CNN(img_shape,num_class)
    model.compile(loss='binary_crossentropy',optimizer=optimizers.RMSprop(lr=1e-4),metrics=['accuracy'])
    
    #load weights
    model.load_weights()


    #keras evaluation
    score=model.evaluate(x_test4D_normalize,y_testOneHot)
    print("score:")
    print(score)

    #sklearn evaluation
    # predict probabilities for test set
    yhat_probs = model.predict(x_test4D_normalize, verbose=0)
    # predict crisp classes for test set
    yhat_classes = model.predict_classes(x_test4D_normalize, verbose=0)
    # reduce to 1d array
    yhat_probs = yhat_probs[:, 0]
    yhat_classes = yhat_classes[:, 0]
    # accuracy: (tp + tn) / (p + n)
    accuracy = accuracy_score(y_testOneHot, yhat_classes)
    print('Accuracy: %f' % accuracy)
    # precision tp / (tp + fp)
    precision = precision_score(y_testOneHot, yhat_classes)
    print('Precision: %f' % precision)
    # recall: tp / (tp + fn)
    recall = recall_score(y_testOneHot, yhat_classes)
    print('Recall: %f' % recall)
    # f1: 2 tp / (2 tp + fp + fn)
    f1 = f1_score(y_testOneHot, yhat_classes)
    print('F1 score: %f' % f1)    
    
def main(): 
    training(CLASS_DIC,IMG_SHAPE,NUM_CLASS,TRAINING_DATA_PATH,TESTING_DATA_PATH)
    evaluation(CLASS_DIC,IMG_SHAPE,NUM_CLASS,TESTING_DATA_PATH,EVALUATION_MODEL_PATH)
        
    
    

    


    
    

if __name__=="__main__":
    main()
