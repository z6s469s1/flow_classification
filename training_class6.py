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

CLASS_DIC={0:"Normal_nonStreaming",1:"Normal_isStreaming",2:"VPN_nonStreaming",3:"VPN_isStreaming",4:"Tor_nonStreaming",5:"Tor_isStreaming"}
IMG_SHAPE=(200,200,1)
NUM_CLASS=6
    
TRAINING_DATA_PATH_VPN="/home/lab507hyc/disk4/workstation_40575018h/classification_streaming_VPN/dataset/training/"
TRAINING_DATA_PATH_TOR="/home/lab507hyc/disk4/workstation_40575018h/classification_streaming_Tor/dataset/training/"
TRAINING_DATA_PATH_NORMAL="/home/lab507hyc/disk4/workstation_40575018h/classification_streaming_Normal/dataset/training/"
    
TESTING_DATA_PATH_VPN="/home/lab507hyc/disk4/workstation_40575018h/classification_streaming_VPN/dataset/testing/"
TESTING_DATA_PATH_TOR="/home/lab507hyc/disk4/workstation_40575018h/classification_streaming_Tor/dataset/testing/"
TESTING_DATA_PATH_NORMAL="/home/lab507hyc/disk4/workstation_40575018h/classification_streaming_Normal/dataset/testing/"

EVALUATION_CNN_PATH="record/CNN-weights-improvement-07-0.80.hdf5"
EVALUATION_VGG16_PATH="record/VGG16-weights-improvement-10-0.86.hdf5"



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

def get_dataset(path1,path2,class_dic):
    print("Loading Dataset from "+path1)
    filenames=os.listdir(path1)
    imgs=[]
    labels=[]
    for i,filename in enumerate(filenames):
        if ("jpg" in filename) or("png" in filename):
            imgs.append(cv2.imread(path1+filename,cv2.IMREAD_UNCHANGED))
        
            label=""
            for key in class_dic.keys():
                if class_dic[key] in filename:
                    labels.append(key)
                    break
        
        print(str(i+1)+"/"+str(len(filenames)))
    
    print("Loading Dataset from "+path2)
    filenames=os.listdir(path2)
    for i,filename in enumerate(filenames):
        if ("jpg" in filename) or("png" in filename):
            imgs.append(cv2.imread(path2+filename,cv2.IMREAD_UNCHANGED))
        
            label=""
            for key in class_dic.keys():
                if class_dic[key] in filename:
                    labels.append(key)
                    break
        
        print(str(i+1)+"/"+str(len(filenames)))


    print("Loading Completed!")
    print("total loaded imgs:"+str(len(imgs)))

    imgs=np.array(imgs)
    labels=np.array(labels)
        
    
    return imgs,labels

def get_dataset(path1,path2,path3,class_dic):
    print("Loading Dataset from "+path1)
    filenames=os.listdir(path1)
    imgs=[]
    labels=[]
    for i,filename in enumerate(filenames):
        if ("jpg" in filename) or("png" in filename):
            imgs.append(cv2.imread(path1+filename,cv2.IMREAD_UNCHANGED))
        
            label=""
            for key in class_dic.keys():
                if class_dic[key] in filename:
                    labels.append(key)
                    break
        
        print(str(i+1)+"/"+str(len(filenames)))
    
    print("Loading Dataset from "+path2)
    filenames=os.listdir(path2)
    for i,filename in enumerate(filenames):
        if ("jpg" in filename) or("png" in filename):
            imgs.append(cv2.imread(path2+filename,cv2.IMREAD_UNCHANGED))
        
            label=""
            for key in class_dic.keys():
                if class_dic[key] in filename:
                    labels.append(key)
                    break
        
        print(str(i+1)+"/"+str(len(filenames)))


    print("Loading Dataset from "+path3)
    filenames=os.listdir(path3)
    for i,filename in enumerate(filenames):
        if ("jpg" in filename) or("png" in filename):
            imgs.append(cv2.imread(path3+filename,cv2.IMREAD_UNCHANGED))
        
            label=""
            for key in class_dic.keys():
                if class_dic[key] in filename:
                    labels.append(key)
                    break
        
        print(str(i+1)+"/"+str(len(filenames)))
    print("Loading Completed!")

    print("total loaded imgs:"+str(len(imgs)))

    imgs=np.array(imgs)
    labels=np.array(labels)
        
    
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
    
def training(class_dic,img_shape,num_class,training_data_path_VPN,training_data_path_Tor,training_data_path_Normal,testing_data_path_VPN,testing_data_path_Tor,testing_data_path_Normal):
    

    
    #load datasets
    x_train,y_train=get_dataset(training_data_path_VPN,training_data_path_Tor,training_data_path_Normal,class_dic)
    x_test,y_test=get_dataset(testing_data_path_VPN,testing_data_path_Tor,testing_data_path_Normal,class_dic)

    
    #data preprocessing
    x_train4D=x_train.reshape(x_train.shape[0],img_shape[0],img_shape[1],img_shape[2]).astype('float32')
    x_test4D=x_test.reshape(x_test.shape[0],img_shape[0],img_shape[1],img_shape[2]).astype('float32')
    
    x_train4D_normalize=x_train4D/255
    x_test4D_normalize=x_test4D/255
    
    y_trainOneHot=np_utils.to_categorical(y_train,len(class_dic.keys()))
    y_testOneHot=np_utils.to_categorical(y_test,len(class_dic.keys()))
 
    
    #setting model and training model
    #model=CNN(img_shape,num_class)
    model=VGG16(img_shape,num_class)
    print(model.summary())
    model.compile(loss='categorical_crossentropy',optimizer=optimizers.RMSprop(lr=1e-4),metrics=['accuracy'])
    
    # checkpoint
    filepath="weights-improvement-{epoch:02d}-{val_acc:.2f}.hdf5"
    checkpoint = ModelCheckpoint(filepath, monitor='val_acc', verbose=1, save_best_only=True,mode='max')
    callbacks_list = [checkpoint]
    training_history=model.fit(x=x_train4D_normalize,y=y_trainOneHot,validation_split=0.15,epochs=50,batch_size=128,verbose=1,callbacks=callbacks_list)


    score=model.evaluate(x_test4D_normalize,y_testOneHot)
    print(score)

    


def evaluation(class_dic,img_shape,num_class,testing_data_path_VPN,testing_data_path_Tor,testing_data_path_Normal,evaluation_CNN_path,evaluation_VGG16_path):
    
    x_test,y_test=get_dataset(testing_data_path_VPN,testing_data_path_Tor,testing_data_path_Normal,class_dic)

    #data preprocessing
    x_test4D=x_test.reshape(x_test.shape[0],img_shape[0],img_shape[1],img_shape[2]).astype('float32')
    x_test4D_normalize=x_test4D/255
    y_testOneHot=np_utils.to_categorical(y_test,len(class_dic.keys()))
 
    

    model=CNN(img_shape,num_class)
    model.compile(loss='categorical_crossentropy',optimizer=optimizers.RMSprop(lr=1e-4),metrics=['accuracy'])   
    model.load_weights(evaluation_CNN_path)
    score=model.evaluate(x_test4D_normalize,y_testOneHot)
    print("CNN score:")
    print(score)


    model=VGG16(img_shape,num_class)
    model.compile(loss='categorical_crossentropy',optimizer=optimizers.RMSprop(lr=1e-4),metrics=['accuracy'])   
    
    model.load_weights(evaluation_VGG16_path)
    score=model.evaluate(x_test4D_normalize,y_testOneHot)
    print("VGG16 score:")
    print(score)






def main():
    training(CLASS_DIC,IMG_SHAPE,NUM_CLASS,TRAINING_DATA_PATH_VPN,TRAINING_DATA_PATH_TOR,TRAINING_DATA_PATH_NORMAL,TESTING_DATA_PATH_VPN,TESTING_DATA_PATH_TOR,TESTING_DATA_PATH_NORMAL)
    #evaluation(CLASS_DIC,IMG_SHAPE,NUM_CLASS,TESTING_DATA_PATH_VPN,TESTING_DATA_PATH_TOR,TESTING_DATA_PATH_NORMAL,EVALUATION_CNN_PATH,EVALUATION_VGG16_PATH)       
    
    
    
    
    
    

    


    
    

if __name__=="__main__":
    main()
