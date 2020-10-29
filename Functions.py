import numpy as np 
import pandas as pd 
import os
import tensorflow as tf
import keras
from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D, AveragePooling2D
from keras.layers import Dense, Activation, Dropout, Flatten
from keras.preprocessing import image
from keras.preprocessing.image import ImageDataGenerator
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from keras.models import Sequential
from keras.layers import Dense , Activation , Dropout ,Flatten
from keras.layers.convolutional import Conv2D
from keras.layers.convolutional import MaxPooling2D
from keras.metrics import categorical_accuracy
from keras.models import model_from_json
from keras.callbacks import ModelCheckpoint
from keras.optimizers import *
from keras.layers.normalization import BatchNormalization
import seaborn as sn
from skimage import io
import cv2 #opencv-python
from time import time


def getData(filname):
    # images are 48x48
    # N = 35887
    Y = []
    X = []
    first = True
    for line in open(filname):
        if first:
            first = False
        else:
            row = line.split(',')
            Y.append(int(row[0]))
            X.append([int(p) for p in row[1].split()])

    X, Y = np.array(X) / 255.0, np.array(Y)
    return X, Y
def my_model():
    model = Sequential()
    input_shape = (48,48,1)
    model.add(Conv2D(64, (5, 5), input_shape=input_shape,activation='relu', padding='same'))
    model.add(Conv2D(64, (5, 5), activation='relu', padding='same'))
    model.add(BatchNormalization())
    model.add(MaxPooling2D(pool_size=(2, 2)))

    model.add(Conv2D(128, (5, 5),activation='relu',padding='same'))
    model.add(Conv2D(128, (5, 5),activation='relu',padding='same'))
    model.add(BatchNormalization())
    model.add(MaxPooling2D(pool_size=(2, 2)))

    model.add(Conv2D(256, (3, 3),activation='relu',padding='same'))
    model.add(Conv2D(256, (3, 3),activation='relu',padding='same'))
    model.add(BatchNormalization())
    model.add(MaxPooling2D(pool_size=(2, 2)))

    model.add(Flatten())
    model.add(Dense(128))
    model.add(BatchNormalization())
    model.add(Activation('relu'))
    model.add(Dropout(0.2))
    model.add(Dense(7))
    model.add(Activation('softmax'))

    model.compile(loss='categorical_crossentropy', metrics=['accuracy'],optimizer='adam')
    
    
    return model
def emotion_analysis(emotions):
    objects = ['angry', 'disgust', 'fear', 'happy', 'sad', 'surprise', 'neutral']
    y_pos = np.arange(len(objects))
    plt.bar(y_pos, emotions, align='center', alpha=0.9)
    plt.tick_params(axis='x', which='both', pad=10,width=4,length=10)
    plt.xticks(y_pos, objects)
    plt.ylabel('percentage')
    plt.title('emotion')
    plt.show()
def load_cascade_classifier_xml():
    return cv2.CascadeClassifier("haarcascade_frontalface_default.xml")
def find_faces_in_img(frame,faceCascade):
    
    frameg = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    faces = faceCascade.detectMultiScale(
        frameg,
        scaleFactor=1.1,
        minNeighbors=5,
        minSize=(48, 48),
        maxSize = (300,300)
        #flags = cv2.CV_HAAR_SCALE_IMAGE
    )

#     print("Found {0} faces!".format(len(faces)))

    # Draw a rectangle around the faces
    facecoordslist = []
    imglist = []
    for (x, y, w, h) in faces:
            tup = ( y, y+h,x, x+w)
            cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
            facecoordslist.append(tup)
    for (y1,y2,x1,x2) in facecoordslist:
        img= frameg[y1:y2,x1:x2]
        imglist.append(img)
    return facecoordslist,imglist,frame
def prep_image(img):
    img = cv2.resize(img, (48,48), 1, 1, interpolation=cv2.INTER_AREA)
#     print('frame post resize:',frame)
#     frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
#     print('gray',frame)
    img = img /255.0
#     print('gray/255',frame)
#     print('img pre img to array:',frame)
    img = image.img_to_array(img)
#     print('img to array x:',frame)
    img = np.expand_dims(img, axis = 0)
    return img
def predict_emotion(preppedfaceimg,model):
    objects = ['angry', 'disgust', 'fear', 'happy', 'sad', 'surprise', 'neutral']
    custom = model.predict(preppedfaceimg)
    m=0.000000000000000000001
    a=custom[0]
    for i in range(0,len(a)):
        if a[i]>m:
            m=a[i]
            ind=i
    return objects[ind]
def maketextaboveface(frame,text, coords):
    font = cv2.FONT_HERSHEY_SIMPLEX
#     cv2.imshow('mod input', frame[0])    
    cv2.putText(frame, text, (coords[0]+10, coords[1]), font, 1, (100,255,0), 1, cv2.LINE_AA)
    
def main():
    
    pass

if __name__ == "__main__":
   # stuff only to run when not called via 'import' here
   main()