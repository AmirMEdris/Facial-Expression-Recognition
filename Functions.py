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
from IPython import display
from IPython.display import clear_output
import PIL
import tensorflow as tf
import scipy
from tensorflow.keras import Sequential
from tensorflow.keras.layers import Conv2D,MaxPool2D,Dropout,Flatten,Dense,BatchNormalization
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.callbacks import EarlyStopping
from keras.models import Sequential
import glob
from keras import models
#from ann_visualizer.visualize import ann_viz;
import pickle as pkl
import numpy as np
import pandas as pd #more libraries and modules
import matplotlib.pyplot as plt
import six
np.random.seed(123)
# from tf_cnnvis import deepdream_visualization
import splitfolders
import seaborn as sns
from keract import get_activations, display_activations,display_heatmaps,get_gradients_of_activations
from tensorflow.keras.applications.inception_v3 import InceptionV3
from sklearn import metrics
import itertools
import numpy as np
import pandas as pd #more libraries and modules
import matplotlib.pyplot as plt
import six
from sklearn import metrics
import numpy as np
import pyautogui 
import matplotlib.pyplot as plt
import itertools
import matplotlib as mpl


import PIL.Image

from tensorflow.keras.preprocessing import image
def plot_confusion_matrix(cm, class_names):
    """
    Returns a matplotlib figure containing the plotted confusion matrix.
    
    Args:
       cm (array, shape = [n, n]): a confusion matrix of integer classes
       class_names (array, shape = [n]): String names of the integer classes
    """
    cm = np.array(cm)
    figure = plt.figure(figsize=(8, 8))
    plt.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
    plt.title("Confusion matrix")
    plt.colorbar()
    tick_marks = np.arange(len(class_names))
    plt.xticks(tick_marks, class_names, rotation=45)
    plt.yticks(tick_marks, class_names)
    
    # Normalize the confusion matrix.
    cm = np.around(cm.astype('float') / cm.sum(axis=1)[:, np.newaxis], decimals=2)
    
    # Use white text if squares are dark; otherwise black.
    threshold = cm.max() / 2.
    
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        color = "white" if cm[i, j] > threshold else "black"
        plt.text(j, i, cm[i, j], horizontalalignment="center", color=color)
        
    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    return 

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
def deprocess(img):
    img = 255*(img + 1.0)/2.0
    return tf.cast(img, tf.float32)

# Display an image
def show(img):
    display.display(PIL.Image.fromarray(np.array(tf.cast(img, tf.uint8))))
# # Create the feature extraction model
    
def calc_loss(img, model):
  # Pass forward the image through the model to retrieve the activations.
  # Converts the image into a batch of size 1.
    img_batch = tf.expand_dims(img, axis=0)
    layer_activations = model(img_batch)
    if len(layer_activations) == 1:
        layer_activations = [layer_activations]

    losses = []
    for act in layer_activations:
        loss = tf.math.reduce_mean(act)
        losses.append(loss)

    return  tf.reduce_sum(losses)
class DeepDream(tf.Module):
    def __init__(self, model):
        self.model = model

#     @tf.function(
#         input_signature=(
#         tf.TensorSpec(shape=[None,None,1], dtype=tf.float32),
#         tf.TensorSpec(shape=[], dtype=tf.int32),
#         tf.TensorSpec(shape=[], dtype=tf.float32),))
    def __call__(self, img, steps, step_size):
        print("Tracing")
        loss = tf.constant(0.0)
        for n in tf.range(steps):
            with tf.GradientTape() as tape:
                # This needs gradients relative to `img`
          # `GradientTape` only watches `tf.Variable`s by default
                tape.watch(img)
                loss = calc_loss(img, self.model)

        # Calculate the gradient of the loss with respect to the pixels of the input image.
                gradients = tape.gradient(loss, img)

        # Normalize the gradients.
                gradients = gradients*tf.math.reduce_std(gradients) + 1e-8 
        
        # In gradient ascent, the "loss" is maximized so that the input image increasingly "excites" the layers.
        # You can update the image by directly adding the gradients (because they're the same shape!)
                img = img + gradients*step_size
                show(img)
                print("Step {}, loss {}".format(step_size, loss))
                display.clear_output(wait=True)
#                 img = tf.clip_by_value(img, -1, 1)

        return loss, img
def run_deep_dream_simple(deepdream,img, steps=100, step_size=0.01):
      # Convert from uint8 to the range expected by the model.
    img = tf.convert_to_tensor(img)
    step_size = tf.convert_to_tensor(step_size)
    steps_remaining = steps
    step = 0
    while steps_remaining:
        if steps_remaining>100:
            run_steps = tf.constant(100)
        else:
            run_steps = tf.constant(steps_remaining)
        steps_remaining -= run_steps
        step += run_steps
    
        loss, img = deepdream(img, run_steps, tf.constant(step_size))
    
        display.clear_output(wait=True)
        show(img)
        print ("Step {}, loss {}".format(step, loss))


        result = img
        display.clear_output(wait=True)
        show(result)

    return result
def generatedeepdreamimg(imagebatch,model,imageposinbatch):
    base_model = model

    names = [name.name for name in  base_model.layers]
    layers = [base_model.get_layer(name).output for name in names[:]]

# Create the feature extraction model
    dream_model = tf.keras.Model(inputs=base_model.input, outputs=layers)

    original_img = np.array(imagebatch)
    original_img = original_img[imageposinbatch]
    original_img = ((original_img)).astype(np.float32)
    dream_model = tf.keras.Model(inputs=base_model.input, outputs=layers)

# show(deprocess(original_img))
    deepdream = DeepDream(dream_model)
    dream_img = run_deep_dream_simple(img=original_img, 
                                  steps=1000, step_size=0.5)
    return dream_img   
def modelaccuracy(model,X,y_test,label_map):
    y_pred=model.predict(X)
    y_test.shape

    ly_pred = []
    for i in y_pred:
        ly_pred.append(i.argmax())
    ly_test = []
    for i in y_test:
        ly_test.append(i.argmax())

    cm = tf.math.confusion_matrix(ly_pred,ly_test)

    cm=plot_confusion_matrix(cm,label_map)
    return cm
def main():
    
    pass

if __name__ == "__main__":
   # stuff only to run when not called via 'import' here
   main()