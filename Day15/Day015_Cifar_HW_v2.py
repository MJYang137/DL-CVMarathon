# -*- coding: utf-8 -*-
"""
Created on Thu Jun  4 21:20:26 2020

@author: mingg
"""

'''
==============================================================================
'''
dirc = r'C:\Users\mingg\Documents\GitHub\DL-CVMarathon\Day15'
model_name = 'cifarCnnModel'
model_weights_name = 'cifarCnnWeights'
Report_name = 'Day15'
'''
==============================================================================
'''

import csv, glob, os, sys,copy
import sys
import datetime




save_dir = os.path.join(dirc, 'saved_models')
model_path = os.path.join(save_dir, model_name+'.h5')
model_weights_path = os.path.join(save_dir, model_weights_name+'.h5')
history_path = os.path.join(save_dir, 'trainhistory.npy')



libdirc = r'C:\Users\mingg\Documents\GitHub\DL-CVMarathon\myfunctions'
os.chdir(libdirc)

import myDLCVfn as myfn



#from keras.models import Sequential
from keras.layers import Conv2D,MaxPooling2D,Flatten,Dense,Dropout,BatchNormalization,Activation
from keras.datasets import cifar10
import numpy as np
import tensorflow as tf
from sklearn.preprocessing import OneHotEncoder
import cv2
import matplotlib.pyplot as plt
import pandas as pd


(x_train, y_train), (x_test, y_test) = cifar10.load_data()

print(x_train.shape) #(50000, 32, 32, 3)
plt.imshow(x_train[0],cmap='binary')
lab_dict = {0:'airplane',1:'automobile',2:'bird',3:'cat',4:'deer',5:'dog',6:'frog',7:'horse',8:'ship',9:'truck'}
myfn.plot_imgs_labs_preds(x_train,y_train,[] ,lab_dict,0,12)


## Normalize Data
def normalize(X_train,X_test):
        mean = np.mean(X_train,axis=(0,1,2,3))
        std = np.std(X_train, axis=(0, 1, 2, 3))
        X_train = (X_train-mean)/(std+1e-7)
        X_test = (X_test-mean)/(std+1e-7) 
        return X_train, X_test,mean,std
    
    
## Normalize Training and Testset    
x_train_N, x_test_N ,mean_train,std_train = normalize(x_train, x_test) 

## OneHot Label 由(None, 1)-(None, 10)
## ex. label=2,變成[0,0,1,0,0,0,0,0,0,0]
one_hot=OneHotEncoder()
y_train_OH = one_hot.fit_transform(y_train).toarray()
y_test_OH = one_hot.transform(y_test).toarray()



'''
B
'''

inputs = tf.keras.Input( x_train.shape[1:])

conv1 = tf.keras.layers.Conv2D(32,(3,3),activation = 'relu')(inputs)
BN = tf.keras.layers.BatchNormalization(momentum=0.99,epsilon=0.001)(conv1)
MP1 = tf.keras.layers.MaxPooling2D(pool_size=(2,2))(BN)
DO0 = tf.keras.layers.Dropout(0.25)(MP1)
conv4 = tf.keras.layers.Conv2D(64,(3,3),activation = 'relu')(DO0)
DO1 = tf.keras.layers.Dropout(0.25)(conv4)
MP2 = tf.keras.layers.MaxPooling2D(pool_size=(2,2))(DO1)

Flt = tf.keras.layers.Flatten()(MP2)
DO2 = tf.keras.layers.Dropout(0.5)(Flt)
H = tf.keras.layers.Dense(512,activation='relu')(DO2)
DO3 = tf.keras.layers.Dropout(0.5)(H)
outputs = tf.keras.layers.Dense(10,activation='softmax')(DO3)

model = tf.keras.Model(inputs,outputs,name=model_name)
model.summary()

df_Keras_model = myfn.Keras_model_df(model)





'''
===============================================================================
trainning
===============================================================================
'''

#long running



model.compile(loss='categorical_crossentropy',optimizer = 'adam',metrics=['accuracy'])

from keras.models import load_model

import pickle


try:
    model.load_weights(model_weights_path)
    model = load_model(model_path)
    print("model loaded!")
except:
    print("New model training kick-off!")
    history = model.fit(x_train_N,y_train_OH,validation_split = 0.2,epochs=400,
                  batch_size=1000,verbose=2)
    if not os.path.isdir(save_dir):
        os.makedirs(save_dir)
    np.save(history_path,history.history)
    model.save_weights(model_weights_path)
    model.save(model_path)
    print("Save model to disk")


scores = model.evaluate(x_test_N,y_test_OH)

print('scores=',scores[1])

his = np.load(history_path,allow_pickle=True).item()

fig_train_his = myfn.myshow_train_history(his,'accuracy','val_accuracy')
fig_loss_his = myfn.myshow_train_history(his,'loss','val_loss')

'''







