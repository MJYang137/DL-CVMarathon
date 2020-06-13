# -*- coding: utf-8 -*-
"""
Created on Sat Jun 13 14:42:50 2020

@author: mingg
"""

from keras.models import Sequential
from keras.layers import Convolution2D
from keras.layers import MaxPooling2D
from keras.layers import Flatten
from keras.layers import Dense
from keras.layers import Dropout
from keras.preprocessing import image
from keras.models import Model
from keras.layers import Dense, GlobalAveragePooling2D
from keras import backend as K
import keras
from keras.layers import Input
 
from keras.datasets import cifar10
import numpy as np
import tensorflow as tf
from sklearn.preprocessing import OneHotEncoder


input_tensor = Input(shape=(32, 32, 3))
#include top 決定要不要加入 fully Connected Layer

'''Xception 架構'''

model_Xception = keras.applications.Xception(
    include_top=False,
    weights="imagenet",
    input_tensor=input_tensor,
    pooling=None,
    classes=10)
'''Resnet 50 架構'''
#model=keras.applications.ResNet50(include_top=False, weights='imagenet',
                                    #input_tensor=input_tensor,
                                    #pooling=None, classes=10)
model_Xception.summary()
model_Xception.layers
#添加層數

x = model_Xception.output

'''可以參考Cifar10實作章節'''
x = GlobalAveragePooling2D()(x)
x = Dense(output_dim=128, activation='relu')(x)
x=Dropout(p=0.1)(x)
predictions = Dense(output_dim=10,activation='softmax')(x)
model = Model(inputs=model_Xception.input, outputs=predictions)
print('Xception深度：', len(model_Xception.layers),'; transfer model 深度',len(model.layers))

#鎖定特定幾層不要更新權重

for layer in model.layers[:len(model_Xception.layers)]:
    layer.trainable = False
for layer in model.layers[len(model_Xception.layers):]:
    layer.trainable = True
#準備 Cifar 10 資料

(x_train, y_train), (x_test, y_test) = cifar10.load_data()

print(x_train.shape) #(50000, 32, 32, 3)

## Normalize Data
def normalize(X_train,X_test):
        mean = np.mean(X_train,axis=(0,1,2,3))
        std = np.std(X_train, axis=(0, 1, 2, 3))
        X_train = (X_train-mean)/(std+1e-7)
        X_test = (X_test-mean)/(std+1e-7)
        return X_train, X_test
    
    
## Normalize Training and Testset    
x_train, x_test = normalize(x_train, x_test) 

## OneHot Label 由(None, 1)-(None, 10)
## ex. label=2,變成[0,0,1,0,0,0,0,0,0,0]
one_hot=OneHotEncoder()
y_train=one_hot.fit_transform(y_train).toarray()
y_test=one_hot.transform(y_test).toarray()

#training

# compile the model (should be done *after* setting layers to non-trainable)
model.compile(optimizer = 'adam', loss = 'categorical_crossentropy', metrics = ['accuracy'])
model.fit(x_train,y_train,batch_size=32,epochs=10)