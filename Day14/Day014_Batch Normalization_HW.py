# -*- coding: utf-8 -*-
"""
Created on Thu Jun  4 09:09:00 2020

@author: mingg
"""

from keras.models import Sequential  #用來啟動 NN
from keras.layers import Conv2D  # Convolution Operation
from keras.layers import MaxPooling2D # Pooling
from keras.layers import Flatten
from keras.layers import Dense # Fully Connected Networks
from keras.layers import BatchNormalization
from keras.layers import Activation


input_shape = (32, 32, 3)

model = Sequential()

##  Conv2D-BN-Activation('sigmoid') 

#BatchNormalization主要參數：
#momentum: Momentum for the moving mean and the moving variance.
#epsilon: Small float added to variance to avoid dividing by zero.

model.add(Conv2D(32, kernel_size=(3, 3),padding='same',input_shape = input_shape))
model.add(BatchNormalization(momentum=0.99,epsilon=0.001)) 
model.add(Activation('sigmoid'))


##、 Conv2D-BN-Activation('relu')
model.add(Conv2D(32, kernel_size=(3, 3),padding='same',input_shape = input_shape))
model.add(BatchNormalization(momentum=0.99,epsilon=0.001)) 
model.add(Activation('relu'))


model.summary()