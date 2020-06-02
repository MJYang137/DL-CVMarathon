# -*- coding: utf-8 -*-
"""
Created on Tue Jun  2 21:33:35 2020

@author: mingg
"""

from keras.models import Sequential
from keras.layers import Convolution2D
from keras.layers import Input, Dense
from keras.models import Model


##kernel size=(6,6)
##kernel數量：32

## Same padding、strides=(1,1)
classifier=Sequential()
inputs = Input(shape=(13,13,1))
x=Convolution2D(filters=32,kernel_size=(6,6), strides=(1, 1), padding = 'Same')(inputs)
model = Model(inputs=inputs, outputs=x)
model.summary()
''''output_size = 13 = ceil((13)/1)'''
## Same padding、strides=(2,2)
classifier=Sequential()
inputs = Input(shape=(13,13,1))
x=Convolution2D(filters=32,kernel_size=(6,6), strides=(2, 2), padding = 'Same')(inputs)
model = Model(inputs=inputs, outputs=x)
model.summary()
''''output_size = 4 = ceil((13)/2) = ceil(12.5)'''
## Valid padding、strides=(1,1)
classifier=Sequential()
inputs = Input(shape=(13,13,1))
x=Convolution2D(filters=32,kernel_size=(6,6), strides=(1, 1), padding = 'valid')(inputs)
model = Model(inputs=inputs, outputs=x)
model.summary()
''''output_size = 4 = (13-6+1)/1'''
## Valid padding、strides=(2,2)
classifier=Sequential()
inputs = Input(shape=(13,13,1))
x=Convolution2D(filters=32,kernel_size=(6,6), strides=(2, 2), padding = 'valid')(inputs)
model = Model(inputs=inputs, outputs=x)
model.summary()
''''output_size = 4 = (13-6+1)/2'''