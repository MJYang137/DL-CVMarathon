# -*- coding: utf-8 -*-
"""
Created on Mon Jun  1 08:55:16 2020

@author: mingg
"""

from keras.models import Sequential
from keras.layers import Convolution2D
from keras.layers import Input, Dense
from keras.models import Model

##輸入照片尺寸==28*28*1
##都用一層，288個神經元

##建造一個一層的CNN層
classifier=Sequential()

##Kernel size 3*3，用32張，輸入大小28*28*1
classifier.add(Convolution2D(filters=32,kernel_size=(3,3),input_shape=(28,28,1)))
'''32張Kernel，大小為3*3，輸入尺寸為28*28*1'''
##看看model結構
print(classifier.summary())
'''理解輸出Total params為何==320'''
'''320 = 32*3*3+32'''

##建造一個一層的FC層
classifier=Sequential()
##輸入為28*28*1攤平==784
inputs = Input( shape=(784,) ) 
'''輸入尺寸為28*28*1'''
##CNN中用了(3*3*1)*32個神經元，我們這邊也用相同神經元數量
x=Dense(288)(inputs)
'''使用288個神經元'''
model = Model(inputs=inputs, outputs=x)
##看看model結構
print(model.summary())
'''理解輸出Total params為何==226080'''
'''226080 = 784*288+288'''


##大家可以自己變換設定看看參數變化
