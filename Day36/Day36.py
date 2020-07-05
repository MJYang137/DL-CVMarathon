#!/usr/bin/env python
# coding: utf-8

# ### Day36.YOLO 細節理解-網絡架構      
# 用實際的影像，嘗試自己搭建一個 1乘1和 3乘 3的模型    
# 看通過 1乘1和 3乘3 卷積層後會有甚麼變化?     
# 大家可以自己嘗試著搭建不同層數後，觀察圖形特徵的變化     

# In[ ]:


#宣告
import cv2
import matplotlib.patches as patches
import numpy as np
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')

from keras import backend as K
from keras.models import Sequential 
from keras.layers.core import Activation
from keras.layers.convolutional import Conv2D, MaxPooling2D


# In[47]:


##讀入照片
# 下載圖片範例，如果已經下載過就可以註解掉
#get_ipython().system('wget https://github.com/pjreddie/darknet/blob/master/data/dog.jpg?raw=true -O dog.jpg')
image=cv2.imread('dog.jpg')
#ax.imshow(image)

def show(img):
    plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB)) # plt.imshow 預設圖片是 rgb 的
    plt.show()
show(image)


# In[48]:


# create model
#Sequential 是一個多層模型
#透過 add() 函式將一層一層 layer 加上去
#data_format='channels_last' 尺寸为 (batch, rows, cols, channels)
#搭建一個 3 個 1*1 的 filters
model=Sequential()
model.add(Conv2D(3,
                 (1,1),
          padding="same",
         data_format='channels_last',
         activation='relu',
         input_shape=image.shape))
#作業: 接續搭建一個 4 個 3*3 的 filters 
model.add(Conv2D(4,
                 (3,3),
          padding="same",
         data_format='channels_last',
         activation='relu',
         input_shape=(576,768,3)))


print(model.summary())
#權重都是亂數值


# In[49]:


# keras 在讀取檔案實是以 batch 的方式一次讀取多張，
#但我們這裡只需要判讀一張，
#所以透過 expand_dims() 函式來多擴張一個維度
image_batch=np.expand_dims(image,axis=0)
print(image_batch.shape)


# In[50]:


#model.predict() 函式，得到回傳便是 feature map
image_conv=model.predict(image_batch)
img=np.squeeze(image_conv,axis=0)
print(img.shape)
plt.imshow(img)


# #### 由於權重都是亂數值，所以每次跑出來的結果不同
# 大家可以自己嘗試著搭建不同層數後，觀察圖形特徵的變化

