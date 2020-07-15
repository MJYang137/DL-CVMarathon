#!/usr/bin/env python
# coding: utf-8

# # 作業

# ### 嘗試用 keras 的 DepthwiseConv2D 等 layers 實做 Inverted Residual Block.
#    - depthwise's filter shape 爲 (3,3), padding = same
#    - 不需要給 alpha, depth multiplier 參數
#    - expansion 因子爲 6

# ##### 載入套件

# In[11]:


from keras.models import Input, Model
from keras.layers import DepthwiseConv2D, Conv2D, BatchNormalization, ReLU, Add
from keras.activations import linear

# ##### 定義 Separable Convolution 函數 (請在此實做)

# In[1]:


def InvertedRes(input, expansion):
    '''
    Args:
        input: input tensor
        expansion: expand filters size
    Output:
        output: output tensor
    '''
    #Pointwise Convolution
    x = Conv2D(expansion*3,(1,1), padding='same')(input)
    x = BatchNormalization()(x)
    x = ReLU()(x) 
    #Depthwise Convolution
    x = DepthwiseConv2D((3,3), padding='same')(x)
    x = BatchNormalization()(x)
    x = ReLU()(x)
    #Pointwise Convolution
    x = Conv2D(3,(1,1))(x)
    x = BatchNormalization()(x)
    x = linear(x)

    x = Add()([x, input])
    
    return x
# ##### 建構模型

# In[21]:


input = Input((64, 64, 3))
output = InvertedRes(input, 6)
model = Model(inputs=input, outputs=output)
model.summary()


# In[ ]:





