#!/usr/bin/env python
# coding: utf-8

# # Object Detection with SSD
# ### 在了解具體如何操作前，我們先來看看SSD效果

# In[2]:


import os
import sys
module_path = os.path.abspath(os.path.join('..'))
if module_path not in sys.path:
    sys.path.append(module_path)


dirc = r'C:\Users\mingg\Documents\GitHub\DL-CVMarathon\Day29\Object Detection'
os.chdir(dirc)


import json

import torch
import torch.nn as nn
import torch.backends.cudnn as cudnn
from torch.autograd import Variable
import numpy as np
import cv2

torch.set_default_tensor_type('torch.FloatTensor')


# if torch.cuda.is_available():
#     torch.set_default_tensor_type('torch.cuda.FloatTensor')
from ssd import build_ssd

cocoa = r'C:\Users\mingg\Documents\GitHub\DL-CVMarathon\Day29\annotations_trainval2014\annotations'

item = r'captions_train2014.json'

cocopath = os.path.join(cocoa,item)

with open(cocopath) as f:
  data = json.load(f)

data.keys()

# ## 建立整個模型(SSD 300)
#  

# #### 這個預訓練權重是Train在VOC資料集上
# #### 輸入影像是300*300，一共有20類+背景=21類
# #### 先將模型架起

# In[3]:


net = build_ssd('test', 300, 21)    # initialize SSD


# ## 將模型架構打印出來看

# In[4]:


print(net)


# ## 載入預訓練權重

# In[5]:


net.load_weights('ssd300_mAP_77.43_v2.pth')


# ## Load Image 
# ### Here we just load a sample image from the VOC07 dataset 

# In[17]:


image = cv2.imread('./data/example.jpg', cv2.IMREAD_COLOR)  # uncomment if dataset not downloaded

from matplotlib import pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')
import numpy as np
import cv2
from PIL import Image, ImageFont, ImageDraw
image = Image.open(r'C:\Users\mingg\Documents\GitHub\DL-CVMarathon\Day29\Object Detection\demo\golden-retriever-puppy.jpg')
plt.imshow(image)
plt.show()
#image = image.resize((300,300), Image.ANTIALIAS)
image=np.asarray(image)
rgb_image=np.asarray(image)
# View the sampled input image before transform


# ## 輸入影像的預處理，與Training 時需要相同
# 

# In[18]:


'''先Resize 成 300*300'''
x = cv2.resize(image, (300, 300)).astype(np.float32)
'''Vgg 16的預處理方式'''
x -= (104.0, 117.0, 123.0)
'''將type轉乘np.float32'''
x = x.astype(np.float32)
'''將RGB轉換為BGR'''
x = x[:, :, ::-1].copy()
plt.imshow(x)
plt.show()
'''轉換成torch 的 tensor，並且將 H,W,C 轉成 C,H,W'''
x = torch.from_numpy(x).permute(2, 0, 1)


# ## 將處理好之影像送入模型運算(Forward Pass)

# In[8]:


'''要先用Variable包裝才能送給Pytorch模型'''
xx = Variable(x.unsqueeze(0))     # wrap tensor in Variable
# if torch.cuda.is_available():
#     xx = xx.cuda()
'''Forward Pass'''
y = net(xx)


# In[9]:


'''Batch Size,類別數,top 200個框, 5=(delta x,delta y,delta h, delta w, confidence)'''
print(y.shape)


# ## 預測結果
# Filter outputs with confidence scores lower than a threshold 
# Here we choose 60% 

# In[14]:


from data import VOC_CLASSES as labels
top_k=10

plt.figure(figsize=(10,10))
colors = plt.cm.hsv(np.linspace(0, 1, 21)).tolist()
plt.imshow(rgb_image)  # plot the image for matplotlib
currentAxis = plt.gca()

detections = y.data
# scale each detection back up to the image
'''得到『寬、高、寬、高』的值，藉此將x1,y1,x2,y2回放為原本尺寸'''
scale = torch.Tensor(rgb_image.shape[1::-1]).repeat(2)
for i in range(detections.size(1)):
    j = 0
    '''信心程度>0.6的預測框才計算'''
    while detections[0,i,j,0] >= 0.6:
        score = detections[0,i,j,0]
        label_name = labels[i-1]
        display_txt = '%s: %.2f'%(label_name, score)
        pt = (detections[0,i,j,1:]*scale).cpu().numpy()
        coords = (pt[0], pt[1]), pt[2]-pt[0]+1, pt[3]-pt[1]+1
        color = colors[i]
        currentAxis.add_patch(plt.Rectangle(*coords, fill=False, edgecolor=color, linewidth=2))
        currentAxis.text(pt[0], pt[1], display_txt, bbox={'facecolor':color, 'alpha':0.5})
        j+=1


# ## Real Time

# In[15]:


from data import BaseTransform, VOC_CLASSES as labelmap

transform = BaseTransform(net.size, (104/256.0, 117/256.0, 123/256.0))

def detect(frame, net, transform):
    height, width = frame.shape[:2]
    frame_t = transform(frame)[0]
    x = torch.from_numpy(frame_t).permute(2, 0, 1)
    x = Variable(x.unsqueeze(0))
    y = net(x)
    detections = y.data
    scale = torch.Tensor([width, height, width, height])
    # detections = [batch, number of classes, number of occurence, (score, x0, Y0, x1, y1)]
    for i in range(detections.size(1)):
        j = 0
        while detections[0, i, j, 0] >= 0.6:
            pt = (detections[0, i, j, 1:] * scale).numpy()
            cv2.rectangle(frame, (int(pt[0]), int(pt[1])), (int(pt[2]), int(pt[3])), (255, 0, 0), 2)
            cv2.putText(frame, labelmap[i - 1], (int(pt[0]), int(pt[1])), cv2.FONT_HERSHEY_SIMPLEX, 2, (255, 255, 255), 2, cv2.LINE_AA)
            j += 1
    return frame
 


# In[ ]:


video_capture = cv2.VideoCapture(0)

while True:
    _, frame = video_capture.read()
    canvas = detect(frame, net, transform)
    cv2.imshow('Video', canvas)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
video_capture.release()
cv2.destroyAllWindows()


# ---

