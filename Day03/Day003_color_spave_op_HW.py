# -*- coding: utf-8 -*-
"""
Created on Wed May 27 21:37:47 2020

@author: mingg
"""

import sys
import os
dirc = r'C:\Users\mingg\Documents\GitHub\DL-CVMarathon'
sys.path.append(dirc)
os.chdir(dirc)

import cv2
import copy
import matplotlib.pyplot as plt
import numpy as np


img = cv2.imread('lena.png')


'''
HW1 COLOR SPACE
'''
#(1)change color space (BGR-->HSV)
img_hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
change_percentage = 0.2
s = 1

#(2)change dtype for tuning
img_hsv_down = img_hsv.astype(np.float64)
img_hsv_up = img_hsv.astype(np.float64)

#(3)tuning channel s & check bound
img_hsv_down[:,:,s] = np.clip(img_hsv[:,:,s]/255 - change_percentage, 0, 1)
img_hsv_up[:,:,s] = np.clip(img_hsv[:,:,s]/255 + change_percentage, 0, 1)

#reverse(3) 
img_hsv_down[:,:,s] = 255*img_hsv_down[:,:,s]
img_hsv_up[:,:,s] = 255*img_hsv_up[:,:,s]

#reverse(2)
img_hsv_down = img_hsv_down.astype(np.uint8)
img_hsv_up = img_hsv_up.astype(np.uint8)

#reverse(1)
img_hsv_down = cv2.cvtColor(img_hsv_down,cv2.COLOR_HSV2BGR)
img_hsv_up = cv2.cvtColor(img_hsv_up,cv2.COLOR_HSV2BGR)

img_concat = np.hstack( (img, img_hsv_down, img_hsv_up) )
cv2.imshow('change_saturation', img_concat)
cv2.waitKey(0)
cv2.destroyAllWindows

'''
HW2 HISTOGRAM EQUALIZATION
'''
#case 1: 把彩圖拆開對每個 channel 個別做直方圖均衡再組合起來
#case 2: 轉換 color space 到 HSV 之後對其中一個 channel 做直方圖均衡
img = cv2.imread('lena.png')

# case 1
# 每個 channel 個別做直方圖均衡

# 組合經過直方圖均衡的每個 channel
img_bgr_equal = np.zeros( (512,512,3) ).astype(np.uint8)
img_bgr_equal[:,:,0] =  cv2.equalizeHist(img[:,:,0])
img_bgr_equal[:,:,1] =  cv2.equalizeHist(img[:,:,1])
img_bgr_equal[:,:,2] =  cv2.equalizeHist(img[:,:,2])

# case 2 - 轉換 color space 後只對其中一個 channel 做直方圖均衡
img_hsv_equal = np.zeros( (512,512,3) ).astype(np.uint8)
img_hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
img_hsv_equal[:,:,0] = img_hsv[:,:,0]
img_hsv_equal[:,:,1] = cv2.equalizeHist(img_hsv[:,:,1])
img_hsv_equal[:,:,2] = img_hsv[:,:,2]
img_hsv_equal = cv2.cvtColor(img_hsv_equal, cv2.COLOR_HSV2BGR)

# 組合圖片 + 顯示圖片
img_bgr_equalHist = np.hstack((img, img_bgr_equal, img_hsv_equal))

# 比較 (原圖, BGR color space 對每個 channel 做直方圖均衡, HSV color space 對明度做直方圖均衡)
cv2.imshow('bgr equal histogram', img_bgr_equalHist )
cv2.waitKey(0)
cv2.destroyAllWindows



'''
HW3 CONTRAST
'''

imgA = np.zeros((512,512,3))
imgB = np.zeros((512,512,3))


for b in range(img.shape[0]):
    for g in range(img.shape[1]):
        for r in range(img.shape[2]):
            imgA[b,g,r] = np.clip(2*img[b,g,r],0,255)
            imgB[b,g,r] = np.clip(1*img[b,g,r]+50,0,255)

imgA = imgA.astype(np.uint8)
imgB = imgB.astype(np.uint8)
            
img_concat = np.hstack( (img, imgA, imgB) )
cv2.imshow('bgr_split', img_concat)
cv2.waitKey(0)
cv2.destroyAllWindows