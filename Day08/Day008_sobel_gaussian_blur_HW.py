# -*- coding: utf-8 -*-
"""
Created on Sun May 31 11:04:04 2020

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

rows, cols = img.shape[:2]

'''
邊緣檢測
比較 Sobel 如果在 uint8 的情況下做會 overflow 的狀況
'''


# 轉為灰階圖片
img_grey = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

# 對 x 方向以包含負數的資料格式 (cv2.CV_16S) 進行 Sobel 邊緣檢測
img_sobel_x = cv2.Sobel(img_grey, cv2.CV_16S, dx=1, dy=0, ksize=3)


# 對 x 方向依照比例縮放到所有數值都是非負整數
img_sobel_x = cv2.convertScaleAbs(img_sobel_x)

# 對 x 方向直接以非負整數的資料格式 (uint8) 進行 Sobel 邊緣檢測
img_sobel_x_uint8 = cv2.Sobel(img_grey, cv2.CV_8U, dx=1, dy=0, ksize=3)

#　組合 + 顯示圖片
img_show = np.hstack((img_grey, img_sobel_x, img_sobel_x_uint8))
cv2.imshow('Edge Detection', img_show)
cv2.waitKey(0)
'''
比較一次與兩次計算偏微分的結果
'''

img_grey = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

# 求一次導數取得邊緣檢測結果
img_sobel_x = cv2.Sobel(img_grey, cv2.CV_16S, dx=1, dy=0, ksize=3)
img_sobel_x = cv2.convertScaleAbs(img_sobel_x)
# 求二次導數取得邊緣檢測結果
img_sobel_xx = cv2.Sobel(img_grey, cv2.CV_16S, dx=2, dy=0, ksize=3)
img_sobel_xx = cv2.convertScaleAbs(img_sobel_xx)
#　組合 + 顯示圖片
img_show = np.hstack((img_grey, img_sobel_x, img_sobel_xx))
cv2.imshow('Edge Detection', img_show)
cv2.waitKey(0)