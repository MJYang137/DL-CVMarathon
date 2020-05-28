# -*- coding: utf-8 -*-
"""
Created on Thu May 28 20:36:52 2020

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

'''
上下左右翻轉圖片
'''
img = cv2.imread('lena.png')



# 水平翻轉 (horizontal)
img_hflip = img[:, ::-1, :]

# 垂直翻轉 (vertical)
img_vflip = img[::-1, :, :]

# 水平 + 垂直翻轉
img_hvflip = img[::-1,::-1,:]

# 組合 + 顯示圖片
hflip = np.hstack((img, img_hflip))
vflip = np.hstack((img_vflip, img_hvflip))
img_flip = np.vstack((hflip, vflip))
# 組合 + 顯示圖片
cv2.imshow('flip', img_flip)
cv2.waitKey(0)
cv2.destroyAllWindows

'''
'''
import time


# 將圖片縮小成原本的 20%
img_test = cv2.resize(img, None, fx=0.2, fy=0.2)

# 將圖片放大為"小圖片"的 8 倍大 = 原圖的 1.6 倍大
fx, fy = 8, 8

# 組合 + 顯示圖片
start_time = time.time()
orig_img = cv2.resize(img, img_area_scale.shape[:2])
print('INTER_LINEAR(default) zoom cost {}'.format(time.time() - start_time))


# 鄰近差值 scale + 計算花費時間
start_time = time.time()
img_area_scale = cv2.resize(img_test, None, fx=fx, fy=fy, interpolation=cv2.INTER_NEAREST)
print('INTER_NEAREST zoom cost {}'.format(time.time() - start_time))

# 組合 + 顯示圖片
start_time = time.time()
orig_img_cubic = cv2.resize(img, img_area_scale.shape[:2], interpolation=cv2.INTER_CUBIC)
print('INTER_CUBIC zoom cost {}'.format(time.time() - start_time))

# 組合 + 顯示圖片
start_time = time.time()
orig_img_area = cv2.resize(img, img_area_scale.shape[:2], interpolation=cv2.INTER_AREA)
print('INTER_AREA zoom cost {}'.format(time.time() - start_time))

img_zoom = np.hstack((orig_img, orig_img_cubic, orig_img_area, img_area_scale))
cv2.imshow('scale', img_zoom)
cv2.waitKey(0)
cv2.destroyAllWindows


'''
'''
# 設定 translation transformation matrix
# x 平移 100 pixel; y 平移 50 pixel
tx = 50; ty = 100
M = np.array([[1, 0, tx],[0, 1, ty]], dtype=np.float32)
shift_img = cv2.warpAffine(img, M, (img.shape[1], img.shape[0]))

# 組合 + 顯示圖片
img_shift = np.hstack((img, shift_img))
cv2.imshow('scale', img_shift)
cv2.waitKey(0)
cv2.destroyAllWindows