# -*- coding: utf-8 -*-
"""
Created on Fri May 29 19:44:54 2020

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
C
ase1
'''


# 取得旋轉矩陣
# getRotationMatrix2D(center, angle, scale)
M_rotate = cv2.getRotationMatrix2D((cols//2, rows//2), 45, 0.5)
print('Rotation Matrix')
print(M_rotate)
print()

# 取得平移矩陣
M_translate = np.array([[1, 0, 100], [0, 1, -50]], dtype=np.float32)
print('Translation Matrix')
print(M_translate)

# 旋轉
img_rotate = cv2.warpAffine(img, M_rotate, (cols, rows))

# 平移
img_rotate_trans = cv2.warpAffine(img_rotate, M_translate, (cols, rows))

# 組合 + 顯示圖片
img_show_rotate_trans = np.hstack((img, img_rotate, img_rotate_trans))
cv2.imshow('image', img_show_rotate_trans)
cv2.waitKey(0)

'''
Case2
'''
# 給定兩兩一對，共三對的點
# 這邊我們先用手動設定三對點，一般情況下會有點的資料或是透過介面手動標記三個點
rows, cols = img.shape[:2]
pt1 = np.array([[50,50], [300,100], [200,300]], dtype=np.float32)
pt2 = np.array([[80,80], [330,150], [300,300]], dtype=np.float32)

A = np.array([[50,50,1,0,0,0],
             [0,0,0,50,50,1],
             [300,100,1,0,0,0],
             [0,0,0,300,100,1],
             [200,300,1,0,0,0],
             [0,0,0,200,300,1]])

B = np.array([80,80,330,150,300,300]).reshape(6,1)

A_inv = np.linalg.inv(A)
#X = A_inv.dot(B)
X = np.dot(A_inv,B)

# 取得 affine 矩陣並做 affine 操作
M_affine = np.array([ [X[0][0], X[1][0], X[2][0]], [X[3][0], X[4][0], X[5][0]] ], dtype=np.float32)
#M_affine = cv2.getAffineTransform(pt1,pt2)
img_affine = cv2.warpAffine(img, M_affine, (cols, rows))

# 在圖片上標記點
img_copy = img.copy()
for idx, pts in enumerate(pt1):
    pts = tuple(map(int, pts))
    cv2.circle(img_copy, pts, 3, (0, 255, 0), -1)
    cv2.putText(img_copy, str(idx), (pts[0]+5, pts[1]+5), cv2.FONT_HERSHEY_COMPLEX, 1, (0, 255, 0), 1)

for idx, pts in enumerate(pt2):
    pts = tuple(map(int, pts))
    cv2.circle(img_affine, pts, 3, (0, 255, 0), -1)
    cv2.putText(img_affine, str(idx), (pts[0]+5, pts[1]+5), cv2.FONT_HERSHEY_COMPLEX, 1, (0, 255, 0), 1)

# 組合 + 顯示圖片
img_show_affine = np.hstack((img_copy, img_affine))
cv2.imshow('image', img_show_affine)
cv2.waitKey(0)