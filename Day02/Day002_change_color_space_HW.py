# -*- coding: utf-8 -*-
"""
Created on Wed May 27 20:57:31 2020

@author: mingg
"""

import sys
import os
dirc = r'D:\MJ_Python_codes\DL-CVMarathon'
sys.path.append(dirc)
os.chdir(dirc)

import cv2

import matplotlib.pyplot as plt
import numpy as np



'''
'''
img = cv2.imread('lena.png')


img_HSV = cv2.cvtColor( img ,cv2.COLOR_BGR2HSV )
img_HSL = cv2.cvtColor( img ,cv2.COLOR_BGR2HLS )
img_LAB = cv2.cvtColor( img ,cv2.COLOR_BGR2LAB )

img_concat = np.hstack( (img, img_HSV, img_HSL, img_LAB) )

cv2.imshow('bgr_split', img_concat)
cv2.waitKey(0)
cv2.destroyAllWindows

'''
c.f. through plt
'''
plt.imshow(img)