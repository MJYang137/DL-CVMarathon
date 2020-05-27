# -*- coding: utf-8 -*-
"""
Created on Wed May 27 08:49:38 2020

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
import pandas as pd
import seaborn as sns

img = cv2.imread('lena.png')
cv2.imshow('rgb',img)
cv2.waitKey(0)
cv2.destroyAllWindows

img = cv2.imread('lena.png',cv2.IMREAD_GRAYSCALE)
cv2.imshow('gray',img)
cv2.waitKey(0)
cv2.destroyAllWindows