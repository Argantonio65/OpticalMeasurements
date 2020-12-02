# -*- coding: utf-8 -*-
"""
Created on Fri Sep 11 16:26:58 2020

@author: morenoro
"""
import numpy as np
import cv2
import os
import json
from tqdm import tqdm

#%% Undistort Images

path = r'P:\11205202-fatracker\PretorialaanInstallation\FATrackerCNNtrainingDataset\dataset\Test1FATracker1'
pathoutput = r'P:\11205202-fatracker\PretorialaanInstallation\FATrackerCNNtrainingDataset\dataset\Test1FATracker1_und'


if not os.path.exists(pathoutput):
    os.makedirs(pathoutput)
    
# Load Calibration    
pathCalibrationParams = r'P:\11205202-fatracker\PretorialaanInstallation\CameraCalibration\CalibrationDataFATracker1.json'

with open(pathCalibrationParams) as f:
  calpar = json.load(f)

mtx = np.array(calpar['mtx'])
dist = np.array(calpar['dist'])

#%%
for imagename in tqdm(os.listdir(path)):
    
    img = cv2.imread(os.path.join(path, imagename))
    h,  w = img.shape[:2]
#    newcameramtx, roi=cv2.getOptimalNewCameraMatrix(mtx,dist,(w,h),1,(w,h))
    
    newcameramtx = cv2.fisheye.estimateNewCameraMatrixForUndistortRectify(np.eye(3), dist, (h,w), np.eye(3), balance=1)

    # undistort
    dst = cv2.undistort(img, mtx, dist, None, newcameramtx)
    # crop the image
    #x,y,w,h = roi
    #dst = dst[y:y+h, x:x+w]
    cv2.imwrite(os.path.join(pathoutput, imagename),dst)
    
