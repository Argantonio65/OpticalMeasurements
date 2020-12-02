# -*- coding: utf-8 -*-
"""
Created on Sun Sep 13 12:26:55 2020

@author: morenoro
"""

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

path = r'P:\11205202-fatracker\PretorialaanInstallation\FATrackerCNNtraining\dataset\Test1FATracker2'
pathoutput = r'P:\11205202-fatracker\PretorialaanInstallation\FATrackerCNNtraining\dataset\Test1FATracker1_und2'


if not os.path.exists(pathoutput):
    os.makedirs(pathoutput)
    
# Load Calibration    
pathCalibrationParams = r'P:\11205202-fatracker\PretorialaanInstallation\CameraCalibration\CalibrationDataFATracker1_Fisheye.json'

with open(pathCalibrationParams) as f:
  calpar = json.load(f)

D = np.array(calpar['D'])
K = np.array(calpar['K'])
DIM = np.array(calpar['dim'])

#%%
for imagename in tqdm(os.listdir(path)):
    
    balance = 1 # 0-1, 0 crop max, 1 no crop
    
    img = cv2.imread(os.path.join(path, imagename))
    h,  w = img.shape[:2]
    dim1 = img.shape[:2][::-1]
    
    new_K = cv2.fisheye.estimateNewCameraMatrixForUndistortRectify(K, D, dim1, np.eye(3), balance=balance)    
    map1, map2 = cv2.fisheye.initUndistortRectifyMap(K, D, np.eye(3), new_K, dim1, cv2.CV_16SC2) # if images in calibration and undistortion have different sizes, K and dim must be scaled.
    
    undistorted_img = cv2.remap(img, map1, map2, interpolation=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT)

    cv2.imshow("undistorted", undistorted_img)
    cv2.waitKey(1000)
    cv2.destroyAllWindows()
    
    cv2.imwrite(os.path.join(pathoutput, imagename),undistorted_img)
    
