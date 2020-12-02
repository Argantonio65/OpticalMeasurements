# -*- coding: utf-8 -*-
"""
Created on Sun Sep 13 12:21:12 2020

@author: morenoro
"""

import numpy as np
import cv2
import os
import json


pathimages = r'P:\11205202-fatracker\PretorialaanInstallation\CameraCalibration\FAtracker2_CalibrationImages'

if not os.path.exists(os.path.join(pathimages,'OutCalibration')):
    os.makedirs(os.path.join(pathimages,'OutCalibration'))
    
#%% Prepare

CHECKERBOARD = (17,21)

subpix_criteria = (cv2.TERM_CRITERIA_EPS+cv2.TERM_CRITERIA_MAX_ITER, 30, 0.1)
calibration_flags = cv2.fisheye.CALIB_RECOMPUTE_EXTRINSIC+cv2.fisheye.CALIB_FIX_SKEW+cv2.fisheye.CALIB_CHECK_COND

objp = np.zeros((1, CHECKERBOARD[0]*CHECKERBOARD[1], 3), np.float32)
objp[0,:,:2] = np.mgrid[0:CHECKERBOARD[0], 0:CHECKERBOARD[1]].T.reshape(-1, 2)
_img_shape = None

objpoints = [] # 3d point in real world space
imgpoints = [] # 2d points in image plane.

############ CAM CALIBRATION

images = [s for s in os.listdir(pathimages) if '.png' in s] # get al jpgs in your folder

for fname in images:
    
    img = cv2.imread(os.path.join(pathimages,fname))
    gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
    
    _img_shape = img.shape[:2]
    # Find the chess board corners
    # ret, corners = cv2.findChessboardCorners(gray,(gridnumy,gridnumx),None)
    ret, corners = cv2.findChessboardCornersSB(gray,CHECKERBOARD, cv2.CALIB_CB_ADAPTIVE_THRESH+cv2.CALIB_CB_FAST_CHECK+cv2.CALIB_CB_NORMALIZE_IMAGE)

    print(ret)
    # If found, add object points, image points (after refining them)
    
    if ret == True:
        objpoints.append(objp)

        cv2.cornerSubPix(gray,corners,(3,3),(-1,-1),subpix_criteria)
        imgpoints.append(corners)  # store the detected corners

        # Draw and display the corners
        img = cv2.drawChessboardCorners(img, CHECKERBOARD, corners,ret)
        
        font = cv2.FONT_HERSHEY_SIMPLEX
        cv2.putText(img, fname, (20,20), font, 0.5, (0, 255, 0), 1, cv2.LINE_AA)
        cv2.imshow('img',img)
        cv2.waitKey(10)
        
        cv2.imwrite(os.path.join(pathimages,'OutCalibration/' + fname.split('.')[0] + '_patternDetected.jpg'), img)
        
        cv2.destroyAllWindows()


cv2.destroyAllWindows()

N_OK = len(objpoints)
K = np.zeros((3, 3))
D = np.zeros((4, 1))
rvecs = [np.zeros((1, 1, 3), dtype=np.float64) for i in range(N_OK)]
tvecs = [np.zeros((1, 1, 3), dtype=np.float64) for i in range(N_OK)]
rms, _, _, _, _ = cv2.fisheye.calibrate(objpoints,
                                        imgpoints,
                                        gray.shape[::-1],
                                        K,
                                        D,
                                        rvecs,
                                        tvecs,
                                        calibration_flags,
                                        (cv2.TERM_CRITERIA_EPS+cv2.TERM_CRITERIA_MAX_ITER, 30, 1e-6))

#%%
print("Found " + str(N_OK) + " valid images for calibration")
print("K=np.array(" + str(K.tolist()) + ")")
print("D=np.array(" + str(D.tolist()) + ")")

###############
#%% SAVE calibration data:

CalibrationData = {'K':K.tolist(), 'D': D.tolist(), 'dim' : _img_shape}
with open('CalibrationDataFATracker2_Fisheye.json', 'w') as fp:
    json.dump(CalibrationData, fp)

#%% Example undistort
    
pathim = r'P:\11205202-fatracker\Datasets_raw\Pretorialaan\FATracker2_data\2020-07-28\FATracker2_Rot_ImageT_2020-07-28_20%3A51%3A45.775.jpg'

balance = 1 # 0-1, 0 crop max, 1 no crop

img = cv2.imread(pathim)
h,  w = img.shape[:2]
dim1 = img.shape[:2][::-1]

new_K = cv2.fisheye.estimateNewCameraMatrixForUndistortRectify(K, D, dim1, np.eye(3), balance=balance)    
map1, map2 = cv2.fisheye.initUndistortRectifyMap(K, D, np.eye(3), new_K, dim1, cv2.CV_16SC2) # if images in calibration and undistortion have different sizes, K and dim must be scaled.

undistorted_img = cv2.remap(img, map1, map2, interpolation=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT)

cv2.imshow("undistorted", undistorted_img)
cv2.waitKey(0)
cv2.destroyAllWindows()


    
    
    
    
    
    


