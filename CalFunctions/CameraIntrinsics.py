import numpy as np
import cv2
import os
import json

# termination criteria for the optimization routine
criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)

gridnumx = 17  # get the size and number of squares in your pattern (in this case 6x7)
gridnumy = 21

# prepare object points, like (0,0,0), (1,0,0), (2,0,0) ....,(6,5,0)
objp = np.zeros((gridnumx*gridnumy,3), np.float32) 
objp[:,:2] = np.mgrid[0:gridnumy,0:gridnumx].T.reshape(-1,2)

# Arrays to store object points and image points from all the images.
objpoints = [] # 3d point in real world space
imgpoints = [] # 2d points in image plane.

pathimages = r'P:\11205202-fatracker\PretorialaanInstallation\CameraCalibration\FAtracker2_CalibrationImages'
images = [s for s in os.listdir(pathimages) if '.png' in s] # get al jpgs in your folder

if not os.path.exists(os.path.join(pathimages,'OutCalibration')):
    os.makedirs(os.path.join(pathimages,'OutCalibration'))
    
#%%
############ CAM CALIBRATION
for fname in images:
    
    img = cv2.imread(os.path.join(pathimages,fname))
    gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)

    # Find the chess board corners
    # ret, corners = cv2.findChessboardCorners(gray,(gridnumy,gridnumx),None)
    ret, corners = cv2.findChessboardCornersSB(gray,(gridnumy,gridnumx),None)

    print(ret)
    # If found, add object points, image points (after refining them)
    
    if ret == True:
        objpoints.append(objp)

        corners2 = cv2.cornerSubPix(gray,corners,(11,11),(-1,-1),criteria)
        imgpoints.append(corners2)  # store the detected corners

        # Draw and display the corners
        img = cv2.drawChessboardCorners(img, (gridnumy,gridnumx), corners2,ret)
        
        font = cv2.FONT_HERSHEY_SIMPLEX
        cv2.putText(img, fname, (20,20), font, 0.5, (0, 255, 0), 1, cv2.LINE_AA)
        cv2.imshow('img',img)
        cv2.waitKey(10)
        
        cv2.imwrite(os.path.join(pathimages,'OutCalibration/' + fname.split('.')[0] + '_patternDetected.jpg'), img)
        

cv2.destroyAllWindows()

# calibrate
ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(objpoints, imgpoints, gray.shape[::-1],None,None) # provides distortion param (dist) and intrinsic (mtx) / extrinsic (rvec, tvec) matrices

# reprojection error
T_error = 0
for i in range(len(objpoints)):
    imgpoints2, _ = cv2.projectPoints(objpoints[i], rvecs[i], tvecs[i], mtx, dist)
    error = cv2.norm(imgpoints[i],imgpoints2, cv2.NORM_L2)/len(imgpoints2)
    T_error += error

print("Mean error: ", T_error/len(objpoints))

###############
#%% SAVE calibration data:

CalibrationData = {'mtx':mtx.tolist(), 'dist': dist.tolist()}
with open('CalibrationDataFATracker2.json', 'w') as fp:
    json.dump(CalibrationData, fp)

#%%
########### Camera rectification for a new image

#img = cv2.imread('left12.jpg') 
#h,  w = img.shape[:2]
#newcameramtx, roi=cv2.getOptimalNewCameraMatrix(mtx,dist,(w,h),1,(w,h))
## undistort
#dst = cv2.undistort(img, mtx, dist, None, newcameramtx)
#
## crop the image
#x,y,w,h = roi
#dst = dst[y:y+h, x:x+w]
#cv2.imwrite('calibresult.png',dst)
