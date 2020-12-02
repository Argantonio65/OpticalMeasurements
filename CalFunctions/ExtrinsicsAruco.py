import numpy as np
import matplotlib.pyplot as plt
import cv2
import os
from tqdm import tqdm
import json
import cv2.aruco as aruco
import imutils
import pandas as pd
import math
   

# pre-computed Aruco locations
thresh_aruco = 50 # max distance allowed

#### GET world coordinates from the markers, create Extrinsic estimation
WorldCoords = pd.read_excel('Aruco_ModelCoordinates.xlsx', index_col=[0])
#### GET ESTIMATED ARUCO COORDINATES IN CAMERA FRAME (pre-computed) # gives further robustness to the extrinsic computation
arucoPreset = pd.read_csv('Aruco_Estimated_position_Camframe_{}.csv'.format(camera), index_col = [0])


### Preloading aruco library
aruco_dict = aruco.Dictionary_get(aruco.DICT_4X4_1000) # loading ARUCO fiduciary marker dictionary (4x4 code)
parameters_aruco =  aruco.DetectorParameters_create()

# Compute the current Extrinsics 
frame_gray = cv2.cvtColor(frame_ud, cv2.COLOR_BGR2GRAY)  # convert image to grayscale
corners, ids, rejectedImgPoints = aruco.detectMarkers(frame_gray, aruco_dict, parameters=parameters_aruco) # detect aruco markers

n = 10
corners = np.array([corners[i] for i, s in enumerate(ids) if s in np.arange(n)])  # get only n first arucos
ids = np.array([[s[0]] for s in ids if s in np.arange(n)])

### Refine aruco detection (drop erroneous detection)
ids_f = []
corners_f = []
for i, id_i in enumerate(ids):
    pts = corners[i].astype(int).reshape((-1,1,2))
    dist = np.linalg.norm(arucoPreset.loc[id_i].values - pts.mean(axis = 0))
    if dist < thresh_aruco:
        cv2.polylines(frame_display,[pts],True,(0,255,0), thickness = 2)
        ids_f.append(id_i)
        corners_f.append(corners[i])

WorldCoordinates, FrameCoordinates = Get_Extrinsic_frame_to_world_points(ids_f, corners_f, WorldCoords)
    
dist_params_iPNP = np.array([0.,0.,0.,0.,0.]) #dist_params_dummy

# Solve PNP, and retrieve rotation and translation vectors from known correspondence points
retval, rvec_ran, tvec_ran = cv2.solvePnP(WorldCoordinates, FrameCoordinates, IntrinsicM_i, dist_params_iPNP)#, useExtrinsicGuess=True, rvec = rvec_ran_p, tvec = tvec_ran_p, flags=cv2.SOLVEPNP_ITERATIVE)

rvec_ran_p = rvec_ran 
tvec_ran_p = tvec_ran

## Validate projection
P2, _ = cv2.projectPoints(np.array(WorldCoordinates).astype(float) , rvec_ran, tvec_ran, IntrinsicM_i, dist_params_iPNP)

for p in P2:
    cv2.circle(frame_display,tuple(p.astype(int)[0]), 4, (255,190,0), -1)