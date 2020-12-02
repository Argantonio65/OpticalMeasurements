# -*- coding: utf-8 -*-
"""
Created on Tue Sep 10 16:32:11 2019

@author: morenoro
"""

#!/usr/bin/env python
# coding: utf-8

# In[1]:

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

# #### PROCESSING
# pre-computed Aruco locations
thresh_aruco = 50 # max distance allowed
BackgroundComputationFrames = 15


# In[3] Operating Functions:

def Get_Distort_Intrinsic(cam_init):
    """
Parses the distortion and intrinsicMatrix parameters from the initialization file.
    Opencv distortion parameters: [radial1, radial2, tang1, tang2, radial3]
    Opencv Intrinsic matrix: np.array([fx, 0, cx],
                                      [0, fy, cy],
                                      [0,  0,  1])
    """
    parms = cam_init['ParameterDistortion']
    dist_params = np.array([parms['p1'], parms['p2'], parms['t1'], parms['t2'], parms['p3']])
    
    IntrinsicM = np.array([[cam_init['IntrinsicMatrix']['fx'], 0, cam_init['IntrinsicMatrix']['cx']], 
                          [0, cam_init['IntrinsicMatrix']['fy'], cam_init['IntrinsicMatrix']['cy']], 
                          [0,0,1]])
    
    return dist_params, IntrinsicM


def Get_Extrinsic_frame_to_world_points(ids, corners, WorldCoords):
    """
    Gets the corresponding points in the world and frame CRS for the markers coordinates:[centre, top_left, top_right, bottom_right, bottom_left]
    """
    WorldCoordinates = []
    FrameCoordinates = []

    for i, id_i in enumerate(ids):
        WorldCoordinates.append(WorldCoords.loc[id_i[0]].values.reshape(5,3))
        FrameCoordinates.append(np.vstack([corners[i].mean(axis = 1), corners[i][0]]))
        
    return np.array(WorldCoordinates).reshape(5*len(ids),3,1).astype(float), np.array(FrameCoordinates).reshape(5*len(ids),2).astype(float)


def line_plane_intersect(p_line, v_line, p_plane, n_plane):
    """
    Compute intersection betwen line and plane given point and vector direction.
    """
    n_d_u = n_plane.dot(v_line)
    if abs(n_d_u) < 1e-6: # vector is parallel wrt to the plane
        return []
    else:
        w = p_line - p_plane
        si = - n_plane.dot(w) / n_d_u
        return np.array([w + si * v_line + p_plane])
    
def refracted_ray(i, n_sur, n1, n2):
    """
    Compute the refracted vector ray given an incident vector i, a surface normal, n_sur and the refraction indices of each media
    """
    i = i/(i**2).sum()**0.5
    n_sur = n_sur/(n_sur**2).sum()**0.5
    
    r = n1/n2
    c = -n_sur.dot(i)
    return r*i + (r*c - np.sqrt(1 - r**2*(1 - c**2)))*n_sur

def ray_retracing(particle_loc, IntrinsicM, cameraPosition, Pose_matrix_camtoworld, p_plane, n_plane, n1, n2):
    """
    Compute the visual ray propagation, intersetion with water plane and refraction vector
    particle_loc : [x_particle_pixels, y_particle_pixels]
    Assumption:  Water surface is described with a plane
    
    v_ray_w : particle visual ray from camera centre in world CRS
    P_watsurf : Projection of particle in the water plane 
    r_i : computed refracted vector
    """
    
    hom_p = np.array([particle_loc[0], particle_loc[1], 1]) # normalized particle coordinates in frame CRS, [xp,yp,1]
    v_ray_f = np.hstack([np.linalg.inv(IntrinsicM).dot(hom_p), 1]) # multiply the homogeneous by the inverse of the intrisic matrix
    v_ray_w = np.linalg.inv(Pose_matrix_camtoworld).dot(v_ray_f)[:-1] - np.array(cameraPosition).reshape(3)  # get visual ray vector in world coordinates

    P_watsurf_i = line_plane_intersect(np.array(cameraPosition).reshape(3), v_ray_w, p_plane, n_plane) # intersect waterplane
    r_i = refracted_ray(v_ray_w , n_plane, n1, n2) # compute the refracted ray

    return v_ray_w, P_watsurf_i, r_i


def detectColor_contour(image_HSV, c, rangescolor):
    color = 'none'
    ((x, y), radius) = cv2.minEnclosingCircle(contour)

    #mean = image_HSV[int(y),int(x),:]
    kern = 3
    mean = image_HSV[int(y)-kern:int(y)+kern,int(x)-kern:int(x)+kern,0:3]
    mean = mean.mean(axis = 0).mean(axis = 0)

    if (20 < mean[2] < 240) & (20 < mean[1] < 240): # check that saturation and value are above 30
        if rangescolor[2] <= mean[0] <= rangescolor[3]:
            color = 'blue_par'
        elif rangescolor[0] <= mean[0] <= rangescolor[1]:
            color = 'green_par'    
        elif rangescolor[4] <= mean[0] <= rangescolor[5]:
            color = 'orange_par'   

    return color, mean



def detectColor_contour2(image_HSV, c, rangescolor):
    color = 'none'
    ((x, y), radius) = cv2.minEnclosingCircle(contour)


    mean = image_HSV[int(y),int(x),:]
    #kern = 3
    #mean = image_HSV[int(y)-kern:int(y)+kern,int(x)-kern:int(x)+kern,0:3]
    #mean = mean.mean(axis = 0).mean(axis = 0)

    if (20 < mean[2] < 240) & (20 < mean[1] < 240): # check that saturation and value are above 30
        if rangescolor[2] <= mean[0] <= rangescolor[3]:
            color = 'blue_par'
        elif rangescolor[0] <= mean[0] <= rangescolor[1]:
            color = 'green_par'    
        elif rangescolor[4] <= mean[0] <= rangescolor[5]:
            color = 'orange_par'   

    return color, mean