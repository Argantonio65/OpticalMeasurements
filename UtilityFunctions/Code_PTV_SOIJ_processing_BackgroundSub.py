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
# In[2]: DEFINE SETTINGS:
    
camera = 'CAM_1'
experiment = 'SOIJ_290'
pathExp = r'P:\11203767-012-3dptv-soij\Data\\' + experiment + '\\' + camera

# pre-computed Aruco locations
thresh_aruco = 50 # max distance allowed
BackgroundComputationFrames = 15

### GET WATER LEVEL
Logbook = pd.read_excel(r'P:\11203285-003-schaalmodel-so-ij\ScaleModel\01-Testing\Logbook\Logbook_SOIJ.xlsx')
WL = pd.read_csv(r'P:\11203285-003-schaalmodel-so-ij\ScaleModel\01-Testing\Postprocessing\Output\OutputData\\' +  str(Logbook[Logbook.Name == experiment]['Test ID'].values[0]) + r'\\wl.txt', delimiter=r"\s+")
WL = WL['wlDnS']/1000. + 0.85


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
    kern = 2
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


def detectColor_contour_Point(image_HSV, c, rangescolor):
    '''
    Consider only the central color point. Lets better detection of orange particles.
    '''
    color = 'none'
    ((x, y), radius) = cv2.minEnclosingCircle(contour)

    mean = image_HSV[int(y),int(x),:]

    if (20 < mean[2] < 240) & (20 < mean[1] < 240): # check that saturation and value are above 30
        if rangescolor[2] <= mean[0] <= rangescolor[3]:
            color = 'blue_par'
        elif rangescolor[0] <= mean[0] <= rangescolor[1]:
            color = 'green_par'    
        elif rangescolor[4] <= mean[0] <= rangescolor[5]:
            color = 'orange_par'   

    return color, mean

# #### LOADING INITIALIZATION
#%% Automated initialization
cam_1_Init = json.load(open("Cam1_InitializationFile.txt"))
cam_2_Init = json.load(open("Cam2_InitializationFile.txt"))
cam_3_Init = json.load(open("Cam3_InitializationFile.txt"))

aruco_dict = aruco.Dictionary_get(aruco.DICT_4X4_1000) # loading ARUCO fiduciary marker dictionary (4x4 code)
parameters_aruco =  aruco.DetectorParameters_create()

# Parsing Distortion parameters and Intrinsic Matrix from the initialization file (see Get_Distort_Intrinsic())
cam_1_dist_params, cam1_IntrinsicM = Get_Distort_Intrinsic(cam_1_Init)
cam_2_dist_params, cam2_IntrinsicM = Get_Distort_Intrinsic(cam_2_Init)
cam_3_dist_params, cam3_IntrinsicM = Get_Distort_Intrinsic(cam_3_Init)


#### GET world coordinates from the markers, create Extrinsic estimation
WorldCoords = pd.read_excel('Aruco_ModelCoordinates.xlsx', index_col=[0])

#### GET ESTIMATED ARUCO COORDINATES IN CAMERA FRAME (pre-computed) # gives further robustness to the extrinsic computation
arucoPreset = pd.read_csv('Aruco_Estimated_position_Camframe_{}.csv'.format(camera), index_col = [0])

### 1 IMAGE loading
#initialise
if camera == 'CAM_1':
    IntrinsicM_i = cam1_IntrinsicM
    dist_params_i = cam_1_dist_params
    Init_i = cam_1_Init
elif camera == 'CAM_2':
    IntrinsicM_i = cam2_IntrinsicM
    dist_params_i = cam_2_dist_params
    Init_i = cam_2_Init
elif camera == 'CAM_3':
    IntrinsicM_i = cam3_IntrinsicM
    dist_params_i = cam_3_dist_params
    Init_i = cam_3_Init

frame_i = cv2.imread(os.path.join(pathExp, os.listdir(pathExp)[0]))
height,width,depth = frame_i.shape

out = cv2.VideoWriter('Outputs/Experiment_{}_{}.avi'.format(experiment, camera),cv2.VideoWriter_fourcc(*'DIVX'), 4,(width,height))
CropSize = Init_i['ROI']

if CropSize is not 'None':  # Check if all ROI mask is supplied (black-white .bmp mask)
    mask_crop = cv2.imread(CropSize, cv2.IMREAD_GRAYSCALE)

#processing
timelist = np.sort([int(s.split('_')[-1][:-4]) for s in os.listdir(pathExp)]) # order timefiles

# prepare ouput scheme
OutputFile_1 = pd.DataFrame(index = timelist, columns = ['P_b', 'R_b', 'P_g', 'R_g', 'P_o', 'R_o', 
                                                         'rvec', 'tvec', 'cameraPosition', 'Arucos_id', 'ArucoCorners'])

# Define the color range for the filter in the H chanel of HSV
rangescolor = [Init_i['HSV_green']['low'][0],
               Init_i['HSV_green']['high'][0],
               Init_i['HSV_blue']['low'][0],
               Init_i['HSV_blue']['high'][0],
               Init_i['HSV_orange']['low'][0],
               Init_i['HSV_orange']['high'][0]]

# In[8]:
## PRECOMPUTE BACKGROUND
Background = []
for i_t , framename in enumerate(tqdm([os.listdir(pathExp)[0][:BackgroundComputationFrames] + '{}.bmp'.format(s) for s in timelist][0:BackgroundComputationFrames], desc = 'Processing Background Image')):
    frame_i = cv2.imread(os.path.join(pathExp, framename))
    frame_i = cv2.undistort(frame_i, IntrinsicM_i, dist_params_i)
    image_g = cv2.cvtColor(frame_i, cv2.COLOR_BGR2GRAY)
    Background.append(image_g)

Background = np.mean(Background, axis = 0)

# In[9]: Processing
#for i_t , framename in enumerate(tqdm([os.listdir(pathExp)[0][:BackgroundComputationFrames] + '{}.bmp'.format(s) for s in timelist][BackgroundComputationFrames:], desc = 'Particle dectection processing')):
for i_t , framename in enumerate(tqdm([os.listdir(pathExp)[0][:BackgroundComputationFrames] + '{}.bmp'.format(s) for s in timelist][100:], desc = 'Particle dectection processing')):

    time = int(framename.split('_')[-1][:-4])
    
    frame_i = cv2.imread(os.path.join(pathExp, framename))

    ### 2 Undistorting Image
    frame_ud = cv2.undistort(frame_i, IntrinsicM_i, dist_params_i) #frame_i.copy() # 
    frame_display = frame_ud.copy() # create a copy of the undistorted image to display elements
    
    ### 3 Compute the current Extrinsics 
    frame_gray = cv2.cvtColor(frame_ud, cv2.COLOR_BGR2GRAY)  # convert image to grayscale
    corners, ids, rejectedImgPoints = aruco.detectMarkers(frame_gray, aruco_dict, parameters=parameters_aruco) # detect aruco markers
    
    corners = np.array([corners[i] for i, s in enumerate(ids) if s in np.arange(9)])  # get only 9 first arucos
    ids = np.array([[s[0]] for s in ids if s in np.arange(9)])
    
    # Refine aruco detection (drop erroneous detection)
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

    ### BACKGROUND SUBSTRACTION
    Background_s = np.abs(cv2.cvtColor(frame_ud, cv2.COLOR_BGR2GRAY) - Background)
    ret,thresh1 = cv2.threshold(Background_s,15,255,cv2.THRESH_BINARY)
    thresh1 = cv2.convertScaleAbs(thresh1)
    contours_Background, _ = cv2.findContours(thresh1, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)


    ### 4 Detect the particles and update particle memory
    if CropSize is not 'None':  # Check if all ROI mask is supplied (binary .bmp mask) and crop area of interest
        frame_ud = cv2.bitwise_and(frame_ud, frame_ud, mask = mask_crop)
        
    frame_hsv = cv2.cvtColor(frame_ud, cv2.COLOR_BGR2HSV)
    # Filter Balls

    Particles_blue = []
    Particles_green = []
    Particles_orange = []

    if len(contours_Background) > 0:
        for i, contour in enumerate(contours_Background):
            ((x, y), radius) = cv2.minEnclosingCircle(contour)
            area = cv2.contourArea(contour)
            perimeter = cv2.arcLength(contour, True)
            
            color, m = detectColor_contour_Point(frame_hsv, contour, rangescolor)
            
            if (area>5) & (area<200):
#                 cv2.putText(frame_display,'{}, {}'.format(str(m), color), (int(x), int(y)), cv2.FONT_HERSHEY_SIMPLEX,1,(255,255,255),1,cv2.LINE_AA)
                if perimeter == 0:
                    break
                circularity = 4*math.pi*(area/(perimeter*perimeter))
                if 0.6 < circularity < 1.2:
                    
                    # check color range
                    if color == 'blue_par':
                        M = cv2.moments(contour)
                        center = (int(M["m10"] / M["m00"]), int(M["m01"] / M["m00"]))
                        Particles_blue.append(center)

                        cv2.circle(frame_display, center, int(5), (255, 0, 0), 2)
                        
                    if color == 'green_par':
                        M = cv2.moments(contour)
                        center = (int(M["m10"] / M["m00"]), int(M["m01"] / M["m00"]))
                        Particles_green.append(center)

                        cv2.circle(frame_display, center, int(5), (120, 255, 0), 2)
                        
                    if color == 'orange_par':
                        M = cv2.moments(contour)
                        center = (int(M["m10"] / M["m00"]), int(M["m01"] / M["m00"]))
                        Particles_orange.append(center)

                        cv2.circle(frame_display, center, int(5), (0, 0, 255), 2)

#     ### 5 Retrace the ray, compute water surface normal and refraction vector
#     # needed:  Surface characteristics (e.g. water level), # extrinsics rvec and tvec, Intrinsic matrix
#     for each particle:

    # get waterlevel
    try:
        WaterLevel = WL.loc[i_t + BackgroundComputationFrames]
    except:
        WaterLevel = WL.mean()
    
    p_plane = np.array([0,0,WaterLevel]) # point of water plane, from water level sensor [0,0,waterlevel]
    n_plane = np.array([0,0,1]) # ASSUMING THAT THE PLANE IS PERFECTLY HORIZONTAL AND WORLD CRS IS ALLIGNED WITH G.

    cameraPosition = -np.matrix(cv2.Rodrigues(rvec_ran)[0]).T * np.matrix(tvec_ran)
    Pose_matrix_camtoworld = np.vstack([np.hstack([cv2.Rodrigues(rvec_ran)[0], tvec_ran]), [0,0,0,1]])

    P_b = []
    R_b = []
    for p in Particles_blue:
        inc_ray, P_water, refrac_ray = ray_retracing(p, IntrinsicM_i, cameraPosition, Pose_matrix_camtoworld, p_plane, n_plane, n1 = 1.001, n2 = 1.033)
        P_b.append(P_water)
        R_b.append(refrac_ray)
        
    P_g = []
    R_g = []
    for p in Particles_green:
        inc_ray, P_water, refrac_ray = ray_retracing(p, IntrinsicM_i, cameraPosition, Pose_matrix_camtoworld, p_plane, n_plane, n1 = 1.001, n2 = 1.033)
        P_g.append(P_water)
        R_g.append(refrac_ray)

    P_o = []
    R_o = []
    for p in Particles_orange:
        inc_ray, P_water, refrac_ray = ray_retracing(p, IntrinsicM_i, cameraPosition, Pose_matrix_camtoworld, p_plane, n_plane, n1 = 1.001, n2 = 1.033)
        P_o.append(P_water)
        R_o.append(refrac_ray)
        
#     ### 6 Save frame_time, particle_id:{id, positionframe, incident_vector_wc, position_waterSurface_wc, refraction_vector_wc}, rvec, tvec  (wc = "wold coordinates")
    
#     # write to output file    
    OutputFile_1.loc[time, 'rvec'] = rvec_ran.reshape(1,3)[0]
    OutputFile_1.loc[time, 'tvec'] = tvec_ran.reshape(1,3)[0]
    OutputFile_1.loc[time, 'cameraPosition'] =  np.array(cameraPosition).T[0]
    OutputFile_1.loc[time, 'P_b'] = P_b
    OutputFile_1.loc[time, 'R_b'] = R_b
    OutputFile_1.loc[time, 'P_g'] = P_g
    OutputFile_1.loc[time, 'R_g'] = R_g
    OutputFile_1.loc[time, 'P_o'] = P_o
    OutputFile_1.loc[time, 'R_o'] = R_o
    OutputFile_1.loc[time, 'Arucos_id'] = ids_f
    OutputFile_1.loc[time, 'ArucoCorners'] = FrameCoordinates

    
    cv2.putText(frame_display,"{} T:{} s, id:{}".format(camera, i_t, time), (40,40), cv2.FONT_HERSHEY_SIMPLEX,1.5,(0,255,0),2,cv2.LINE_AA)
    cv2.putText(frame_display,"rvec = {}".format([float('{:.3f}'.format(s)) for s in rvec_ran.reshape(1,3)[0]]), (40,90), cv2.FONT_HERSHEY_SIMPLEX,1,(0,255,0),2,cv2.LINE_AA)
    cv2.putText(frame_display,"Cam Pos = {} [m]".format([float('{:.3f}'.format(s[0])) for s in np.array(cameraPosition)]), (40,150), cv2.FONT_HERSHEY_SIMPLEX,1,(0,255,0),2,cv2.LINE_AA)

    cv2.imshow('frame_{}_{}'.format(experiment, camera),imutils.resize(frame_display, width = 1000))
    out.write(frame_display)
    
    if cv2.waitKey(1) & 0xFF == ord('q'):
        out.release()
        break
        
OutputFile_1.to_json(r'Outputs/{}_{}_ParticleLocations.json'.format(experiment, camera))

cv2.destroyAllWindows()
out.release()


# In[31]:
#
#from mpl_toolkits.mplot3d import Axes3D  # noqa: F401 unused import
#
#fig = plt.figure()
#ax = fig.add_subplot(111, projection='3d')
#
#for i in range(500):
#    try:
#        x = np.array(OutputFile_1['P_g'].iloc[i]).T[0]
#        y = np.array(OutputFile_1['P_g'].iloc[i]).T[1]
#        z = np.array(OutputFile_1['P_g'].iloc[i]).T[2]
#
#        ax.scatter(x, y, z, s = 1, color = 'g', marker='.', alpha = 0.5)
#        ax.set_zlim(-1, 1)
#    except:
#        continue
#ax.set_xlabel('x-wcoords [m]')
#ax.set_ylabel('y-wcoords [m]')
#ax.set_zlabel('Water Surface [m]')
#
#plt.savefig('Example_{}_{}_trackingParticles_worldCoordsProjectionWaterSurface.png'.format(experiment, camera))
#
