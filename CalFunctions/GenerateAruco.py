import numpy as np
import cv2
import cv2.aruco as aruco
import os

#%%
# Select type of aruco marker (size)
aruco_dict = aruco.Dictionary_get(aruco.DICT_4X4_1000)

directory = r'Aruco_4x4_1000_png'
N = 5

if not os.path.exists(directory):
    os.makedirs(directory)
    
#%%
# Create an image from the marker
# second param is ID number
# last param is total image size

for id_a in range(N):
    img = aruco.drawMarker(aruco_dict, id_a, 800)
    img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
    h,w,_ = img.shape
    
    # Add Centre Circle
    img = cv2.circle(img, (int(h/2.),int(w/2.)), int(h/200), (0,0,255), -1)
    
    # Add white Border
    S = int(0.1*h)   
    base_size=h+2*S,w+2*S,3
    # make a 3 channel image for base which is slightly larger than target img
    base=np.zeros(base_size,dtype=np.uint8)
    cv2.rectangle(base,(0,0),(w+2*S,h+2*S),(255,255,255),-1) # really thick white rectangle
    base[S:h+S,S:w+S]=img # this works
    img = base.copy()
    
    
    
    cv2.imwrite(os.path.join(directory, f'Aruco_id_{id_a}.jpg'), img)
    
    cv2.imshow('frame', img)
    # Exit on any key
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    
