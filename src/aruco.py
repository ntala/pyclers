# -*- coding: utf-8 -*-
"""
Created on Sat May 11 13:15:55 2019

@author: caged2013
"""


# import numpy as np
import cv2#, PIL
from cv2 import aruco
# import matplotlib.pyplot as plt
# import matplotlib as mpl
# import pandas as pd

aruco_dict = aruco.Dictionary_get(aruco.DICT_4X4_100)

def marque(image, point):
    x,y = map(int, point)
    cv2.rectangle(image, (x-20, y-20), (x+20, y+20), (0,0,255))

cap = cv2.VideoCapture(0)
while True :
    retour, frame = cap.read()
    if retour :
            coins, ids, rejets = aruco.detectMarkers(frame,aruco_dict)
            if ids :
                premier_coin = coins[0][0][0]
                marque(frame, premier_coin)
                print(premier_coin)
            cv2.imshow('capture',frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
cap.release()
cv2.destroyWindow('capture')