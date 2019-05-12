# -*- coding: utf-8 -*-

import cv2  # , PIL
import numpy as np
from cv2 import aruco

# import matplotlib.pyplot as plt
# import matplotlib as mpl
# import pandas as pd

aruco_dict = aruco.Dictionary_get(aruco.DICT_4X4_100)

# def marque(image, point):
#     x,y = map(int, point)
#     cv2.rectangle(image, (x-20, y-20), (x+20, y+20), (0,0,255))


def ordonnees_repere(points):
    points_2 = np.concatenate([points[1:], points[0:1]])
    res = (points[:,1] + points_2[:,1]) / 2
    return res  


def presents(panneaux_detectes):
    try :
        if panneaux_detectes :
            return True
        else :
            return False
    except :
        if panneaux_detectes.any():
            return True
        else :
            return False


cap = cv2.VideoCapture(1)
while True :
    retour, frame = cap.read()
    if retour :
            listes_coins, num_panneaux, rejets = aruco.detectMarkers(frame,aruco_dict)
            if presents(num_panneaux):
                for rang, num in enumerate(num_panneaux) :
                    # premier_coin = listes_coins[0][rang][0]
                    # marque(frame, premier_coin)
                    tableau_ordonnees = ordonnees_repere(listes_coins[0][rang])
                    ordonnee_max = tableau_ordonnees.max()
                    print(np.where(tableau_ordonnees==ordonnee_max)[0][0])
            cv2.imshow('capture',frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
cap.release()
cv2.destroyWindow('capture')
