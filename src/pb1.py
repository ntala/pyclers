import cv2
import numpy as np

cible = np.int0(((10,210),(10,10),(210,10)))
depart = np.int0(((210,10),(10,10),(10,210)))
mat = cv2.getAffineTransform(depart,cible)
