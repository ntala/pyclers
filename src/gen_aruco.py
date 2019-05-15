import cv2
import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from cv2 import aruco
from PIL import Image

aruco_dict = aruco.Dictionary_get(aruco.DICT_4X4_50)

hauteur_texte = 1

marge_blanche = 2


fig = plt.figure()
nx = 1
ny = 2
for i in range(1, nx*ny+1):
    panneau = aruco.drawMarker(aruco_dict,i, 700)
    im = Image.fromarray(panneau)
    im.save(f"image_{i}.jpeg")
    ax = fig.add_subplot(ny,nx, i)
    img = aruco.drawMarker(aruco_dict,i, 700)
    plt.imshow(img, cmap = mpl.cm.gray, interpolation = "nearest")
    ax.axis("off")

#plt.savefig("_data/markers.pdf")
plt.show()
