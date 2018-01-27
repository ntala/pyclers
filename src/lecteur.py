# -*- coding: utf-8 -*-
import cv2
import numpy as np

def get_contours_topologie(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    # A VOIR : le lissage est-il utile ?
    blurred = cv2.blur(gray,(3,3))
    # A VOIR : les paramètres de l'algo. de canny sont-ils optimaux ?
    ced_image = cv2.Canny(blurred,100,180)
    #cv2.imshow('canny', ced_image)
    contours, hierarchy = cv2.findContours(ced_image, cv2.RETR_TREE,
                                            cv2.CHAIN_APPROX_NONE)
    return (contours, hierarchy, gray)

def encadrement_motif(contour):
    rect = cv2.minAreaRect(contour)
    print rect
    box = cv2.cv.BoxPoints(rect) # cv2.boxPoints(rect) for OpenCV 3.x
    box = np.int0(box)
    print box
    return [box]

def marque_sommet(coordonnees,image,couleur):
    cv2.circle(image, coordonnees, 3, couleur,-1)
    
def marque_sommets(box,image) :
    couleurs = [(0,0,0),(255,0,0),(0,0,255),(0,255,255)]
    sommets = [tuple(sommet) for sommet in box[0]]
    # le premier sommet doit-être situé en bas à gauche du rectangle
    if sommets[3][1] < sommets[1][1] :
        sommets = (sommets[1], sommets[2], sommets[3], sommets[0])
    for couleur, sommet in zip(couleurs,sommets):
        marque_sommet(sommet,image,couleur)

def extrait_motif(image, box,nom_fichier) :
    sommets = [tuple(sommet) for sommet in box[0]]
    xmin = min(sommets, key = lambda t: t[0])[0]
    xmax = max(sommets, key = lambda t: t[0])[0]
    ymin = min(sommets, key = lambda t: t[1])[1]
    ymax = max(sommets, key = lambda t: t[1])[1]
    #print [xmin, xmax, ymin, ymax]
    zone = image[ymin:ymax,xmin:xmax]
    print zone
    cv2.imshow(nom_fichier,zone)
    
def encadre_motifs(image):
    contours, hierarchy, gris = get_contours_topologie(image)
    for rang, contour in enumerate(contours) :
        topologie = hierarchy[0][rang]
        # le test suivant vérifie que le contour ne contient pas
        # de contour, a un père et est le seul contour contenu 
        # par son père.
        if topologie[0] == -1 and topologie[1] == -1 \
        and topologie[2] == -1 and topologie[3] != -1:
            rang_pere = topologie[3]
            pere = contours[rang_pere]
            topologie_pere = hierarchy[0][rang_pere]
            # les contours rectangles qui m'intéressent sont 
            # 'souvent' inclus dans eux-même
            rang_grand_pere = topologie_pere[3]
            if rang_grand_pere != -1 :
                grand_pere = contours[rang_grand_pere]
                rectangle = encadrement_motif(grand_pere)
                cv2.drawContours(image, encadrement_motif(grand_pere),
                                 -1,(0,255,0), 2)
                marque_sommets(rectangle,image)
                
def encadre_et_extrait_motifs(image):
    contours, hierarchy, gris = get_contours_topologie(image)
    for rang, contour in enumerate(contours) :
        topologie = hierarchy[0][rang]
        # le test suivant vérifie que le contour ne contient pas
        # de contour, a un père et est le seul contour contenu 
        # par son père.
        if topologie[0] == -1 and topologie[1] == -1 \
        and topologie[2] == -1 and topologie[3] != -1:
            rang_pere = topologie[3]
            pere = contours[rang_pere]
            topologie_pere = hierarchy[0][rang_pere]
            # les contours rectangles qui m'intéressent sont 
            # 'souvent' inclus dans eux-même
            rang_grand_pere = topologie_pere[3]
            if rang_grand_pere != -1 :
                grand_pere = contours[rang_grand_pere]
                rectangle = encadrement_motif(grand_pere)
                cv2.drawContours(image, encadrement_motif(grand_pere),
                                 -1,(0,255,0), 2)
                marque_sommets(rectangle,image)
                extrait_motif(gris,rectangle, 'signal'+str(rang))
                
if __name__ == '__main__' :
    image = cv2.imread('img/plickerexample.jpg')
    cv2.imwrite('img/test2/capture.jpg', image)
    encadre_et_extrait_motifs(image)
    cv2.imshow('example',image)
    cv2.imwrite('img/test2/capture_contours.jpg', image)
    cv2.waitKey(0)
    
    image = cv2.imread('img/output.jpg')
    encadre_et_extrait_motifs(image)
    cv2.imshow('example',image)
    cv2.waitKey(0)
    
    cap = cv2.VideoCapture(0)
    while(True):
        retour, frame = cap.read()
        frame2 = frame.copy()
        ced_image = cv2.Canny(frame,100,180)
        encadre_motifs(frame)
        #cv2.imshow('canny',ced_image)
        cv2.imshow('contours',frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            #cv2.imwrite('img/capture.jpg', frame2)
            #cv2.imwrite('img/capture_contours.jpg', frame)
            #cv2.imwrite('img/capture_canny.jpg', ced_image)
            break
    cap.release()
    cv2.destroyAllWindows()
