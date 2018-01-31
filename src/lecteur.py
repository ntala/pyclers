# -*- coding: utf-8 -*-
import cv2
import numpy as np

def get_contours_topologie(image):
    '''
    Prend en paramètre une image couleur en BGR et retourne :
    - la liste des contours détectés
    - leur topologie (arbre d'inclusion)
    - l'image convertie en niveau de gris
    '''
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
    '''
    Prend en paramètre une tableau de points et retourne :
    le tableau des sommets du rectangle encadrant au mieux le contour.
    '''
    rect = cv2.minAreaRect(contour)
    box = cv2.cv.BoxPoints(rect) # cv2.boxPoints(rect) for OpenCV 3.x
    box = np.int0(box)
    return [box]

def marque_sommet(coordonnees,image,couleur):
    """
    Prend en paramètre les coordonnées d'un point, une image au format
    BGR et une couleur au format (B,G,R) et inscrit dans l'image un 
    disque sur le point dont les coordonnées ont été données, dans la 
    couleur indiquée.
    """
    cv2.circle(image, coordonnees, 3, couleur,-1)
    
def marque_sommets(cadre,image) :
    '''
    Prend en paramètres un rectangle sous forme de tableau de points 
    et une image au format BGR et marque d'un disque respectivement 
    noir, bleu, rouge et jaune les quatre sommets du rectangle, en 
    commençant par le sommet le plus à gauche du côté le plus bas et en 
    tournant dans le sens des aiguilles d'une montre.
    '''
    couleurs = [(0,0,0),(255,0,0),(0,0,255),(0,255,255)]
    sommets = [tuple(sommet) for sommet in cadre[0]]
    # le premier sommet doit-être situé en bas à gauche du rectangle
    if sommets[3][1] < sommets[1][1] :
        sommets = (sommets[1], sommets[2], sommets[3], sommets[0])
    for couleur, sommet in zip(couleurs,sommets):
        marque_sommet(sommet,image,couleur)

def extrait_motif(image, box, nom_fichier) :
    """
    Prend en paramètre une image en nuance de gris, un rectangle sous 
    forme du tableau de sommets supposé encadrer un panneau et un nom de
    fichier.
    Affiche l'image du panneau redressée et découpée en rectangles 
    significatifs et retourne la "matrice" résumant le panneau.
    """
    sommets = [tuple(sommet) for sommet in box[0]]
    xmin = min(sommets, key = lambda t: t[0])[0]
    xmax = max(sommets, key = lambda t: t[0])[0]
    ymin = min(sommets, key = lambda t: t[1])[1]
    ymax = max(sommets, key = lambda t: t[1])[1]
    zone = image[ymin:ymax,xmin:xmax]
    depart = np.float32(tuple(map(lambda t: (t[0]-xmin, t[1]-ymin),
                                    sommets[:3])))
    cible = np.float32(((10,210),(10,10),(210,10)))
    mat = cv2.getAffineTransform(depart,cible)
    zone_redressee = cv2.warpAffine(zone,mat,(210,210))
    zone_contrastee = cv2.threshold(zone_redressee,
                                   80,
                                   255,
                                   cv2.THRESH_BINARY)[1]
    #zone_contrastee = cv2.adaptiveThreshold(zone_redressee,
                                            #255,
                                            #cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                            #cv2.THRESH_BINARY,11,2)
    ECART = 5
    code = [[0,0,0,0,0],
            [0,0,0,0,0],
            [0,0,0,0,0],
            [0,0,0,0,0],
            [0,0,0,0,0]]
    for i in xrange(5) :
        for j in xrange(5) :
            xmin = 10+40*i+ECART
            xmax = 10+40*(i+1)-ECART
            ymin = 10+40*j+ECART
            ymax = 10+40*(j+1)-ECART
            zone = zone_contrastee[ymin:ymax,xmin:xmax]
            
            if cv2.mean(zone)[0]>127 :
                code[j][i] = 1
            cv2.rectangle(zone_contrastee,
                          (10+40*i+ECART,10+ECART+40*j),
                          (10+40*(i+1)-ECART,10+40*(j+1)-ECART),
                          (255,255,255))
    cv2.imshow(nom_fichier,zone_contrastee)
    return code

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
                cv2.drawContours(image, rectangle,
                                 -1,(0,255,0), 2)
                marque_sommets(rectangle,image)
                #if cv2.contourArea(grand_pere) > 400 :
                matrice = extrait_motif(gris,rectangle, 'signal')
                print 'matrice'
                for l in matrice :
                    print l
                #cv2.imshow('signal'+str(rang),image)
                
if __name__ == '__main__' :
    cv2.waitKey(1)
    cap = cv2.VideoCapture(0)
    while(True):
        retour, frame = cap.read()
        frame2 = frame.copy()
        ced_image = cv2.Canny(frame,100,180)
        #encadre_motifs(frame)
        encadre_et_extrait_motifs(frame)
        #cv2.imshow('canny',ced_image)
        cv2.imshow('contours',frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            #cv2.imwrite('img/capture.jpg', frame2)
            #cv2.imwrite('img/capture_contours.jpg', frame)
            #cv2.imwrite('img/capture_canny.jpg', ced_image)
            break
    cap.release()
    cv2.destroyAllWindows()
