# -*- coding: utf-8 -*-
import cv2
import numpy as np
import os

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
    str_code=''
    sommets = [tuple(sommet) for sommet in box[0]]
    # le premier sommet doit-être situé en bas à gauche du rectangle
    if sommets[3][1] < sommets[1][1] :
        sommets = (sommets[1], sommets[2], sommets[3], sommets[0])
    xmin = min(sommets, key = lambda t: t[0])[0]
    xmax = max(sommets, key = lambda t: t[0])[0]
    ymin = min(sommets, key = lambda t: t[1])[1]
    ymax = max(sommets, key = lambda t: t[1])[1]
    zone = image[ymin:ymax,xmin:xmax]
    depart = np.float32(tuple(map(lambda t: (t[0]-xmin, t[1]-ymin),
                                    sommets[:3])))
    cible = np.float32(((0,200),(0,0),(200,0)))
    mat = cv2.getAffineTransform(depart,cible)
    zone_redressee = cv2.warpAffine(zone,mat,(200,200))
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
    for j in xrange(5) :
        for i in xrange(5) :
            xmin = 40*i+ECART
            xmax = 40*(i+1)-ECART
            ymin = 40*j+ECART
            ymax = 40*(j+1)-ECART
            zone = zone_contrastee[ymin:ymax,xmin:xmax]
            
            if cv2.mean(zone)[0]>127 :
                code[j][i] = 1
                str_code +='0';
            else:
                str_code +='1';
            cv2.rectangle(zone_contrastee,
                          (40*i+ECART,ECART+40*j),
                          (40*(i+1)-ECART,40*(j+1)-ECART),
                          (255,255,255))
    
    str_baseCode='1000101110010100111010001' #= 18295249
    if (int(str_code,2) & int(str_baseCode,2)) == 18295249:
        int_ret=int(str_code,2)
    else:
        int_ret=18295249
    #cv2.imshow(nom_fichier,zone_contrastee)    
    return int_ret
    

def encadre_et_extrait_motifs(image,visu):
    res=[]
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
            # les contours rectangles qui m'intéressent sont '
            # 'souvent' inclus dans eux-même
            rang_grand_pere = topologie_pere[3]
            if rang_grand_pere != -1 :
                grand_pere = contours[rang_grand_pere]
                rectangle = encadrement_motif(grand_pere)
                if visu:
                    cv2.drawContours(image, rectangle,-1,(0,255,0), 2)
                    marque_sommets(rectangle,image)
                if cv2.contourArea(grand_pere) > 0 :
                    matrice = extrait_motif(gris,rectangle, 'signal')
                    res.append(matrice)
                #cv2.imshow('signal'+str(rang),image)
    return res

def build_db(path_folder):
    student_name=[]
    laDB=np.array([[0,0,0,0]])
    listF = os.listdir(path_folder)
    for ima in listF :
        image = cv2.imread(path_folder+ima)
        imaSav = image
        rows,cols,rgb = image.shape
        listBoxA=encadre_et_extrait_motifs(image,False)
        M = cv2.getRotationMatrix2D((cols/2,rows/2),90,1)
        image = cv2.warpAffine(imaSav,M,(cols,rows))
        listBoxB=encadre_et_extrait_motifs(image,False)
        M = cv2.getRotationMatrix2D((cols/2,rows/2),180,1)
        image = cv2.warpAffine(imaSav,M,(cols,rows))
        listBoxC=encadre_et_extrait_motifs(image,False)
        M = cv2.getRotationMatrix2D((cols/2,rows/2),-90,1)
        image = cv2.warpAffine(imaSav,M,(cols,rows))
        listBoxD=encadre_et_extrait_motifs(image,False)
        data=np.array([[listBoxA[0],listBoxB[0],listBoxC[0],listBoxD[0]]])
        laDB=np.concatenate((laDB,data), axis=0)
        student_name.append(ima)
    #bon je reconnais que faire une ligne avec des 0 et l'effacer à la fin ce n'est pas propre et il y a moyen de faire mieux
    print(laDB)
    print(np.size(laDB))
    np.delete(laDB,0)
    return student_name,laDB

if __name__ == '__main__' :
    #creation de la base lecture des images dans le repertoire (un repertoire par classe)
    #une liste de nom associée à un tableau meme nombre de ligne et quatres colonnes
    listeName,listeIma=build_db('img/db/')
    print(listeName)
    print(listeIma)
    cv2.waitKey(1)
    cap = cv2.VideoCapture(0)
    Letter_code=['A','B','C','D']
    #tant qu'il y a des noms d'eleves dans la liste
    while(np.size(listeName)>0):
        retour, frame = cap.read()
        frame2 = frame.copy()
        ced_image = cv2.Canny(frame,100,180)
        #encadre_motifs(frame)
        listBox=encadre_et_extrait_motifs(frame,True)
        for box in enumerate(listBox):
            print(box)
            box_index=np.where(listeIma == box) 
            print(box_index)
            if np.size(box_index) > 0:
                #affichage du nom de l'eleve et de la reponse
                print(listeName[int(box_index[0])])
                print(Letter_code[int(box_index[1])])
                #je supprime des deux listes (nom et index) 
                del listeName[int(box_index[0])]
                del listeIma[int(box_index[0])]
        #cv2.imshow('canny',ced_image)
        cv2.imshow('contours',frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            #cv2.imwrite('img/capture.jpg', frame2)
            #cv2.imwrite('img/capture_contours.jpg', frame)
            #cv2.imwrite('img/capture_canny.jpg', ced_image)
            break
    cap.release()
    cv2.destroyAllWindows()
