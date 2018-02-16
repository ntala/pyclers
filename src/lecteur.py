# -*- coding: utf-8 -*-
import cv2
import numpy as np
import os
from itertools import product

CATALOGUE = {28831199: (19, 'D'), 29323263: (32, 'B'), 23063547: (11, 'C'), 33501183: (8, 'B'), 33516503: (5, 'B'), 29322235: (34, 'B'), 33548799: (8, 'D'), 27242493: (26, 'A'), 32975835: (27, 'B'), 31436789: (19, 'A'), 29356029: (8, 'A'), 33025017: (36, 'A'), 33532917: (35, 'A'), 33549819: (32, 'D'), 32992211: (23, 'B'), 32976887: (28, 'B'), 28814847: (3, 'D'), 33516507: (3, 'B'), 32993267: (22, 'B'), 33501147: (11, 'B'), 32975859: (30, 'B'), 26734577: (29, 'A'), 26718201: (30, 'A'), 20951007: (36, 'C'), 31452671: (4, 'D'), 31451615: (28, 'D'), 23064539: (13, 'C'), 33517559: (4, 'B'), 25161683: (28, 'C'), 18870227: (29, 'C'), 33516531: (6, 'B'), 31435263: (12, 'D'), 31436283: (36, 'D'), 33532383: (24, 'D'), 25161695: (4, 'C'), 33500127: (9, 'B'), 33517523: (7, 'B'), 28815357: (14, 'A'), 25160667: (14, 'C'), 30928351: (21, 'D'), 32976863: (25, 'B'), 29322231: (36, 'B'), 23047167: (35, 'C'), 25160703: (2, 'C'), 33024479: (25, 'D'), 32993279: (16, 'B'), 20950015: (34, 'C'), 33009653: (7, 'A'), 31436793: (18, 'A'), 20967411: (24, 'C'), 26717663: (23, 'D'), 25160691: (26, 'C'), 30911999: (5, 'D'), 33008625: (39, 'A'), 33025535: (1, 'D'), 27240927: (30, 'D'), 18869247: (3, 'C'), 30928885: (21, 'A'), 18870239: (5, 'C'), 30928379: (37, 'D'), 29305855: (40, 'B'), 23064575: (1, 'C'), 32976891: (26, 'B'), 33009657: (6, 'A'), 33532411: (40, 'D'), 32992251: (18, 'B'), 29356017: (9, 'A'), 29322195: (39, 'B'), 23064535: (21, 'C'), 20951035: (40, 'C'), 29338111: (10, 'D'), 25161719: (16, 'C'), 32992223: (17, 'B'), 27258873: (24, 'A'), 29355001: (40, 'A'), 20966367: (6, 'C'), 33500151: (12, 'B'), 30912509: (22, 'A'), 18852831: (39, 'C'), 33025013: (37, 'A'), 27257343: (14, 'D'), 29323227: (35, 'B'), 20967383: (20, 'C'), 33550325: (1, 'A'), 28813791: (27, 'D'), 32992247: (20, 'B'), 26718197: (31, 'A'), 20966355: (30, 'C'), 23048159: (37, 'C'), 23064563: (25, 'C'), 23063507: (31, 'C'), 33009119: (17, 'D'), 29339641: (10, 'A'), 23063519: (7, 'C'), 33008637: (38, 'A'), 30910943: (29, 'D'), 27242481: (27, 'A'), 27241983: (6, 'D'), 33549297: (33, 'A'), 26734589: (28, 'A'), 33517535: (1, 'B'), 25161723: (8, 'C'), 18853887: (33, 'C'), 25145343: (32, 'C'), 25144287: (38, 'C'), 23063543: (19, 'C'), 26734079: (7, 'D'), 29323251: (38, 'B'), 30928889: (20, 'A'), 33501143: (13, 'B'), 29339131: (34, 'D'), 32975831: (29, 'B'), 33533937: (3, 'A'), 33009147: (33, 'D'), 28831733: (13, 'A'), 32993243: (19, 'B'), 31453169: (17, 'A'), 27258869: (25, 'A'), 33532921: (34, 'A'), 20966391: (18, 'C'), 29323223: (37, 'B'), 18869207: (23, 'C'), 20966395: (10, 'C'), 30912497: (23, 'A'), 27258335: (22, 'D'), 33500115: (15, 'B'), 32975871: (24, 'B'), 33500155: (10, 'B'), 32993239: (21, 'B'), 33533949: (2, 'A'), 27258363: (38, 'D'), 29339103: (18, 'D'), 18869211: (15, 'C'), 29322207: (33, 'B'), 26717691: (39, 'D'), 31436255: (20, 'D'), 20967387: (12, 'C'), 33517563: (2, 'B'), 33008127: (9, 'D'), 33026033: (5, 'A'), 25160663: (22, 'C'), 33549791: (16, 'D'), 29355519: (2, 'D'), 29354463: (26, 'D'), 18869235: (27, 'C'), 31453181: (16, 'A'), 28815345: (15, 'A'), 32976851: (31, 'B'), 33501171: (14, 'B'), 26733023: (31, 'D'), 29339637: (11, 'A'), 30927359: (13, 'D'), 18870263: (17, 'C'), 28831227: (35, 'D'), 28831737: (12, 'A'), 33549309: (32, 'A'), 18870267: (9, 'C'), 33026045: (4, 'A'), 26716671: (15, 'D'), 28830207: (11, 'D')}

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

def extrait_matrice(image, box, nom_fichier) :
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
    #print zone
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
    code = [[1,1,1,1,1],
            [1,1,1,1,1],
            [1,1,1,1,1],
            [1,1,1,1,1],
            [1,1,1,1,1]]
    for j in xrange(5) :
        for i in xrange(5) :
            xmin = 40*i+ECART
            xmax = 40*(i+1)-ECART
            ymin = 40*j+ECART
            ymax = 40*(j+1)-ECART
            zone = zone_contrastee[ymin:ymax,xmin:xmax]
            
            if cv2.mean(zone)[0]>127 :
                code[j][i] = 0
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
    return code

def code_depuis_matrice(mat):
    """
    Prends en paramètre une matrice résumant un panneau et renvoie le 
    nombre codant pour cette matrice. 
    """
    coding_string = ""
    for i,j in product(xrange(len(mat)),xrange(len(mat))):
        coding_string += str(mat[i][j])
    print coding_string
    return int(coding_string,2)

def encadre_et_extrait_motifs(image,visu=False):
    """
    extrait tous les panneaux détectés dans l'image et renvoie la liste
    des matrices lues
    """
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
                    matrice = extrait_matrice(gris,rectangle, 'signal')
                    res.append(matrice)
                    print matrice
                #cv2.imshow('signal'+str(rang),image)
    return res
    
def reconnait_panneaux(image) :
    return [CATALOGUE.get(code_depuis_matrice(matrice),None) for matrice in encadre_et_extrait_motifs(image)]
    
def encode_positions (chemin_image):
    """
    prend en paramètre le chemin d'un panneau de référence unique et 
    renvoie la liste des codes correspondant au panneau dans les 
    différentes orientations.
    """
    image_ref = cv2.imread(chemin_image)
    #cv2.imshow("image_A",image_ref)
    
    image_travail = image_ref
    rows,cols,bgr = image_ref.shape
    listBoxA=encadre_et_extrait_motifs(image_travail,False)
        
    M = cv2.getRotationMatrix2D((cols/2,rows/2),90,1)
    image_travail = cv2.warpAffine(image_ref,M,(cols,rows))
    listBoxB=encadre_et_extrait_motifs(image_travail,False)
    #cv2.imshow("image_B",image_travail)
    
    M = cv2.getRotationMatrix2D((cols/2,rows/2),180,1)
    image_travail = cv2.warpAffine(image_ref,M,(cols,rows))
    listBoxC=encadre_et_extrait_motifs(image_travail,False)
    #cv2.imshow("image_C",image_travail)
    
    M = cv2.getRotationMatrix2D((cols/2,rows/2),-90,1)
    image_travail = cv2.warpAffine(image_ref,M,(cols,rows))
    listBoxD=encadre_et_extrait_motifs(image_travail,False)
    #cv2.imshow("image_D",image_travail)
    
    data=np.array([[listBoxA[0],listBoxB[0],listBoxC[0],listBoxD[0]]])
    #print data
    
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
    #matrice =[
    #[1,0,0,0,1],
    #[0,1,1,1,0],
    #[0,1,0,1,0],
    #[0,1,1,1,0],
    #[1,0,0,0,1]]
    #print code_depuis_matrice(matrice)
    #creation de la base lecture des images dans le repertoire (un repertoire par classe)
    #une liste de nom associée à un tableau meme nombre de ligne et quatres colonnes
    #listeName,listeIma=build_db('img/db/')
    #print encode_positions('img/db/rb_034.jpg')
    #image = cv2.imread('img/reference/reference-0002.jpg')
    #print encadre_et_extrait_motifs(image)
    #print(listeName)
    #print(listeIma)
    image = cv2.imread('img/myphoto4.jpg')
    print reconnait_panneaux(image)
    image = cv2.imread('img/test_15B.jpg')
    print reconnait_panneaux(image)
    #cap = cv2.VideoCapture(0)
    ##Letter_code=['A','B','C','D']
    ###tant qu'il y a des noms d'eleves dans la liste
    #while True:
        #retour, frame = cap.read()
        ##frame2 = frame.copy()
        #print reconnait_panneaux(frame)
        #ced_image = cv2.Canny(frame,100,180)
        ##encadre_motifs(frame)
        #listBox=encadre_et_extrait_motifs(frame,True)
        #for box in enumerate(listBox):
            #print(box)
            #box_index=np.where(listeIma == box) 
            #print(box_index)
            #if np.size(box_index) > 0:
                ##affichage du nom de l'eleve et de la reponse
                #print(listeName[int(box_index[0])])
                #print(Letter_code[int(box_index[1])])
                ##je supprime des deux listes (nom et index) 
                #del listeName[int(box_index[0])]
                #del listeIma[int(box_index[0])]
        ##cv2.imshow('canny',ced_image)
        #cv2.imshow('contours',frame)
        #if cv2.waitKey(1) & 0xFF == ord('q'):
            ##cv2.imwrite('img/capture.jpg', frame2)
            ##cv2.imwrite('img/capture_contours.jpg', frame)
            ##cv2.imwrite('img/capture_canny.jpg', ced_image)
            #break
    #cap.release()
    cv2.destroyAllWindows()
