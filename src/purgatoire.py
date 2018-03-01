# -*- coding: utf-8 -*-
import os, os.path, cv2

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
    zone_redressee = cv2.warpAffine(zone,mat,(200,200))
    zone_contrastee = cv2.threshold(zone_redressee,
                                   80,
                                   255,
                                   cv2.THRESH_BINARY)[1]
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
                rectangle = encadrement_contour(grand_pere)
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
    liste_primaire = [CATALOGUE.get(code_depuis_matrice(matrice)) \
    for matrice in encadre_et_extrait_motifs(image)]
    return [signal for signal in liste_primaire if signal]

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
A = 'A'
B = 'B'
C = 'C'
D = 'D'

liste_initiale = [B,D,D,D,A,A,A,B,A,D,C,D,A,B,A,C,D,C,D,D,B,B,A,B,C,
                  D,B,B,D,B,C,B,C,C,C,B,B,C,D,A]


def cycle(lettre):
    if lettre == A :
        return [A, B, C, D]
    elif lettre == B :
        return [B, C, D, A]
    elif lettre == C :
        return [C, D, A, B]
    elif lettre == D :
        return [D, A, B, C]

def rotation(matrice):                              
    return [[matrice[i][4-j] for i in range(5)] for j in range(5)]

def declinaisons_panneau(panneau, chemin_image, lettre):
    res = {}
    lettres = cycle(lettre)
    image = cv2.imread(chemin_image)
    matrice_initiale = encadre_et_extrait_motifs(image)[0]
    matrices = [matrice_initiale,
                rotation(matrice_initiale),
                rotation(rotation(matrice_initiale)),
                rotation(rotation(rotation(matrice_initiale)))
                ]
    for lettre_courante, matrice in zip(lettres, matrices) :
        cle = code_depuis_matrice(matrice)
        res.update({cle : (panneau, lettre_courante)})
    return res


if __name__ == '__main__' :
    catalogue = {}
    os.chdir('img/reference/')
    liste_fichiers = os.listdir(os.path.curdir)
    liste_fichiers.sort()
    for i,fichier,lettre_initiale in zip(xrange(len(liste_fichiers)),
                                         liste_fichiers,
                                         liste_initiale):
        catalogue.update(declinaisons_panneau(i+1,fichier,lettre_initiale))
    print catalogue
