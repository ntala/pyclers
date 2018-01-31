# -*- coding: utf-8 -*-
import os, os.path, cv2

from lecteur import encadre_et_extrait_motifs

A = 'A'
B = 'B'
C = 'C'
D = 'D'

liste_initiale = [B,D,D,D,A,A,A,B,A,D,C,D,A,B,A,C,D,C,D,D,B,B,A,B,C,
                  D,B,B,D,B,C,B,C,C,C,B,B,C,D,A]
                  
liste_fichiers = os.listdir('img/reference')
liste_fichiers.sort()

print liste_fichiers


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

if __name__ == '__main__' :
    from random import randint
    matrice_1 = [[randint(0,1) for j in range(5)] for i in range(5)]

    for l in matrice_1 : 
        print l

    print ''        

    matrice_2 = rotation(matrice_1)
    
    for l in matrice_2 : 
        print l
    os.chdir('img/reference/')
    listefichier = os.listdir(os.path.curdir)
    i=0
    for fichier in liste_fichiers :
        print str(2*i+1) + ' ' + str(2*i+2)
        image = cv2.imread(fichier)
        encadre_et_extrait_motifs(image)
        i=i+1
        cv2.imshow(str(i),image)
        cv2.waitKey(0)
