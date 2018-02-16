# -*- coding: utf-8 -*-
import os, os.path, cv2

from lecteur import encadre_et_extrait_motifs, extrait_matrice, \
                     code_depuis_matrice



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
