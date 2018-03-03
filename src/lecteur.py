# -*- coding: utf-8 -*-
import threading
import os
from itertools import product

import Tkinter as Tk
import cv2
import numpy as np

from data_reference import CATALOGUE, CLASSE_TEST

#variables globales
eleves_restant = []

def get_contours_topologie(image):
    '''
    Prend en paramètre une image couleur en BGR et retourne :
    - la liste des contours détectés
    - leur topologie (arbre d'inclusion)
    - l'image convertie en niveau de gris
    '''
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    blurred = cv2.blur(gray,(3,3))
    # A VOIR : les paramètres de l'algo. de canny sont-ils optimaux ?
    ced_image = cv2.Canny(blurred,100,180)
    contours, hierarchy = cv2.findContours(ced_image, cv2.RETR_TREE,
                                            cv2.CHAIN_APPROX_NONE)
    return (contours, hierarchy, gray)

def encadrement_contour(contour):
    '''
    Prend en paramètre une tableau de points et retourne :
    le tableau des sommets du rectangle encadrant au mieux le contour.
    '''
    rect = cv2.minAreaRect(contour)
    box = cv2.cv.BoxPoints(rect) # cv2.boxPoints(rect) for OpenCV 3.x
    box = np.int0(box)
    return [box]

def extrait_identifiant(image,box):
    """
    Prend en paramètre une image en nuance de gris et un rectangle sous 
    forme du tableau de sommets supposé encadrer un panneau.
    Retourne l'entier identifiant le panneau.
    """
    str_code = ''
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
    for j in xrange(5) :
        for i in xrange(5) :
            xmin = 40*i+ECART
            xmax = 40*(i+1)-ECART
            ymin = 40*j+ECART
            ymax = 40*(j+1)-ECART
            zone = zone_contrastee[ymin:ymax,xmin:xmax]
            
            if cv2.mean(zone)[0]>127 :
                str_code += '0';
            else:
                str_code += '1';
            cv2.rectangle(zone_contrastee,
                          (40*i+ECART,ECART+40*j),
                          (40*(i+1)-ECART,40*(j+1)-ECART),
                          (255,255,255))
    int_ret = int(str_code,2)  
    return int_ret

def extrait_identifiants(image):
    """
    extrait tous les panneaux détectés dans l'image et renvoie la liste
    des identifiants lus.
    """
    res=[]
    contours, hierarchy, im_gris = get_contours_topologie(image)
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
                if cv2.contourArea(grand_pere) > 0 :
                    res.append(extrait_identifiant(im_gris,rectangle))
    return res
    

def reconnait_panneaux(image) :
    return [CATALOGUE.get(identifiant)\
    for identifiant in extrait_identifiants(image)\
    if CATALOGUE.get(identifiant)]

def releve_absents(classe):
    """
    Prend en paramètre la liste des élèves d'une classe.
    Invite l'utilisateur à cocher les absents via une fenêtre Tkinter.
    Renvoie la liste des élèves effectivement présents.
    """
    global eleves_restant
    fenetre = Tk.Tk()
    for eleve in classe :
        presence = Tk.IntVar()
        presence.set(1)
        eleve.update({'present': presence})
        bouton = Tk.Checkbutton(fenetre, 
                            text = eleve['nom'] + ' ' + eleve['prenom'],
                            onvalue = 0, offvalue = 1,
                            variable = eleve['present'],
                            justify = Tk.RIGHT)
        bouton.pack(anchor = "w")
    def valider_saisie():
        fenetre.quit()
        fenetre.destroy()
    bouton = Tk.Button(fenetre, text="Valider", command=valider_saisie)
    bouton.pack()
    fenetre.mainloop()
    #préparation de la liste des élèves restant avant son renvoi
    eleves_restant = [el for el in classe if el['present'].get()]
    for eleve in eleves_restant :
        eleve.pop('present')
    
def affiche_eleves_restant():
    affichage_eleves_restant = Tk.Tk()
    cadre_eleves_restant = Tk.Frame(affichage_eleves_restant)
    for eleve in eleves_restant :
        etiquette = Tk.Label(cadre_eleves_restant,
                        text = eleve['nom'] + ' ' + eleve['prenom'])
        etiquette.pack()
    cadre_eleves_restant.pack()
    
    def met_a_jour_liste(cadre=cadre_eleves_restant,
                         fenetre=affichage_eleves_restant):
        cadre = fenetre.winfo_children()[0]
        cadre.destroy()
        if eleves_restant == [] :
            fenetre.destroy()
        try :
            cadre = Tk.Frame(fenetre)
            for eleve in eleves_restant :
                etiquette = Tk.Label(cadre,text = eleve['nom'] + ' ' + eleve['prenom'])
                etiquette.pack()
            cadre.pack()
            fenetre.after(500, met_a_jour_liste)
        except :
            pass
    affichage_eleves_restant.after(500, met_a_jour_liste)
    affichage_eleves_restant.mainloop()
    
def scanne_flux_video(classe,camera):
    global eleves_restant
    reponses = []
    cap = cv2.VideoCapture(camera)
    # with eleves_restant as listing :
    while eleves_restant != []:
        retour, frame = cap.read()
        cv2.imshow('capture',frame)
        for identifiant, reponse in reconnait_panneaux(frame) :
            if classe[identifiant-1] in eleves_restant :
                reponses.append((identifiant,reponse))
                rang_eleve = eleves_restant.index(classe[identifiant-1])
                eleves_restant.pop(rang_eleve)
                print eleves_restant
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    cap.release
    print reponses
    cv2.destroyAllWindows()
    
def scanner_en_direct(classe, camera=0):
    releve_absents(classe)
    t = threading.Thread(target=affiche_eleves_restant)
    t.start()
    scanne_flux_video(classe, camera)
    t.join()

    
if __name__ == '__main__' :
    image = cv2.imread('img/myphoto4.jpg')
    print reconnait_panneaux(image)
    image = cv2.imread('img/test_15B.jpg')
    print reconnait_panneaux(image)
    image = cv2.imread('img/capture4.jpg')
    print reconnait_panneaux(image)
    scanner_en_direct(CLASSE_TEST,1)
    

