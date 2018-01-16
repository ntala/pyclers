# -*- coding: utf-8 -*-
import cv2
import numpy as np

def est_quadrilatere(contour):
    peri = cv2.arcLength(contour, True)
    approx = cv2.approxPolyDP(contour, 0.02 * peri, True)
    res = len(approx) == 4 and peri > 10
    return res

def get_contours_topologie(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    mystere, threshold = cv2.threshold(gray,100,255,cv2.THRESH_BINARY)
    contours, hierarchy = cv2.findContours(threshold, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)
    return (contours, hierarchy)

def redessine_quadrilateres(image):
    contours, hierarchy = get_contours_topologie(image)
    for contour in contours : 
        if est_quadrilatere(contour) :
            cv2.drawContours(image, contour,-1,(0,0,2555),2)
    
def redessine_quadrilateres_et_peres(image):
    contours, hierarchy = get_contours_topologie(image)
    print hierarchy[0]
    for rang, contour in enumerate(contours) : 
        if est_quadrilatere(contour) :
            cv2.drawContours(image, contour,-1,(0,0,2555),2)
            print rang
            #print contours[rang]
            print hierarchy[0][rang]
            try :
                rang_pere = hierarchy[0][rang][3]
                if rang_pere != -1 :
                    pere = contours[rang_pere]
                    cv2.drawContours(image,pere,-1,(0,255,0),2)
            except :
                pass
            
if __name__ == '__main__' :
    cap = cv2.VideoCapture(0)
    #image = cv2.imread('/img/output.jpg')
    #print image
    #redessine_quadrilateres_et_peres(image)
    #cv2.imshow('example',image)
    while(True):
        ret, frame = cap.read()
        frame2 = frame.copy()
        redessine_quadrilateres_et_peres(frame)
        cv2.imshow('frame',frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            cv2.imwrite('img/capture.jpg', frame2)
            cv2.imwrite('img/capture_contours.jpg', frame)
            break
    cap.release()
    cv2.destroyAllWindows()
