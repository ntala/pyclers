# -*- coding: utf-8 -*-
import cv2

def est_quadrilatere(contour):
    peri = cv2.arcLength(contour, True)
    approx = cv2.approxPolyDP(contour, 0.02 * peri, True)
    res = len(approx) == 4
    return res

def get_contours_topologie(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    mystere, threshold = cv2.threshold(gray,150,255,cv2.THRESH_BINARY)
    contours, hierarchy = cv2.findContours(threshold, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)
    return (contours, hierarchy)

def redessine_quadrilateres(image):
    contours, hierarchy = get_contours_topologie(image)
    for contour in contours : 
        if est_quadrilatere(contour) :
            cv2.drawContours(image, contour,-1,(0,0,2555),2)
    
if __name__ == '__main__' :
    cap = cv2.VideoCapture(0)                
    while(True):
        ret, frame = cap.read()
        redessine_quadrilateres(frame)
        cv2.imshow('frame',frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    cap.release()
    cv2.destroyAllWindows()
