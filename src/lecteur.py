# -*- coding: utf-8 -*-
import cv2
import numpy as np

def get_contours_topologie(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    blurred = cv2.blur(gray,(3,3))
    ced_image = cv2.Canny(blurred,100,180)
    cv2.imshow('canny', ced_image)
    contours, hierarchy = cv2.findContours(ced_image, cv2.RETR_TREE,
                                            cv2.CHAIN_APPROX_NONE)
    return (contours, hierarchy)
    
def redessine_quadrilateres_et_peres(image):
    contours, hierarchy = get_contours_topologie(image)
    for rang, contour in enumerate(contours) :
        topologie = hierarchy[0][rang]
        if topologie[0] == -1 and topologie[1] == -1 \
        and topologie[2] == -1 and topologie[3] != -1:
            rang_pere = topologie[3]
            pere = contours[rang_pere]
            topologie_pere = hierarchy[0][rang_pere]
            rang_grand_pere = topologie_pere[3]
            if rang_grand_pere != -1 :
                grand_pere = contours[rang_grand_pere] 
                cv2.drawContours(image, contour,-1,(0,0,2555),2)
                cv2.drawContours(image,grand_pere,-1,(0,255,0),2)
            
if __name__ == '__main__' :
    image = cv2.imread('img/plickerexample.jpg')
    cv2.imwrite('img/test2/capture.jpg', image)
    redessine_quadrilateres_et_peres(image)
    cv2.imshow('example',image)
    cv2.imwrite('img/test2/capture_contours.jpg', image)
    cv2.waitKey(0)
    
    cap = cv2.VideoCapture(0)
    while(True):
        retour, frame = cap.read()
        frame2 = frame.copy()
        ced_image = cv2.Canny(frame,100,180)
        redessine_quadrilateres_et_peres(frame)
        cv2.imshow('canny',ced_image)
        cv2.imshow('contours',frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            #cv2.imwrite('img/capture.jpg', frame2)
            #cv2.imwrite('img/capture_contours.jpg', frame)
            #cv2.imwrite('img/capture_canny.jpg', ced_image)
            break
    cap.release()
    cv2.destroyAllWindows()
