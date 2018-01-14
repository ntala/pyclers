import cv2

def est_quadrilatere(contour):
    peri = cv2.arcLength(contour, True)
    approx = cv2.approxPolyDP(contour, 0.02 * peri, True)
    res = len(approx) == 4
    return res
    
cap = cv2.VideoCapture(1)                
while(True):
    ret, frame = cap.read()
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    mystere, threshold = cv2.threshold(gray,150,255,cv2.THRESH_BINARY)
    contours, hierarchy = cv2.findContours(threshold, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)
    for contour in contours : 
        if est_quadrilatere(contour) :
            cv2.drawContours(frame, contour,-1,(0,0,2555),2)
    cv2.imshow('frame',frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
cap.release()
cv2.destroyAllWindows()
