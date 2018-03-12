import cv2

cochesCasacade = cv2.CascadeClassifier("haar/coches.xml")


def haar(img):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    detectados = cochesCasacade.detectMultiScale(gray)
    for (x,y,w,h) in detectados:
        cv2.rectangle(img,(x,y),(x+w,y+h),(0,255,0),2)
    return img
