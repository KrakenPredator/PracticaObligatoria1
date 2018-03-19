import cv2

cochesCasacade = cv2.CascadeClassifier("haar/coches.xml")


def haar(img):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    detectados = cochesCasacade.detectMultiScale(img)
    for x in detectados:
        cv2.rectangle(img,(x[0],x[1]),(x[0]+x[2],x[1]+x[3]),(0,255,0),2)
    return img
