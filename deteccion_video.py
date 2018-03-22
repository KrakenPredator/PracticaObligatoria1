import deteccion_haar
import cv2

import pruebaFlann

cap = cv2.VideoCapture("Videos/video2.wmv")

while(cap.isOpened()):
    ret, frame = cap.read()
    img = pruebaFlann.matchIndividual(frame)
    cv2.imshow('frame', img)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
cap.release()
cv2.destroyAllWindows()