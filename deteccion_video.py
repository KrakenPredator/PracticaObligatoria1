import deteccion_haar
import cv2

cap = cv2.VideoCapture("Videos/video1.wmv")

while(cap.isOpened()):
    ret, frame = cap.read()
    img = deteccion_haar.haar(frame)
    cv2.imshow('frame', img)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
cap.release()
cv2.destroyAllWindows()