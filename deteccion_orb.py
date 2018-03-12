import glob
import cv2
import numpy as np
from matplotlib import pyplot as plt
from matplotlib import pylab

carga = 0
for file in glob.glob('training/*.jpg'):
    if carga < 4:
        image = cv2.imread(file, 0)
        orb = cv2.ORB_create(nfeatures=100, nlevels=4, scaleFactor=1.3)
        kp, des = orb.detectAndCompute(image, None)
        print(des)
        image = cv2.drawKeypoints(image, kp, None, color=(0,255,0), flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
        cv2.imwrite("image"+str(carga)+".jpg", image)
    carga += 1

'''aqui iria la carga de imagenes en bucle
buscar en internet cómo cargar muchas imágenes de una carpeta

luego usamos cv2.orb_create(3, 1) para coger los keypoints y descriptores primero 3 y 1 y luego 100, 4
para comprobar que funciona bien dibujamos en la imagen con drawKeypoints

guardamos el resultado de los descriptores en una estructura de datos
el indeice nos los da cv2.FlannBasedMatcher ó cv2.BFMatcher

tambien hay que crear un vector de votación que tiene el keypoint donde se encontraría el coche


tras el entrenamiento, procesamos una imagen con knnsearch (mirar esa parte en el pdf)'''
