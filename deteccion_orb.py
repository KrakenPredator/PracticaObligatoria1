import glob
import cv2
import numpy as np
from matplotlib import pyplot as plt
from matplotlib import pylab

indices = dict(algorithm = 6,
               table_number = 6,
               key_size = 12,
               multi_probe_level = 1)
busqueda = dict(checks = 50)

flann = cv2.FlannBasedMatcher(indices, busqueda)
orb = cv2.ORB_create(nfeatures=100, nlevels=4, scaleFactor=1.3)

for file in glob.glob('training/*.jpg'):
    image = cv2.imread(file, 0)
    kp, des = orb.detectAndCompute(image, None)
    flann.add([des])
    print(len(flann.getTrainDescriptors()))

flann.train()

imgPrueba = cv2.imread("testing/test11.jpg", 0)
kp, des = orb.detectAndCompute(imgPrueba, None)
matches = flann.knnMatch(des, k=2)
img3 = cv2.drawMatches(imgPrueba, kp, None, None, matches)

cv2.imwrite("fuera.jpg", img3)

'''aqui iria la carga de imagenes en bucle
buscar en internet cómo cargar muchas imágenes de una carpeta

luego usamos cv2.orb_create(3, 1) para coger los keypoints y descriptores primero 3 y 1 y luego 100, 4
para comprobar que funciona bien dibujamos en la imagen con drawKeypoints

guardamos el resultado de los descriptores en una estructura de datos
el indeice nos los da cv2.FlannBasedMatcher ó cv2.BFMatcher

tambien hay que crear un vector de votación que tiene el keypoint donde se encontraría el coche


tras el entrenamiento, procesamos una imagen con knnsearch (mirar esa parte en el pdf)'''
