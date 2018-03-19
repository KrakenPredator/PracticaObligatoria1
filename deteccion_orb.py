import glob
from collections import namedtuple

import cv2
import numpy as np


detector=cv2.ORB_create(nfeatures=1, nlevels=1)

FLANN_INDEX_TREE = 6
index_params = dict(algorithm=FLANN_INDEX_TREE,
                    table_number=5,
                    key_size=12,
                    multi_probe_level=1)

matcher = cv2.FlannBasedMatcher(index_params, {})
trainingData = []
points = []
descs = None
imagenEntrenamiento = None


for file in glob.glob('training/*.jpg'):
    imagenEntrenamiento = cv2.imread(file, 0)
    trainKP, trainDesc = detector.detectAndCompute(imagenEntrenamiento, None)
    trainingData.append((trainKP, trainDesc))
    for i in range(len(trainKP)):
        points.append(trainKP[i])
    if descs is None:
        descs = trainDesc
    else:
        descs = np.concatenate((descs, trainDesc))
    matcher.add([trainDesc])

imgPruebas = cv2.imread("testing/test15.jpg", 0)
acumulador = np.zeros(imgPruebas.shape[0], imgPruebas.shape[1])
testKp, testDesc = detector.detectAndCompute(imgPruebas, None)

for k in range(len(trainingData)):
    t = trainingData[k]
    keypt = t[0]
    desct = t[1]

    matches = matcher.knnMatch(desct, testDesc, k=2)

    for m in range(len(matches)):  # Recorremos los matches, obteniendo las sublistas de los k descriptores semejantes
        sublista = matches[m]

        for n in range(len(sublista)):  # Recorremos cada sublista con k descriptores para realizar la votacion de cada uno
            dmatch = sublista[n]  # Objeto dmatch de la relacion
            indice = dmatch.queryIdx  # Indice del keypoint del descriptor de la relacion
            key = keypt[indice]  # Indexamos para obtener el keypoint y extraemos sus coordenadas x e y
            x = int(key.pt[0])
            y = int(key.pt[1])
            acumulador[y][x] = acumulador[y][x] + 1  # Sumamos uno a los votos de ese keypoint encontrado en un dmatch
puntos = maximo(acumulador)
cv2.circle(img2,(punto[1],punto[0]), 25, (255,255,255), 1) # Dibujamos un circulo en su centro para localizar el punto de interes calculado
    cv2.imshow("Coche " +str(j),img2)


'''aqui iria la carga de imagenes en bucle
buscar en internet cómo cargar muchas imágenes de una carpeta

luego usamos cv2.orb_create(3, 1) para coger los keypoints y descriptores primero 3 y 1 y luego 100, 4
para comprobar que funciona bien dibujamos en la imagen con drawKeypoints

guardamos el resultado de los descriptores en una estructura de datos
el indeice nos los da cv2.FlannBasedMatcher ó cv2.BFMatcher

tambien hay que crear un vector de votación que tiene el keypoint donde se encontraría el coche


tras el entrenamiento, procesamos una imagen con knnsearch (mirar esa parte en el pdf)'''
