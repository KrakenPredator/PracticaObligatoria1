import glob
from collections import namedtuple

import cv2
import numpy as np
from matplotlib import pyplot as plt
from matplotlib import pylab

PlanarTarget = namedtuple('Objetivo','image, rect, keypoints, descrs, data')
TrackedTarget = namedtuple('TrackedTarget', 'target, p0, p1, H, quad')

indices = dict(algorithm = 6,
               table_number = 6,
               key_size = 12,
               multi_probe_level = 1)

flann = cv2.FlannBasedMatcher(indices, {})
orb = cv2.ORB_create(nfeatures=100, nlevels=4, scaleFactor=1.3)
objetivos = []

def get_kpydesc(imagen):
    kp, des = orb.detectAndCompute(imagen, None)
    if des is None:
        des = []
    return kp, des


def entrenar(imagen, rect):
    kp, des = get_kpydesc(imagen)
    des = np.uint8(des)
    flann.add([des])
    objeto = PlanarTarget(image=imagen, rect=rect, keypoints=kp, descrs=des, data=None)
    objetivos.append(objeto)


for file in glob.glob('training/*.jpg'):
    image = cv2.imread(file, 0)
    entrenar(image, (0, 0, 420, 220))
print(objetivos)

imgPrueba = cv2.imread("testing/test19.jpg", 0)
kp, des = get_kpydesc(imgPrueba)
matches = flann.knnMatch(des, k=2)
matches = [m[0] for m in matches if len(m) == 2 and m[0].distance < m[1].distance * 0.75]
matches_by_id = [[] for _ in range(len(objetivos))]
for m in matches:
    matches_by_id[m.imgIdx].append(m)
tracked = []
for imgIdx, matches in enumerate(matches_by_id):
    if len(matches) < 10:
        continue
    target = objetivos[imgIdx]
    p0 = [target.keypoints[m.trainIdx].pt for m in matches]
    p1 = [kp[m.queryIdx].pt for m in matches]
    p0, p1 = np.float32((p0, p1))
    H, status = cv2.findHomography(p0, p1, cv2.RANSAC, 3.0)
    status = status.ravel() != 0
    if status.sum() < 10:
        continue
    p0, p1 = p0[status], p1[status]

    x0, y0, x1, y1 = target.rect
    quad = np.float32([[x0, y0], [x1, y0], [x1, y1], [x0, y1]])
    quad = cv2.perspectiveTransform(quad.reshape(1, -1, 2), H).reshape(-1, 2)

    track = TrackedTarget(target=target, p0=p0, p1=p1, H=H, quad=quad)
    tracked.append(track)
tracked.sort(key=lambda t: len(t.p0), reverse=True)

img3 = imgPrueba.copy()
for tr in tracked:
    cv2.polylines(img3, [np.int32(tr.quad)], True, (0, 255, 0), 2)
    for (x, y) in np.int32(tr.p1):
        cv2.circle(img3, (x, y), 2, (0, 255, 0))


return img3

'''aqui iria la carga de imagenes en bucle
buscar en internet cómo cargar muchas imágenes de una carpeta

luego usamos cv2.orb_create(3, 1) para coger los keypoints y descriptores primero 3 y 1 y luego 100, 4
para comprobar que funciona bien dibujamos en la imagen con drawKeypoints

guardamos el resultado de los descriptores en una estructura de datos
el indeice nos los da cv2.FlannBasedMatcher ó cv2.BFMatcher

tambien hay que crear un vector de votación que tiene el keypoint donde se encontraría el coche


tras el entrenamiento, procesamos una imagen con knnsearch (mirar esa parte en el pdf)'''
