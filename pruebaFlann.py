import glob

import cv2
import numpy as np
import scipy
import itertools
import sys

detector=cv2.ORB_create(nfeatures=100, nlevels=4, scaleFactor=1.3)

FLANN_INDEX_LSH = 6
index_params = dict(algorithm=FLANN_INDEX_LSH,
                    table_number=6,
                    key_size=3,
                    multi_probe_level=1)
search_params = dict(checks=-1)
points = []
flann = cv2.FlannBasedMatcher(index_params, search_params)
descs = None
for file in glob.glob('training/*.jpg'):
    imagenEntrenamiento = cv2.imread(file, 0)
    trainKP, trainDesc = detector.detectAndCompute(imagenEntrenamiento, None)
    flann.add([trainDesc])
    for i in range(len(trainKP)):
        kp = trainKP[i]
        points.append(trainKP[i])
    if descs is None:
        descs = trainDesc
    else:
        descs = np.concatenate((descs, trainDesc))

def matchIndividual(img1):
    kp, des = detector.detectAndCompute(img1, None)
    knnmatches = flann.knnMatch(des, k=5)
    salida = 0
    salida = cv2.drawKeypoints(img1, kp, salida, (0, 255, 0), cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
    res = []
    '''''
    for m, n in knnmatches:
        if m.distance < n.distance - 4:
            res.append(kp[m.queryIdx])

    '''''

    filtro = 13

    for m, n, k, j, i in knnmatches:
        if m.distance < n.distance - filtro:
            res.append(kp[m.queryIdx])
        if n.distance < k.distance - filtro:
            res.append(kp[n.queryIdx])
        if k.distance < j.distance - filtro:
            res.append(kp[k.queryIdx])
        if j.distance < i.distance - filtro:
            res.append(kp[j.queryIdx])

    print(res)
    salida = cv2.drawKeypoints(img1, res, salida, (0, 255, 0), cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)


    centers = []
    k = 0
    for j in range(len(res)):
        centers.append(calc_center(j,imgPrueba))
        print("Center",centers.__getitem__(k))
        k = k+1

    return salida


def orb(image):
    imgPrueba = image
    kp, des = detector.detectAndCompute(imgPrueba, None)
    output = imgPrueba.copy()
    imgPrueba = cv2.drawKeypoints(image, kp, imgPrueba, (0, 255, 0), cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)

    knnmatches = flann.knnMatch(des, k=5)
    encontrados = []
    puntetes = []
    res = []
    for m, n in knnmatches:
        print(n.distance)
        print(m.distance)
        if m.distance < 0.7 * n.distance:
            res.append(m)
    for i in range(len(res)):
        match = res[i]
        indice = match.trainIdx
        imgIdx = match.imgIdx
        encontrados.append(descs[indice])
        puntetes.append(points[indice])

    output = cv2.drawKeypoints(image, puntetes, output, (0, 255, 0), cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
    salida = 0
    salida = cv2.drawMatches(imgPrueba, kp, output, points, res, flags=2, outImg=salida)


    return salida

#Codigo obtenido de https://github.com/rainer85ah/CV/blob/master/Project3/Main.py
def calc_center(kp, img):
    x_key_point, y_key_point = kp.pt[:2]
    height, weight = img.shape[:2]
    x_center = weight / 2
    y_center = height / 2
    center_img = (x_center, y_center)

    xVector = x_center - x_key_point
    yVector = y_center - y_key_point
    vector = (xVector, yVector)

    module = scipy.sqrt(scipy.power((x_center - x_key_point), 2) + scipy.power((y_center - y_key_point), 2))

    if (y_center - y_key_point) == 0:
        angle = 0
    else:
        angle = scipy.arctan((x_center - x_key_point) / (y_center - y_key_point))

    distance_center = (module, vector, angle, center_img)
    return distance_center





imgPrueba = cv2.imread("testing/test16.jpg", 0)
imgdos = cv2.imread("training/frontal_27.jpg", 0)

imOut = matchIndividual(imgPrueba)
cv2.imwrite("outpur.jpg", imOut)
