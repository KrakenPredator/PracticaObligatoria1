import glob

import cv2
import numpy as np
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
    for m, n, k, j, i in knnmatches:
        if m.distance < n.distance - 10:
            res.append(kp[m.queryIdx])
        if n.distance < k.distance - 10:
            res.append(kp[n.queryIdx])
        if k.distance < j.distance - 10:
            res.append(kp[k.queryIdx])
        if j.distance < i.distance - 10:
            res.append(kp[j.queryIdx])
    print(res)
    salida = cv2.drawKeypoints(img1, res, salida, (0, 255, 0), cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
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


imgPrueba = cv2.imread("testing/test5.jpg", 0)
imgdos = cv2.imread("training/frontal_27.jpg", 0)

imOut = matchIndividual(imgPrueba)
cv2.imwrite("outpur.jpg", imOut)
