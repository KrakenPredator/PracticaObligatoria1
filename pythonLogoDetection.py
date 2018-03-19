import glob

import cv2
import numpy as np
import itertools
import sys

detector=cv2.ORB_create(nfeatures=10, nlevels=4, scaleFactor=1.3)

FLANN_INDEX_TREE = 6
index_params = dict(algorithm=FLANN_INDEX_TREE,
                    table_number=5,
                    key_size=12,
                    multi_probe_level=1)

matcher = cv2.FlannBasedMatcher(index_params, {})
points = []
descriptoresGuardaos = []
descs = None
indxKpV = []
acumulador = []
'def vector(p1, p2):'

imagenEntrenamiento = None
for file in glob.glob('training/*.jpg'):
    imagenEntrenamiento = cv2.imread(file, 0)
    trainKP, trainDesc = detector.detectAndCompute(imagenEntrenamiento, None)
    descriptoresGuardaos.append(trainDesc)
    for i in range(len(trainKP)):
        kp = trainKP[i]
        points.append(trainKP[i])
        indxKpV.append(((imagenEntrenamiento.shape[0]/2)-kp.pt[0], (imagenEntrenamiento.shape[1]/2)-kp.pt[1]))
    if descs is None:
        descs = trainDesc
    else:
        descs = np.concatenate((descs, trainDesc))


def orb(image):
    votacion = np.zeros((image.shape[0], image.shape[1]))
    flann = cv2.flann_Index(descs, index_params)
    imgPrueba = image
    kp, des = detector.detectAndCompute(imgPrueba, None)
    idx, dist = flann.knnSearch(des, knn=1)
    keypnts = []
    puntos = []

    for i in range(len(idx)):
        puntos.append(idx[i])

    x, y = idx.shape
    for i in range(x):
        ind = idx[i]
        for j in range(y):
            vector = indxKpV[ind[j]]
            print(vector)
    output = imgPrueba.copy()
    output = cv2.drawKeypoints(imgPrueba, points, output, (0, 255, 0), cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
    return output

imgPrueba = cv2.imread("testing/test17.jpg", 0)
imOut = orb(imgPrueba)
cv2.imwrite("outpur.jpg", imOut)

'''cap = cv2.VideoCapture("Videos/video1.wmv")

while(cap.isOpened()):
    ret, frame = cap.read()
    img = orb(frame)

    cv2.imshow('frame', img)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
cap.release()
cv2.destroyAllWindows()'''