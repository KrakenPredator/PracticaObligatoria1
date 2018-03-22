import glob

import cv2
import numpy as np
import scipy

detector=cv2.ORB_create(nfeatures=100, nlevels=4, scaleFactor=1.3)

FLANN_INDEX_LSH = 6
index_params = dict(algorithm=FLANN_INDEX_LSH,
                    table_number=6,
                    key_size=3,
                    multi_probe_level=1)
search_params = dict(checks=-1)
points = []
factorEscala = 20
flann = cv2.FlannBasedMatcher(index_params, search_params)
descs = None
for file in glob.glob('training/*.jpg'):
    imagenEntrenamiento = cv2.imread(file, 0)
    trainKP, trainDesc = detector.detectAndCompute(imagenEntrenamiento, None)
    flann.add([trainDesc])
    flann.train()
    for i in range(len(trainKP)):
        kp = trainKP[i]
        points.append(trainKP[i])
    if descs is None:
        descs = trainDesc
    else:
        descs = np.concatenate((descs, trainDesc))


def maximo(acumulador):
    max = 0
    f = 0
    c = 0
    for fila in range(len(acumulador)):
        for columna in range(len(acumulador)):
            # Que el numero de votos sea maximo y que en su vecindad no haya un vecino sin votos
            if acumulador[fila][columna] > max:
                max = acumulador[fila][columna]
                f = fila
                c = columna
    print(max)
    return (f*factorEscala, c*factorEscala)  # (Fila,Columna)


def matchIndividual(img1):
    kp, des = detector.detectAndCompute(img1, None)
    knnmatches = flann.knnMatch(des, k=5)
    res = []
    filtro = 12.5
    for m, n, k, j, i in knnmatches:
        if m.distance < n.distance - filtro:
            res.append(kp[n.queryIdx])
        if m.distance < k.distance - filtro:
            res.append(kp[k.queryIdx])
        if m.distance < j.distance - filtro:
            res.append(kp[j.queryIdx])
        if m.distance < i.distance - filtro:
            res.append(kp[i.queryIdx])

    salida = 0
    salida = cv2.drawKeypoints(img1, [], salida, color=(0, 255, 0), flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
    acumulador = np.zeros((int(img1.shape[0]/factorEscala)+(factorEscala*2), int(img1.shape[1]/factorEscala)+(factorEscala*2)))  # Variable que acumulara los votos de los distintos keypoints

    centers = []
    for j in res:
        centers.append(calc_center(j,img1))
        acumulador[int(j.pt[0]/factorEscala)][int(j.pt[1]/factorEscala)] += 1

    cv2.circle(salida, maximo(acumulador), 25, (0, 0, 255), 2)

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



for file in glob.glob('testing/*.jpg'):
    name = "salidasOrb/salidaT"+file[9:]
    imgTest = cv2.imread(file, 0)
    imgOut = matchIndividual(imgTest)
    cv2.imwrite(name, imgOut)
