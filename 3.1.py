import numpy as np
import cv2

# Funcion que encontrara las coordenadas de la imagen con el maximo numero de votos y que indicara la localizacion mas probable del coche
# Ademas de considerar un numero de votos alto, se tiene en cuenta que la vecindad de dichas coordenadas tambien tengan un numero determinado
# de votos. Asi nos aseguramos de encontrar un  punto caracteristico con muchos fiables a su alrededor, y no uno excluido como puediera ser el
# de un retrovisor, que va a tener muchos votos por ser un elemento caracteristico del coche, pero no se rodea de otros elementos caracteristicos que
# nos permitan reconocer el frontal de un coche
def maximo(acumulador):
        max = 0
        f = 0
        c = 0
        for fila in xrange (len(acumulador)):
            for columna in xrange (len(acumulador)):
                # Que el numero de votos sea maximo y que en su vecindad no haya un vecino sin votos
                if(acumulador[fila][columna]>max):
                    if((acumulador[fila][columna+1]>5) and (acumulador[fila][columna-1]>5) and (acumulador[fila+1][columna]>5)
                    and (acumulador[fila-1][columna]>5)):

                        max = acumulador[fila][columna]
                        f = fila
                        c = columna

        return (f,c) #(Fila,Columna)

# Variable global que acumulara los keypoints y descriptores de todas las imagenes de entrenamiento
entrenamiento = []

#CARGA DE KEYPOINTS Y DESCRIPTORES DEL ENTRENAMIENTO EN "entrenamiento"
for i in xrange(1,49):
    #La variable 'pathDirectorioTraining' contiene el directorio donde se encuentran las imagenes de training
    #Para incluir rutas es conveniente elegir directorios sin espacios ni caracteres como el 'Space' o tildes
    pathDirectorioTraining = 'C:\\training\\'  # Ejemplo de directorio
    img1 = cv2.imread(pathDirectorioTraining+'frontal_'+str(i)+'.jpg',0) #Imagen de entrenamiento
    #Si se incluyeran imagenes con otro nombre habria que sustituir 'frontal_' por el nombre de la imagen

    sift = cv2.SIFT() # Inicializacion del detector de descriptores SIFT

    kp1, des1 = sift.detectAndCompute(img1,None)
    par = (kp1,des1) # Creamos la tupla (keypoints,descriptores) de cada img, y la insertamos en nuestro vector de entrenamiento

    entrenamiento.append(par)

#CON CADA IMG NO VISTA EN EL ENTRENAMIENTO (TESTING), sacamos SUS KEYPOINTS Y DESCRIPTORES
for j in xrange(1,34):
    #La variable 'pathDirectorioTesting' contiene el directorio donde se encuentran las imagenes de testing
    #Para incluir rutas es conveniente elegir directorios sin espacios ni caracteres como el 'Space' o tildes
    pathDirectorioTesting = 'C:\\testing\\'  # Ejemplo de directorio
    img2 = cv2.imread('test'+str(j)+'.jpg',0) #Imagen de Test
    acumulador = np.zeros((img2.shape[0],img2.shape[1])) #Variable que acumulara los votos de los distintos keypoints

    sift = cv2.SIFT()
    kp2, des2 = sift.detectAndCompute(img2,None) # Extraccion de los keypoints y descriptores de cada imagen de test

    # Usamos la estructura de datos tipo KD-Tree del objeto FLANN para
    # acelerar la busqueda de descriptores semejantes entre imagenes
    FLANN_INDEX_KDTREE = 0
    index_params = dict(algorithm = FLANN_INDEX_KDTREE, trees = 5)
    search_params = dict(checks = 50)
    flann = cv2.FlannBasedMatcher(index_params, search_params)

    #Proceso de votacion
    for k in xrange(len(entrenamiento)):# Recorremos el vector de entrenamiento con los descriptores y keypoints

        tuplaT = entrenamiento[k] # Recordemos que los tenemos almacenados en tuplas
        kpT = tuplaT[0]
        desT = tuplaT[1]

        # matches contiene los descriptores(vectores de keypoints) mas parecidos entre las dos imgs en objetos DMATCH
        matches = flann.knnMatch(desT,des2,k=6)

        for m in xrange (len(matches)): # Recorremos los matches, obteniendo las sublistas de los k descriptores semejantes
            sublista = matches[m]

            for n in xrange (len(sublista)): # Recorremos cada sublista con k descriptores para realizar la votacion de cada uno
               dmatch = sublista[n] # Objeto dmatch de la relacion
               indice = dmatch.queryIdx #Indice del keypoint del descriptor de la relacion
               key = kpT[indice] # Indexamos para obtener el keypoint y extraemos sus coordenadas x e y
               x = int(key.pt[0])
               y = int(key.pt[1])
               acumulador[y][x] = acumulador[y][x] + 1 # Sumamos uno a los votos de ese keypoint encontrado en un dmatch

    punto = maximo(acumulador) # Tras acabar la votacion para cada imagen de testing, ya tenemos el punto mas votado y  caracteristico de todos
    cv2.circle(img2,(punto[1],punto[0]), 25, (255,255,255), 1) # Dibujamos un circulo en su centro para localizar el punto de interes calculado
    cv2.imshow("Coche " +str(j),img2)
    cv2.waitKey()
    cv2.destroyAllWindows()