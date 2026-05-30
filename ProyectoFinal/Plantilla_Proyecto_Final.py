#! /usr/bin/env python3
#encoding: utf-8


import numpy as np
import numpy.linalg as lin
import cv2 
import sys
from rigidBody import *


def isInteger (s):
   """
   Determina si el parámetro s es un entero.
   """
   try:
      int(s)
      return True
   except ValueError:
      return False

def normPH(P):
   """
   Esta función normaliza una matriz que contiene 
   coordenadas homogeneas.
   """
   r, c = P.shape
   if r > 2:
      lr = r - 1
      for i in range(c):
         P[:,i] /= P[lr, i]

def drawAxes(img, K, R, T, fact=1.):
   """
   Esta funcion dibuja en la matriz de imagen img un marco de referencia con
   pose definida por la matriz de rotación R y el vector de Translación T.
   El parámetro K corresponde a la matriz de parámetros intrínsecos de la
   cámara, mientras que el parámetro fact es un factor de escala que determina
   el támaño de los ejes.
   """
   ax3 = np.array([[0, fact,    0,    0],\
                  [ 0,    0, fact,    0],\
                  [ 0,    0,    0, fact],\
                  [ 1,    1,    1,    1]], dtype='double')
   ax3T = np.concatenate([R,T], axis=1) @ ax3
   ax2 = K @ ax3T

   normPH(ax2)

   C = (int(ax2[0, 0]), int(ax2[1, 0]))
   X = (int(ax2[0, 1]), int(ax2[1, 1]))
   Y = (int(ax2[0, 2]), int(ax2[1, 2]))
   Z = (int(ax2[0, 3]), int(ax2[1, 3]))
      
   cv2.line(img, (C[0],C[1]), (X[0], X[1]), (0, 0, 255), 5)
   cv2.line(img, (C[0],C[1]), (Y[0], Y[1]), (0, 255, 0), 5)
   cv2.line(img, (C[0],C[1]), (Z[0], Z[1]), (255, 0, 0), 5)

def recoverPose(iK, chBCoors, imgCoors):
   """
   This function computes the camera pose defined by a rotation matrix R, and
   a translation vector T, from correspondences between 3D coordinates of points
   laying in a flat surface (chBCoors), and their 2D proyection in the image plane.

   This function assumes that the reference frame of the 3D points is defined in a way
   where the X-Y plane lays in the flat surface plane, and the Z-axis is ortonormal to that
   plane. This implies that the Z-component of the 3D coordinates will be alway be zeros
   and only the X and Y coordinates of the 3D points are needed.

   Parameters:

   iK      : the inverse of the calibration matrix that defines the intrinsic
             parameters of the camera used to obtain the coordinate imgCoors.
   chBCoors: a 3xN array. Each column of this array contains one 3D Coordinate.
   imgCoors: a 2xN array. each column of this array contains one 2D coordinate.

   Important: N >= 4.

   The function returns the rotation matrix R and the translation vector T that
   define the relative pose of the flat surface reference frame with respecto to
   the camera.

   Note: chBCoors could also be a 4xN array that corresponds to homogenous 3D coordinates
   or a 2xN array that contains only the componentes (X,Y) of a 3D coordinate.
   """

   rI, cI = imgCoors.shape
   rC, cC = chBCoors.shape 

   if cI != cC or cI < 4 or cC < 4:
      return None, None
   
   # Convert the image coordinates to canonica coordinates that corresponds to a camera
   # whose calibration matrix K = np.eye(3)
   imgCoorsC = iK @ np.vstack([imgCoors,np.ones(cI)])
   normPH(imgCoorsC)

   if rC == 4:
      normPH(chBCoors)


   # Compute the homography that relates the 3D coordinates of points that lay on a space plane
   # and their proyection in tha camera's plane.
   H, _ = cv2.findHomography(chBCoors[:2,:].T, imgCoorsC.T)

   #Normalize H so that the 1st column norm is equal to 1
   factH = 1/lin.norm(H[:,0])
   H  = factH  *  H

   #Recover the translation vector
   T = H[:,2].reshape(3,1)

   # Use gram-Schmidt algoritm to make sure the first two columna are
   # ortonormal
   v1 = H[:,0]
   v2 = H[:,1]

   v1, v2 = ortGramSchmidt(H[:,0], H[:,1])
   v2 = v2 / lin.norm(v2)

   #Recover the third column of the rotation matrix.
   v3 = np.cross(v1,v2)

   #Compose the Rotation Matrix from the vector v1, v2, v3
   R = np.hstack([v1, v2, v3]).reshape(3,3)

   # Note: R and T are defined with respecto the 3D plane reference frame.
   # we return the inverse of R (i.e. its transpose) so that it corresponds
   # to the camera reference frame.
   return R.T, T

################################################################################################
################################################################################################
################################################################################################
###                                                                                          ###
###                                    MAIN PROGRAM                                          ###
###                                                                                          ###
################################################################################################
################################################################################################
################################################################################################


argc = len(sys.argv)

# Abre el flujo en donde vienen 
if argc < 2:
   print("\nEs necesario pasa un parámetro.")
   print("Si el parámetro es un número se leeran imagenes de una cámara del sistema.")
   print("Si no lo es, el parámetro deberá ser una ruta válida a un archivo de video.\n")
   exit(1)
else:
   videoDevice = sys.argv[1]

if isInteger(videoDevice):
   camera = cv2.VideoCapture(int(videoDevice))
   if len(sys.argv) > 3:
      col = int(sys.argv[2]);
      ren = int(sys.argv[3]);
      cv2.VideoCapture.set(camera, cv2.CAP_PROP_FRAME_WIDTH, col)
      cv2.VideoCapture.set(camera, cv2.CAP_PROP_FRAME_HEIGHT, ren)
else:   
   camera = cv2.VideoCapture(videoDevice)

"""
 Definimos la matriz de calibración K

 Dado que por convención las coordenadas en la imagen tienen una
 reflexión en el eje vertical (y=-y), describimos la matriz de calibración
 convencional de la siguiente manera:
    
       [ 1  0  0] [ fx  0  0 ] [ 1  0  Cx ]   [ fx   0   Cx ]
   K = [ 0 -1  0] [  0 fy  0 ] [ 0  1 -Cy ] = [  0 -fy   Cy ]
       [ 0  0  1] [  0  0  1 ] [ 0  0   0 ]   [  0   0    1 ]

 Donde el tercer termino representa una transformación que traslada las
 coordenadas del origen de la imagen al centro óptico de la cámara, el
 segundo termino corresponde a la conversión a pixeles que hacemos de las
 unidades en que está definida las coordenadas 3D de la escena y el primer
 término corresponde a la reflexión que debemos hacer del eje y para
 tener un sistema de mano derecha.

"""
K = np.array([7.7318146334666767e+02, 0.,                     4.0726293453767408e+02,\
              0.,                    -7.7318146334666767e+02, 3.0623163696686174e+02,\
              0.,                     0.,                                         1.]).reshape((3,3))

iK = lin.inv(K)

# side es el tamaño de un lado de un cuadro del tablero de Ajeders medido en metros.
# En este caso, el tamaño es de .
# side2 es la mitad del tamaño.
side = 2.4069999694824219e-02 # 24.06 mm
side2 = side/2

# Number of chessBoard inner corners
chBrdRows = 6
chBrdCols = 8
nCorners = chBrdRows * chBrdCols

# Esta matriz de 4 x nCorners almacena las coordenadas homogeneas 3D en metros de
# las esquinas del tablero de ajedrez con respecto a un marco de referencia
# centrado en el marcador. Nótese que los ejes X y Y del marco de referencia
# son paralelos a los lados del marcador, y que el plano XY yace exactamente
# sobre el plano que pasa por el marcador, por eso los componentes Z de las
# coordenadas son iguales a 0. 

x = np.linspace(0, (chBrdCols-1) * side, chBrdCols) - (chBrdCols-1) * side / 2
y = np.linspace(0, (chBrdRows-1) * side, chBrdRows) - (chBrdRows-1) * side / 2
X, Y = np.meshgrid(x,y)
P3 = np.vstack([X.flatten(), Y.flatten(), np.zeros(nCorners), np.ones(nCorners)])

# Esta matriz se construye eliminando el tercer renglon de P3
# Esto es, son coordenadas homogeneas 2D en metros de las esquinas del marcador.
P2 = np.vstack([X.flatten(), Y.flatten(), np.ones(nCorners)])

# El numero de frames por segundo en que fueron capturadas las imágenes
FPS=30
DELAY=1000//FPS
dt =1./FPS

if camera.isOpened():
   cv2.namedWindow('chessBoard')
   cont = 0
   R1=np.eye(3);
   T1=np.zeros((3,1))
   cont = 0
   changeFPS = False
   while True:
      etval, img = camera.read()
      if etval == True:   
         retval, Esquinas = cv2.findChessboardCorners(img, (8, 6))
         if retval == True:
            Esquinas = Esquinas[:,0,:]
            for i in range(nCorners):
               cv2.circle(img, np.int32(Esquinas[i,:]), 5, (160,200,248),2)

            if retval == True: #Si se encontraron las esquinas...

               R1, T1 = recoverPose(iK, P3, Esquinas.T)

               print ("R1_%03d = \n"%cont, R1,"\n")
               print ("T1_%03d = "%cont, T1.T,"\n")
               print ("||T1||",lin.norm(T1),"\n\n")


################################################################################################
################################################################################################
################################################################################################
###                                                                                          ###
###                                 Aquí va su código                                        ###
###                                                                                          ###
################################################################################################
################################################################################################
################################################################################################

               drawAxes(img, K, R1, T1, 0.1)
               
               #if cont > 0:
                  # Estimamos la velocidad de rotación de la cámara omega
                  # Véase sección 2.3.2 de "An Invitation of 3D Vision",
                  # formula 2.10, pp..26.
                  

                  # Construimos la matriz xi que corresponde a la derivada de G (G ∈ SE(3)).
                  # Véase sección 2.4.2 de "An Invitation of 3D Vision",
                  # fórmula 2.20, pp. 32.
                  

                  # Estimamos donde podrían estar las equinas del tablero en la siguiente imagen.
                  # Véase en la sección 2.4.2 de "An Invitation of 3D Vision",
                  # el desarrollo posterior a la fórmula 2.20, pp. 32.
                  

                  # Proyectamos los puntos estimados en el plano de la imagen
                  

                  # Graficamos la región donde esperamos se encuentre el tablero de ajedrez.
                  # en la siguiente imagen, así como flechas indicando la velocidad
                  # en 2D de las esquinas del marcadores.
                  
            R0 = R1
            T0 = T1
            cont += 1

         cv2.imshow("chessBoard", img)
         key = cv2.waitKeyEx(DELAY) & 0x000000FF
         if key == 27:
            break
         elif chr(key) == 'a' and FPS > 1:
            FPS -= 1
            changeFPS = True
         elif chr(key) == 'd':
            FPS += 1
            changeFPS = True
         elif chr(key) == 'q':
            if FPS > 10:
                  FPS -= 10
            else:
                  FPS = 1
            changeFPS = True
         elif chr(key) == 'e':
            FPS += 10
            changeFPS = True

         if changeFPS == True:
            print ("FPS = ", FPS)
            DELAY=1000//FPS
            dt =1./FPS
            changeFPS = False
      else:
         break   

   camera.release()
   cv2.destroyAllWindows()

