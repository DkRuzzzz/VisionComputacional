#!/usr/bin/env python3
#encoding: utf-8

import numpy as np
import cv2

# Definimos el ancho y alto en el que queremos almacenar los cuadros que leemos.
IM_WIDTH = 800
IM_HEIGHT = 600

cameraSource = False
cameraNumber = 0

if cameraSource == False:
    videoSource = "./video.mpg"
else:
    videoSource = cameraNumber
    
# Instanciamos el objecto camera, abriendo la fuente de video deseada.    
camera = cv2.VideoCapture(videoSource)


# En caso de que la apertura sea exitosa...
if camera.isOpened():
    # Obtenemos las dimensiones de los cuadros capturados por la cámara.
    frmWidth = camera.get(cv2.CAP_PROP_FRAME_WIDTH)
    frmHeight = camera.get(cv2.CAP_PROP_FRAME_HEIGHT)
    
    # Definimos un factor de escala para escalar las imágenes capturadas a la
    # resolución deseada, i.e. (IM_HEIGHT, IM_WIDTH)
    scaleX = IM_WIDTH  / frmWidth
    scaleY = IM_HEIGHT / frmHeight
    
    # Abrimos una ventana para mostrar el video que leamos.
    cv2.namedWindow('Video', cv2.WINDOW_NORMAL)

    while True:
        # Leemos un cuadro (frame) del flujo de video. 
        # Si etval != True, no se tuvo éxito al leer el cuadro.
        etval, frame = camera.read()
        if etval == True:
            # Si la escala horizontal o vertical son diferentes a 1
            # entonces reescala la imagen.
            if scaleX != 1. or scaleY != 1.:
                frame = cv2.resize(frame, None, fx=scaleX, fy=scaleY, interpolation=cv2.INTER_AREA)
                
            
        # Mostramos un cuadro del video. 
        cv2.imshow("Video", frame)
        
        # Esperamos 33 milisegundos a que se oprima una tecla.
        # esto nos asegura que el video se muestre a una velocidad "adecuada"
        # (adecuada solo si la fuente de video se capturó a una razon de 
        # 30 cuadros por segundo).
        val = cv2.waitKey(33)
        
        # En caso que el valor que regresa waitKey sea diferente a -1, aborta el ciclo.
        # Esto porque waitKey regresa -1 si pasan los 33 msec sin que se haya apretado una
        # tecla.
        if val != -1:
            break
      
    cv2.destroyWindow("Video")
    
    # Liberamos la fuente de Video.
    camera.release()
