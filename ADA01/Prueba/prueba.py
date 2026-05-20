import numpy as np
import cv2
import matplotlib.pyplot as plt

I01 = cv2.imread('Pelota.png', cv2.IMREAD_ANYDEPTH)
I02 = cv2.imread('Gatito.png', cv2.IMREAD_COLOR)

cv2.namedWindow('Imagen01', cv2.WINDOW_NORMAL)
cv2.namedWindow('Imagen02', cv2.WINDOW_NORMAL)

cv2.imshow('Imagen01', I01)
cv2.imshow('Imagen02', I02)


cv2.waitKey(0)
cv2.destroyAllWindows()
