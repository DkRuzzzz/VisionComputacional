#! /usr/bin/env python3
#encoding: utf-8

import numpy as np
import cv2 
import sys
from myAxes import Axes
from trackbar import trackbar


class Square:
   def __init__(self, axes, side=10):
      side /= 2
      self.P = np.array([[-side, side,side,-side],\
                         [-side,-side,side, side],\
                         [   1,    1,   1,    1 ]])
      self.originalP = self.P.copy()
      self.Axes = axes
      self.sReg = np.zeros((4,2),np.int32)
      self.Cx, self.Cy = self.Axes.midX, self.Axes.midY
      self.Scale=np.array([[self.Axes.ticSize, 0, self.Axes.midX], [0, -self.Axes.ticSize, self.Axes.midY], [0, 0, 1]])

   def applyHomography(self,H):
      self.P = H @ self.originalP

   def reset(self):
      self.P = self.originalP.copy()
      

   def drawSquare(self, Img, colorSq=(0, 255, 128), colorCrns=(0,255,255)):
      for i in range(4):
         p = self.Scale @ self.P[:,i]
         p = p / p[2]
         self.sReg[i,:] = [int(np.round(x)) for x in [p[0], p[1]]]
         cv2.circle(Img, (self.sReg[i, 0], self.sReg[i,1]), 3, colorCrns, 2)
      cv2.polylines(Img, [self.sReg], True, colorSq)

class homography:
   def __init__(self, w, h):
      self.width  = w
      self.height = h
      self.change = False
      
      self.tx = self.ty = 0.
      self.theta = self.phi = 0.
      self.l1 = self.l2 = 1.
      self.scale = 1.
      self.v1 = self.v2 = 0.
      self.Rpp = np.eye(2)
      self.Rmp = np.eye(2)
      self.Rth = np.eye(2)
      self.L = np.eye(2)
      self.H = np.eye(3)
      self.hWidth = 600
      self.hHeight = 320
      self.winName = "Homografia"
      self.hDisplay = np.zeros((self.hHeight, self.hWidth, 3), dtype=np.uint8)

      cv2.namedWindow(self.winName, cv2.WINDOW_AUTOSIZE | cv2.WINDOW_KEEPRATIO | cv2.WINDOW_GUI_EXPANDED )
      self.drawHomographyMatrix()
      cv2.waitKey(1)

      self.txTB = trackbar("Tx", self.winName, -(self.width//4), self.width//4,  0, grain=self.width)
      self.tyTB = trackbar("Ty", self.winName, -(self.height//4), self.height//4, 0, grain=self.height)
      self.l1TB = trackbar("L1", self.winName, 1/100, 3, 1, grain=100)
      self.l2TB = trackbar("L2", self.winName,1/100, 3, 1, grain=100)
      #self.sTB  = trackbar("Scale", self.winName, -1, 1, 0, grain=60)
      self.thetaTB = trackbar("Theta", self.winName, -np.pi, np.pi, 0, grain=360)
      self.phiTB = trackbar("Phi", self.winName, -np.pi, np.pi, 0, grain=360)
      self.v1TB = trackbar("V1", self.winName, -0.025, 0.025, 0, grain=200)
      self.v2TB = trackbar("V2", self.winName, -0.025, 0.025, 0, grain=200)
      self.assignCallBacks()
      self.launchTrackbars()

   def drawBracket(self, x, y, w, h, lw):
      bracket = [[w + x, -0.5 * h + y],\
                 [    x, -0.5 * h + y],\
                 [    x,  0.5 * h + y],\
                 [w + x,  0.5 * h + y]]
      for i in range(3):
         pa = [int(np.round(x)) for x in bracket[i][:]]
         pb = [int(np.round(x)) for x in bracket[i + 1][:]]
         cv2.line(self.hDisplay, pa, pb, (255, 255, 255), lw)

   def drawHomographyMatrix(self):
      self.hDisplay = np.zeros((self.hHeight, self.hWidth, 3), dtype=np.uint8)
      
      cv2.putText(self.hDisplay, "H =", (10,90), cv2.FONT_HERSHEY_COMPLEX, 1, (255,255,255), 2, cv2.LINE_AA, False)
      for i in range(3):
         s = "%+ 10.6f, %+ 10.6f, %+ 10.6f" % (self.H[i,0], self.H[i,1], self.H[i,2])
         cv2.putText(self.hDisplay, s, (90,60+30*i), cv2.FONT_HERSHEY_COMPLEX, 0.75, (255,255,255), 1, cv2.LINE_AA, False)
      self.drawBracket( 90, 85, 20, 110, 3)
      self.drawBracket( 580, 85, -20, 110, 3)

      cv2.line(self.hDisplay, (10,180), (self.hWidth - 10, 180), (192, 192, 192), 3)
      
      cv2.putText(self.hDisplay, "Tx = %+ 5.2f, L1 = %+ 5.2f, theta = %+ 5.2f" % (self.tx, self.l1, self.theta), (20, 220), cv2.FONT_HERSHEY_COMPLEX, 0.75, (255,255,255), 1, cv2.LINE_AA, False)
      cv2.putText(self.hDisplay, "Ty = %+ 5.2f, L2 = %+ 5.2f, phi   = %+ 5.2f" % (self.ty, self.l2, self.phi), (20, 250), cv2.FONT_HERSHEY_COMPLEX, 0.75, (255,255,255), 1, cv2.LINE_AA, False)
      cv2.putText(self.hDisplay, "v1 = %+ 8.5f, v2 = %+ 8.5f" % (self.v1, self.v2), (20, 280), cv2.FONT_HERSHEY_COMPLEX, 0.75, (255,255,255), 1, cv2.LINE_AA, False)
      

      cv2.imshow(self.winName, self.hDisplay)
      
   def reset(self):
      self.H = np.eye(3)
      self.tx = self.ty = 1.
      self.v1 = self.v2 = 0.
      self.theta = self.phi = 0.
      self.l1 = self.l2 = 1.
      self.scale = 1.
      self.txTB.reset()
      self.tyTB.reset()
      self.l1TB.reset()
      self.l2TB.reset()
      #self.sTB.reset()
      self.thetaTB.reset()
      self.phiTB.reset()
      self.v1TB.reset()
      self.v2TB.reset()
      self.drawHomographyMatrix()
      self.change = True
      

   def Rot(self, alpha):
      c=np.cos(alpha)
      s=np.sin(alpha)
      return np.array([[c, -s],[s,c]])

   def computeA(self):
      return self.Rth @ self.Rmp @ self.L @ self.Rpp 

   def onTx(self, pos):
      self.txTB.position = pos
      self.txTB.pos2Val(pos)
      self.tx = self.txTB.value
      self.H[0,2] = self.tx
      self.drawHomographyMatrix()
      self.change = True

   def onTy(self, pos):
      self.tyTB.position = pos
      self.tyTB.pos2Val(pos)
      self.ty = self.tyTB.value
      self.H[1,2] = self.ty
      self.drawHomographyMatrix()
      self.change = True

   def onL1(self, pos):
      self.l1TB.position = pos
      self.l1TB.pos2Val(pos)
      self.l1 = self.l1TB.value
      self.L[0,0] = self.l1
      self.H[:2,:2] = self.computeA()
      self.drawHomographyMatrix()
      self.change = True

   def onL2(self, pos):
      self.l2TB.position = pos
      self.l2TB.pos2Val(pos)
      self.l2 = self.l2TB.value
      self.L[1,1] = self.l2
      self.H[:2,:2] = self.computeA()
      self.drawHomographyMatrix()
      self.change = True
   
   def onTheta(self, pos):
      self.thetaTB.position = pos
      self.thetaTB.pos2Val(pos)
      self.theta = self.thetaTB.value
      self.Rth = self.Rot(self.theta)
      self.H[:2,:2] = self.computeA()
      self.drawHomographyMatrix()
      self.change = True

   def onPhi(self, pos):
      self.phiTB.position = pos
      self.phiTB.pos2Val(pos)
      self.phi = self.phiTB.value
      self.Rpp = self.Rot(self.phi)
      self.Rmp = self.Rot(-self.phi)
      self.H[:2,:2] = self.computeA()
      self.drawHomographyMatrix()
      self.change = True

   def onV1(self, pos):
      self.v1TB.position = pos
      self.v1TB.pos2Val(pos)
      self.v1 = self.v1TB.value
      self.H[2,0] = self.v1
      self.drawHomographyMatrix()
      self.change = True

   def onV2(self, pos):
      self.v2TB.position = pos
      self.v2TB.pos2Val(pos)
      self.v2 = self.v2TB.value
      self.drawHomographyMatrix()
      self.H[2,1] = self.v2
      self.change = True

   '''
   def onS(self, pos):
      bkup = [self.sTB.position, self.sTB.value]
      self.sTB.position = pos
      self.sTB.pos2Val(pos)
      newL1Value = self.l1 * self.sTB.value
      newL2Value = self.l2 * self.sTB.value
      if not self.l1TB.withinBounds(newL1Value) or not self.l1TB.withinBounds(newL2Value):
         self.sTB.position = bkup[0]
         self.sTB.value = bkup[1]
         return
      self.scale = self.sTB.value
      self.l1TB.val2Pos(newL1Value)
      self.l2TB.val2Pos(newL2Value)
      self.onL1(self.l1TB.position)
      self.onL2(self.l2TB.position)
      cv2.setTrackbarPos(self.l1TB.barName, self.l1TB.winName, self.l1TB.position)
      cv2.setTrackbarPos(self.l2TB.barName, self.l2TB.winName, self.l2TB.position)
      self.H[:2,:2] = self.computeA()
      self.drawHomographyMatrix()
      self.change = True
'''

   def assignCallBacks(self):
      self.txTB.onTrackbar = self.onTx
      self.tyTB.onTrackbar = self.onTy
      self.l1TB.onTrackbar = self.onL1
      self.l2TB.onTrackbar = self.onL2
      #self.sTB.onTrackbar = self.onS
      self.thetaTB.onTrackbar = self.onTheta
      self.phiTB.onTrackbar = self.onPhi
      self.v1TB.onTrackbar = self.onV1
      self.v2TB.onTrackbar = self.onV2

   def launchTrackbars(self):
      self.txTB.launch()
      self.tyTB.launch()
      self.l1TB.launch()
      self.l2TB.launch()
      #self.sTB.launch()
      self.thetaTB.launch()
      self.phiTB.launch()
      self.v1TB.launch()
      self.v2TB.launch()

argc = len(sys.argv)

if argc > 1:
   imageName = sys.argv[1]
   image = cv2.imread(imageName)
   print (type(image))
   if type(image) == None:
      sys.stderr.write('\nError: Could not open file "%s"\n\n' % imageName)
else:
   image = None

WIDTH = 640
HEIGHT = 480

version = 0.7071

print("homographyExplorer.")
print("version:", version)
# print("\nusing ", cv2.currentUIFramework(     ), " backend\n")


plane = np.zeros((HEIGHT, WIDTH, 3), dtype=np.uint8)

cv2.namedWindow('Plano',\
   cv2.WINDOW_AUTOSIZE | cv2.WINDOW_KEEPRATIO | cv2.WINDOW_GUI_EXPANDED )

A     = Axes(HEIGHT, WIDTH, 1, 3, 15, 2)
Sq    = Square(A, 128)
fixSq = Square(A, 128)

minTx, midTx, maxTx = 0, WIDTH // (A.ticSize * 2), WIDTH // A.ticSize
minTy, midTy, maxTy = 0, HEIGHT // (A.ticSize * 2), HEIGHT // A.ticSize

if type(image) != None:
   r, c, _ = image.shape
   imageT = np.zeros(image.shape, dtype=np.uint8)
   
   c2 = c//2
   r2 = r//2
   Ht = np.array([1 , 0, -c2, 0, -1, r2, 0, 0, 1]).reshape(3,3)
   iHt = np.array([1 , 0, c2, 0, -1, r2, 0, 0, 1]).reshape(3,3)
   cv2.namedWindow("Imagen", cv2.WINDOW_NORMAL)

hG = homography(WIDTH, HEIGHT)   

hG.change = True
arrowsFlag = True
while True:
   if hG.change == True:
      plane=np.zeros(plane.shape, dtype=np.uint8)
      fixSq.drawSquare(plane, (255,255,0), (255,255,0))
      if arrowsFlag == True:
         for i in range(4):
            p = fixSq.Scale @ fixSq.P[:,i]
            p = p / p[2]
            crn1 = [int(np.round(x)) for x in p[:2]]
            p = Sq.Scale @ Sq.P[:,i]
            p = p / p[2]
            crn2 = [int(np.round(x)) for x in p[:2]]
            cv2.arrowedLine(plane, crn1, crn2, (96,96,96), 3, cv2.LINE_AA, 0, 0.2)
      A.drawAxes(plane)
      Sq.applyHomography(hG.H)
      Sq.drawSquare(plane)
      if type(image) != None:
         warpFlags = cv2.INTER_NEAREST
         cv2.warpPerspective(image, iHt @ hG.H @ Ht, (image.shape[1], image.shape[0]), imageT, warpFlags, cv2.BORDER_CONSTANT, 0)
         cv2.imshow("Imagen", imageT)
         cv2.waitKeyEx(1)
      hG.change = False


   cv2.imshow("Plano", plane)

   key = cv2.waitKeyEx(33) & 0x000000FF
   if (key < 255):
      if key == 27 or chr(key) == 'q' or chr(key) == 'Q':
         break
      elif chr(key) == 'r' or chr(key) == 'R':
         Sq.reset()
         hG.reset()
      elif chr(key) == 'a' or chr(key) == 'A':
         arrowsFlag=not arrowsFlag
         hG.change = True
   
cv2.destroyAllWindows()


