#! /usr/bin/env python3
#encoding: utf-8

import numpy as np
import cv2

class Axes:
   def __init__(self, r, c, ts=50, tw=3, smTI=5, smTILT = 2):
      self.rows = r
      self.cols = c
      self.ticSize = ts
      self.ticWidth = tw
      self.midX = self.cols // 2
      self.midY = self.rows // 2
      self.xhTics = ((self.cols - 1) // self.ticSize)//2
      self.yhTics = ((self.rows - 1) // self.ticSize)//2
      self.RED = (0,0,255)
      self.smallTickInterval = smTI
      self.smallTicksInLargeTick = smTILT
      self.tics = True

      hTicsA = np.arange(self.midX, 0, -self.ticSize).astype(int)
      hTicsB = np.arange(self.midX + self.ticSize, self.cols - 1, self.ticSize).astype(int)
      vTicsA = np.arange(self.midY, 0,-self.ticSize).astype(int)
      vTicsB = np.arange(self.midY + self.ticSize, self.rows - 1, self.ticSize).astype(int)
      
      P=[]
      fact = 1
      contA = 0
      contB = 0
      for i in vTicsA:
         if contA % self.smallTickInterval == 0:
            fact = 2
            if contB % self.smallTicksInLargeTick == 0:
               fact  = 3
            contB += 1
         else:
            fact = 1
         P.append([self.midX - fact * self.ticWidth, i, self.midX + fact * self.ticWidth, i])
         contA += 1
      fact = 1
      contA = 1
      contB = 0
      for i in vTicsB:
         if contA % self.smallTickInterval == 0:
            fact = 2
            if contB % self.smallTicksInLargeTick == 0:
               fact  = 3
            contB += 1
         else:
            fact = 1
         P.append([self.midX - fact * self.ticWidth, i, self.midX + fact * self.ticWidth, i])
         contA += 1
      fact = 1
      contA = 0
      contB = 0
      for i in hTicsA:
         if contA % self.smallTickInterval == 0:
            fact = 2
            if contB % self.smallTicksInLargeTick == 0:
               fact  = 3
            contB += 1
         else:
            fact = 1
         P.append([i, self.midY - fact * self.ticWidth, i, self.midY + fact * self.ticWidth])
         contA += 1
      fact = 1
      contA = 1
      contB = 0
      for i in hTicsB:
         if contA % self.smallTickInterval == 0:
            fact = 2
            if contB % self.smallTicksInLargeTick == 0:
               fact  = 3
            contB += 1
         else:
            fact = 1
         P.append([i, self.midY - fact * self.ticWidth, i, self.midY + fact * self.ticWidth])
         contA += 1

      self.pTics = np.array(P)
       
   def drawAxes(self, Img):
      cv2.arrowedLine(Img, (0, self.midY), (self.cols - 1, self.midY), self.RED, 1, cv2.LINE_AA, 0, 0.02)
      cv2.arrowedLine(Img, (self.midX, self.rows - 1), (self.midX, 0), self.RED, 1, cv2.LINE_AA, 0, 0.02)
      if self.tics == True:
         for i in range(len(self.pTics)):      
            p0 = (self.pTics[i,0], self.pTics[i,1])
            p1 = (self.pTics[i,2], self.pTics[i,3])
            cv2.line(Img, p0, p1, self.RED, 1)