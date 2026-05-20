#! /usr/bin/env python3
#encoding: utf-8

import numpy as np
import cv2 

class trackbar:
   def __init__(self, barName, winName, mn, mx, ini, grain = 100):
      self.barName = barName
      self.winName = winName
      self.minValue = mn
      self.maxValue = mx
      self.initialValue = ini
      self.value = ini
      self.grain = grain
      self.defineTrackbarValues()
      
      self.baseValues = [mn, mx, ini, grain]
      self.val2Pos(self.value)

   def withinBounds(self, val):
      if val >=self.minValue and val <= self.maxValue:
         return True
      return False

   def defineTrackbarValues(self):
      rango = self.maxValue - self.minValue

      propL = (self.initialValue - self.minValue) / rango
      propR = (self.maxValue - self.initialValue) / rango

      nL = int(np.floor(propL * (self.grain + 1)))
      nR = int(np.floor(propR * self.grain))
      
      itL = (self.initialValue - self.minValue) / nL
      itR = (self.maxValue - self.initialValue) / nR
      intervalLeft  = np.arange(self.initialValue, self.minValue - itL, -itL)[:nL + 1][::-1]
      intervalRight = np.arange(self.initialValue + itR, self.maxValue + itR, itR)[:nR + 1]

      self.values = np.concatenate([intervalLeft, intervalRight])
      self.minValue = self.values[0]
      self.maxValue = self.values[-1]
      self.grain = self.values.shape[0]

   def pos2Val(self, pos):
      if pos >= 0 and pos < self.grain:
         self.position = pos
         self.value = self.values[self.position]
      
   def val2Pos(self, val):
      if self.withinBounds(val):
         self.value = val
         if np.isin(val, self.values):
            self.position = np.where(self.values == val)[0][0]
         else:
            diff = np.abs(self.values - self.value)
            closest = np.min(diff)
            self.position = np.where(diff == closest)[0][0]

   def reset(self):
      self.minValue = self.baseValues[0]
      self.maxValue = self.baseValues[1]
      self.initialValue = self.value = self.baseValues[2]
      self.grain = self.baseValues[3]
      self.defineTrackbarValues()
      self.val2Pos(self.value)
      cv2.setTrackbarPos(self.barName, self.winName, self.position)
      self.onTrackbar(self.position)

   def onTrackbar(self, pos): pass 

   def launch(self):
      cv2.createTrackbar(self.barName, self.winName, self.position, self.grain, self.onTrackbar)
      self.onTrackbar(self.position)