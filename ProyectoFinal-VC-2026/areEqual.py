#! /usr/bin/env python3
# encoding: utf-8
"""
This module contains just a function, areEqual.

  areEqual(a, b, ord) compares real quantities a and b, and returns True
  if they are the same up to the ord decimal.

  In other words, it returns True is the if the difference between a and b
  is not significative. the parameter ord, determines how many decimal digits
  should be considered.

  >> areaEqual (1.2356, 1.2357, 5)
  >> False
  >> areaEqual (1.2356, 1.2357, 6)
  >> True
"""

import numpy as np

def areEqual(a, b, ord):
   """
   areEqual(a, b, ord) compares real quantities a and b, and returns True
   if they are the same up to the ord decimal.

   In other words, it returns True is the if the difference between a and b
   is not significative. the parameter ord, determines how many decimal digits
   should be considered.

   >> areaEqual (1.2356, 1.2357, 5)
   >> False
   >> areaEqual (1.2356, 1.2357, 6)
   >> True


   """

   if a == b:
      return True
   if a != 0. and b != 0.:
      val = np.abs((a-b)/max(np.abs([a,b])))
   else:
      val = max(np.abs([a,b]));
   if -np.log10(val) > ord:
      return True;
   return False
