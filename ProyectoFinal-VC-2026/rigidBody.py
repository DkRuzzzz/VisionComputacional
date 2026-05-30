#! /usr/bin/python3
#encoding: utf-8

"""
  This module contains functions useful to represent 3D Rigid Body Motion.
"""

import numpy as np
from areEqual import *


def skew(v):
    """
    Computes the antisymmetric matrix from the vector w

    >> import rigidBody as rb
    >>> w = np.array([1,3,5])
    >>> print(w)
    [1 3 5]
    >>> wHat = rb.skew(w)
    >>> print(wHat)
    [[ 0 -5  3]
     [ 5  0 -1]
     [-3  1  0]]
    """
    skv = np.roll(np.roll(np.diag(v.flatten()),1,1),-1,0)
    return skv -skv.T

def rodriguesRot(v):
    """
    Uses the Rodrigues formula to compute the rotation matrix that corresponds
    to a rotation of ||v|| radians around  the rotation  defined by the normal
    vector v/||v||.

    e.g.

    >>> import rigidBody as rb
    >>>
    >>> theta = np.pi/4
    >>> np.set_printoptions(precision = 4)
    >>>
    >>> # Rotation of pi/4 radians around the x-axis:
    >>> v = theta * np.array([1, 0, 0])
    >>> Rx = rb.rodriguesRot(v)
    >>>
    >>> # Rotation of pi/4 radians around the y-axis:
    >>> v = theta * np.array([0, 1, 0])
    >>> Ry = rb.rodriguesRot(v)
    >>>
    >>> # Rotation of pi/4 radians around the z-axis:
    >>> v = theta * np.array([0, 0, 1])
    >>> Rz = rb.rodriguesRot(v)
    >>>
    >>> print (Rx)
    [[ 1.      0.      0.    ]
     [ 0.      0.7071 -0.7071]
     [ 0.      0.7071  0.7071]]
    >>> print (Ry)
    [[ 0.7071  0.      0.7071]
     [ 0.      1.      0.    ]
     [-0.7071  0.      0.7071]]
    >>> print (Rz)
    [[ 0.7071 -0.7071  0.    ]
     [ 0.7071  0.7071  0.    ]
     [ 0.      0.      1.    ]]
    """
    theta = np.linalg.norm(v)
    v = v / theta
    omega = skew(v)
    return np.eye(3) + np.sin(theta) * omega + (1.-np.cos(theta)) * np.dot(omega,omega)

def logSO3(R):
    """
    Compute the rotation axis w, and the rotation angle theta from the rotation matrix R.


    >>> np.set_printoptions(precision = 4)
    >>> # First we build a rotation matrix that corresponds to a 45 degree rotation
    >>> # around the axis w=[1,2,3]:
    >>> theta = 60 * np.pi / 180
    >>> w = np.array([1, 2, 3])
    >>> w = w / np.linalg.norm(w)
    >>> print ("w     = ", w)
    w     =  [0.2673 0.5345 0.8018]
    >>> print ("theta = ", theta * 180 / np.pi, "\\n")
    theta =  59.99999999999999 

    >>> R = rb.rodriguesRot(theta * w)
    >>> print ("R = \\n", R, "\\n")
    R = 
    [[ 0.5357 -0.6229  0.5701]
     [ 0.7658  0.6429 -0.0172]
     [-0.3558  0.4457  0.8214]] 
     
    >>> # Use logSO3 to recover the original rotation axis and the rotation angle.
    >>> v, gamma = rb.logSO3(R)
    >>> print ("v     = \\n", v)
    v     =  [0.2673 0.5345 0.8018]

    >>> print ("gamma = \\n", gamma * 180 / np.pi, "\\n")
    gamma =  60.00000000000001

    """

    #Validate that R is a Rotation Matrix: i.e. R @ R.R == eye(3) and |R| = +1
    if not areEqual(np.linalg.det(R), 1, 6) or not areEqual(np.linalg.norm(R@R.T-np.eye(3)),0, 6):
        print('Error en logSO3: R no es una matriz de Rotación')
        return None, None

    theta = np.arccos((np.trace(R)-1)/2)
    if theta != 0:
        w = (1/(2*np.sin(theta)))*np.array([[R[2,1]-R[1,2]],[R[0,2]-R[2,0]],[R[1,0]-R[0,1]]])
    else:
        w = np.array([1,0,0]).reshape(3,1) #Un vector arbitrario pues theta = 0
    return w.reshape(3,), float(theta),

def ortGramSchmidt(v1, v2):
    """
    Uses the Gram-Schmidt algorithm to make v1 and v2 ortogonal.

    See [Gram-Schmidt Process](https://en.wikipedia.org/wiki/Gram%E2%80%93Schmidt_process)

    Example

    # Lets define 2 ortonormal vectors v1 and v2
    >>> v1=np.array([1., 0., 0.])
    >>> v2=np.array([0., 1., 0.])
    >>> print ("v1 = ", v1,"\\nv2 = ", v2,"\\n")
    v1 =  [1. 0. 0.] 
    v2 =  [0. 1. 0.] 

    # Lets rotate the vector v2 5 degrees around axis Z = [0, 0, 1].
    >>> rotZ = rb.rodriguesRot(np.array([0, 0, 1]) * 5 * np.pi/180)
    >>> v2 = rotZ @ v2
    >>> print ("v1 = ", v1,"\\nv2 = ", v2,"\\n")
    v1 =  [1. 0. 0.] 
    v2 =  [-0.087  0.996  0.   ] 


    #Lets apply the ortGramSchmidt algorithm.
    >>> w1, w2 = rb.ortGramSchmidt(v1, v2)
    >>> print ("w1 = ", w1,"\\nw2 = ", w2,"\\n")
    w1 =  [1. 0. 0.] 
    w2 =  [0. 1. 0.] 

    """
    v2 = v2 - (v2 @ v1) * v1
    n = v2 @ v2
    if n == 0:
        print("ERROR in ortGramSchmidt")
    v2 = v2 / np.sqrt(n)
    return v1, v2